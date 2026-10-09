from pathlib import Path
import json
import pickle
import platform
import shutil
import sys

import numpy as np
import pandas as pd
import scipy

from scripts.run_main import _block_bootstrap_upper, _controls, _slice_envelope, _verify_hash
from src.data_pipeline import file_sha256, load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope


METHODS = ("DYNAMIC", "MEAN", "SHIFT_7D", "PERMUTED_DAYS")
NEW_METHODS = ("SHIFT_7D", "PERMUTED_DAYS")
SEED = 20260802


def signal_source_indices(day, method, permutation):
    target = np.arange(day * 24, (day + 1) * 24)
    previous = (target - 24) % 8760
    if method == "DYNAMIC":
        return previous
    if method == "SHIFT_7D":
        return (previous - 168) % 8760
    if method == "PERMUTED_DAYS":
        return permutation[day] * 24 + np.arange(24)
    raise ValueError(method)


def _run_day(day, frame, envelope, parameters, cap, permutation):
    indices = np.arange(day * 24, (day + 1) * 24)
    previous = (indices - 24) % 8760
    actual = frame.iloc[indices].reset_index(drop=True)
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    dispatch = {}
    solver = {}
    sources = {}
    for method in NEW_METHODS:
        source = signal_source_indices(day, method, permutation)
        forecast = actual.copy()
        forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
        forecast["carbon_kg_per_kwh"] = frame.iloc[source]["carbon_kg_per_kwh"].to_numpy()
        try:
            plan = solve_day(forecast, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap)
        except RuntimeError as error:
            raise RuntimeError(f"day {day} method {method} failed: {error}") from error
        dispatch[method] = replay_day(actual, actual_envelope, parameters, _controls(plan.frame))
        solver[method] = plan.solver
        sources[method] = {"first_hour": int(source[0]), "last_hour": int(source[-1])}
    return {"day": day, "dispatch": dispatch, "solver": solver, "signal_source": sources}


def _daily_metrics(frame, method):
    grid_net = frame["grid_import_kw"] - frame["grid_export_delivered_kw"]
    result = pd.DataFrame({
        "day": frame.index // 24,
        "cost_usd": frame["cost_usd_per_kwh"] * grid_net,
        "emissions_kgco2e": frame["carbon_kg_per_kwh"] * grid_net,
    }).groupby("day", as_index=False).sum()
    result["method"] = method
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    shapley_root = root / "artifacts" / "experiment" / "shapley_relative_2026-08-02"
    shapley = json.loads((shapley_root / "shapley_summary.json").read_text(encoding="utf-8"))
    if not shapley["passed"]:
        raise RuntimeError("verified coalition source did not pass")
    source_records = json.loads((shapley_root / "solver_records.json").read_text(encoding="utf-8"))
    if len(source_records) != 365:
        raise RuntimeError("reference denominator is not 365 days")

    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    permutation = np.random.default_rng(SEED).permutation(365)
    output = root / "artifacts" / "experiment" / "adversarial_signals_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            completed = pickle.load(handle)
    else:
        completed = []

    for day in range(len(completed), 365):
        cap = source_records[day]["references"]["B1C1"]["planned_cost_cap_usd"]
        try:
            completed.append(_run_day(day, inputs.frame, envelope, parameters, cap, permutation))
        except Exception as error:
            failure = {"day": day, "completed_days": len(completed), "error": str(error)}
            (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        if (day + 1) % 5 == 0 or day == 364:
            temporary = output / "checkpoint.tmp"
            with temporary.open("wb") as handle:
                pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(checkpoint)
            print(f"completed_days={day + 1}/365", flush=True)

    sources = {
        "DYNAMIC": shapley_root / "A1B1C1_dispatch.csv",
        "MEAN": shapley_root / "B2_dispatch.csv",
    }
    for method, source in sources.items():
        shutil.copy2(source, output / f"{method}_dispatch.csv")
    for method in NEW_METHODS:
        pd.concat([item["dispatch"][method] for item in completed], ignore_index=True).to_csv(output / f"{method}_dispatch.csv", index=False)

    metrics = {}
    frames = {}
    daily_parts = []
    integrity = {}
    for method in METHODS:
        frame = pd.read_csv(output / f"{method}_dispatch.csv")
        frames[method] = frame
        metrics[method] = evaluate_dispatch(frame)
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        integrity[method] = {"rows": len(frame), "nonfinite_values": int((~np.isfinite(numeric)).sum())}
        daily_parts.append(_daily_metrics(frame, method))
    daily = pd.concat(daily_parts, ignore_index=True)
    daily.to_csv(output / "daily_metrics.csv", index=False)

    dynamic_daily = daily[daily.method == "DYNAMIC"].sort_values("day")
    comparisons = {}
    for method in METHODS[1:]:
        other = daily[daily.method == method].sort_values("day")
        difference = other.emissions_kgco2e.to_numpy() - dynamic_daily.emissions_kgco2e.to_numpy()
        comparisons[method] = {
            "method_minus_dynamic_cost_fraction": metrics[method]["annual_operating_cost_usd"] / metrics["DYNAMIC"]["annual_operating_cost_usd"] - 1,
            "method_minus_dynamic_emissions_fraction": metrics[method]["annual_emissions_kgco2e"] / metrics["DYNAMIC"]["annual_emissions_kgco2e"] - 1,
            "method_minus_dynamic_daily_emissions_95ci_kg": list(_block_bootstrap_upper(difference)),
        }

    carbon = inputs.frame["carbon_kg_per_kwh"].to_numpy()
    planned = {
        method: np.concatenate([carbon[signal_source_indices(day, method, permutation)] for day in range(365)])
        for method in ("DYNAMIC", "SHIFT_7D", "PERMUTED_DAYS")
    }
    planned["MEAN"] = np.full(8760, carbon.mean())
    signal_audit = {}
    for method, values in planned.items():
        correlation = None if method == "MEAN" else float(np.corrcoef(values, carbon)[0, 1])
        signal_audit[method] = {
            "forecast_realized_pearson_r": correlation,
            "forecast_realized_rmse_kg_per_kwh": float(np.sqrt(np.mean((values - carbon) ** 2))),
            "same_sorted_distribution_as_dynamic": bool(np.array_equal(np.sort(values), np.sort(planned["DYNAMIC"]))) if method != "MEAN" else False,
        }
    pd.DataFrame({"target_hour": np.arange(8760), **planned}).to_csv(output / "planned_carbon_signals.csv", index=False)

    replacement_status = {}
    for method in NEW_METHODS:
        comparison = comparisons[method]
        point = comparison["method_minus_dynamic_emissions_fraction"]
        lower, upper = comparison["method_minus_dynamic_daily_emissions_95ci_kg"]
        if point < 0:
            replacement_status[method] = "refutes_positive_timing_claim"
        elif lower > 0:
            replacement_status[method] = "supports_timing_alignment"
        else:
            replacement_status[method] = "timing_inconclusive"
    if any(value == "refutes_positive_timing_claim" for value in replacement_status.values()):
        timing_conclusion = "refuted"
    elif all(value == "supports_timing_alignment" for value in replacement_status.values()):
        timing_conclusion = "supported_within_matched_replacements"
    else:
        timing_conclusion = "inconclusive"

    source_hashes = {method: file_sha256(path).lower() for method, path in sources.items()}
    attached_hashes = {method: file_sha256(output / f"{method}_dispatch.csv").lower() for method in sources}
    solver_leaves = [item["solver"][method] for item in completed for method in NEW_METHODS]
    physical = all(
        value["energy_closure_relative_error"] <= 0.005
        and value["carbon_closure_relative_error"] <= 0.005
        and value["comfort_violation_hours"] == 0
        for value in metrics.values()
    )
    gates = {
        "E1_attachment_identity": source_hashes == attached_hashes,
        "E1_complete_denominator": len(completed) == 365 and all(value["rows"] == 8760 for value in integrity.values()),
        "E1_finite": all(value["nonfinite_values"] == 0 for value in integrity.values()),
        "E1_physical": physical,
        "E1_new_solvers_optimal": len(solver_leaves) == 730 and all(value["status"] == 0 for value in solver_leaves),
        "E1_distribution_invariants": all(signal_audit[method]["same_sorted_distribution_as_dynamic"] for method in NEW_METHODS),
    }
    summary = {
        "status": "matched_signal_replacements",
        "hours": 8760,
        "methods": METHODS,
        "equal_compute_methods": ("DYNAMIC", "SHIFT_7D", "PERMUTED_DAYS"),
        "seed": SEED,
        "metrics": metrics,
        "comparisons": comparisons,
        "signal_audit": signal_audit,
        "replacement_status": replacement_status,
        "timing_conclusion": timing_conclusion,
        "integrity": integrity,
        "new_plan_solves": len(solver_leaves),
        "nonoptimal_new_solver_records": sum(value["status"] != 0 for value in solver_leaves),
        "source_sha256": source_hashes,
        "attached_sha256": attached_hashes,
        "gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "command": "python -m scripts.run_adversarial_signals",
    }
    (output / "signal_ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "solver_records.json").write_text(json.dumps([
        {"day": item["day"], "signal_source": item["signal_source"], "solver": item["solver"]}
        for item in completed
    ], ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "replacement_status": replacement_status, "timing_conclusion": timing_conclusion, "gates": gates}, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("E1 completion gate failed")


if __name__ == "__main__":
    main()
