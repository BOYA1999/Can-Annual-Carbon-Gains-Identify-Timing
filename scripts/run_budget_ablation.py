from pathlib import Path
import argparse
import json
import pickle
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from scripts.run_main import _controls, _slice_envelope, _verify_hash
from scripts.run_shapley import COALITIONS, _attribution, _resource_key
from src.data_pipeline import load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope


ALLOWED = (0.01, 0.08)


def relative_cap(reference_cost, allowance):
    return reference_cost + allowance * abs(reference_cost)


def _run_day(day, allowance, frame, envelope, parameters, source_record):
    indices = np.arange(day * 24, (day + 1) * 24)
    previous = (indices - 24) % 8760
    actual = frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
    forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    dispatch = {}
    solver = {}
    caps = {}
    for coalition in COALITIONS:
        a, b, c = coalition[1] == "1", coalition[3] == "1", coalition[5] == "1"
        reference = source_record["references"][_resource_key(coalition)]["used_cost_usd"]
        cap = relative_cap(reference, allowance)
        try:
            plan = solve_day(
                forecast,
                forecast_envelope,
                parameters,
                objective="emissions" if a else "emissions_lossless",
                cost_cap_usd=cap,
                hvac_flexible=b,
                bess_flexible=c,
            )
        except RuntimeError as error:
            raise RuntimeError(f"day {day} coalition {coalition} failed at allowance {allowance}: {error}") from error
        dispatch[coalition] = replay_day(actual, actual_envelope, parameters, _controls(plan.frame))
        solver[coalition] = plan.solver
        caps[coalition] = {"reference_cost_usd": reference, "planned_cost_cap_usd": cap}
    return {"day": day, "dispatch": dispatch, "solver": solver, "caps": caps}


def _run_allowance(root, allowance, contract_hash, parameter_hash):
    tag = f"{int(round(allowance * 100)):02d}"
    source_root = root / "artifacts" / "experiment" / "shapley_relative_2026-08-02"
    source_summary = json.loads((source_root / "shapley_summary.json").read_text(encoding="utf-8"))
    source_records = json.loads((source_root / "solver_records.json").read_text(encoding="utf-8"))
    if not source_summary["passed"] or len(source_records) != 365:
        raise RuntimeError("verified 4% source is unavailable")
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    output = root / "artifacts" / "experiment" / f"adversarial_budget_{tag}_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            completed = pickle.load(handle)
    else:
        completed = []

    for day in range(len(completed), 365):
        try:
            completed.append(_run_day(day, allowance, inputs.frame, envelope, parameters, source_records[day]))
        except Exception as error:
            failure = {"allowance": allowance, "day": day, "completed_days": len(completed), "error": str(error)}
            (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        if (day + 1) % 5 == 0 or day == 364:
            temporary = output / "checkpoint.tmp"
            with temporary.open("wb") as handle:
                pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(checkpoint)
            print(f"allowance={allowance:.2f} completed_days={day + 1}/365", flush=True)

    metrics = {}
    integrity = {}
    daily_parts = []
    for coalition in COALITIONS:
        dispatch = pd.concat([item["dispatch"][coalition] for item in completed], ignore_index=True)
        dispatch.to_csv(output / f"{coalition}_dispatch.csv", index=False)
        metrics[coalition] = evaluate_dispatch(dispatch)
        numeric = dispatch.select_dtypes(include=[np.number]).to_numpy()
        integrity[coalition] = {"rows": len(dispatch), "nonfinite_values": int((~np.isfinite(numeric)).sum())}
        grid_net = dispatch["grid_import_kw"] - dispatch["grid_export_delivered_kw"]
        daily = pd.DataFrame({
            "day": dispatch.index // 24,
            "cost_usd": dispatch["cost_usd_per_kwh"] * grid_net,
            "emissions_kgco2e": dispatch["carbon_kg_per_kwh"] * grid_net,
        }).groupby("day", as_index=False).sum()
        daily["coalition"] = coalition
        daily_parts.append(daily)
    pd.concat(daily_parts, ignore_index=True).to_csv(output / "daily_metrics.csv", index=False)
    attribution = {
        "emissions_utility_kgco2e": _attribution(metrics, "annual_emissions_kgco2e"),
        "cost_utility_usd": _attribution(metrics, "annual_operating_cost_usd"),
    }
    solver_leaves = [item["solver"][coalition] for item in completed for coalition in COALITIONS]
    physical = all(
        value["energy_closure_relative_error"] <= 0.005
        and value["carbon_closure_relative_error"] <= 0.005
        and value["comfort_violation_hours"] == 0
        for value in metrics.values()
    )
    gates = {
        "E2_complete_denominator": len(completed) == 365 and all(value["rows"] == 8760 for value in integrity.values()),
        "E2_finite": all(value["nonfinite_values"] == 0 for value in integrity.values()),
        "E2_physical": physical,
        "E2_solvers_optimal": len(solver_leaves) == 2920 and all(value["status"] == 0 for value in solver_leaves),
        "E2_emissions_shapley_closure": attribution["emissions_utility_kgco2e"]["closure_relative_error"] <= 0.005,
        "E2_cost_shapley_closure": attribution["cost_utility_usd"]["closure_relative_error"] <= 0.005,
    }
    solver_records = [{"day": item["day"], "caps": item["caps"], "solver": item["solver"]} for item in completed]
    (output / "solver_records.json").write_text(json.dumps(solver_records, ensure_ascii=False), encoding="utf-8")
    summary = {
        "status": "complete_coalition_budget_ablation",
        "allowance": allowance,
        "hours": 8760,
        "coalitions": COALITIONS,
        "features": source_summary["features"],
        "economic_references": "attached from verified daily resource-set minima",
        "metrics": metrics,
        "attribution": attribution,
        "integrity": integrity,
        "new_plan_solves": len(solver_leaves),
        "nonoptimal_solver_records": sum(value["status"] != 0 for value in solver_leaves),
        "gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "command": f"python -m scripts.run_budget_ablation --allowance {allowance}",
    }
    (output / "budget_ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"allowance": allowance, "attribution": attribution, "gates": gates, "passed": summary["passed"]}, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError(f"E2 completion gate failed at allowance {allowance}")


def _summarize(root, contract_hash, parameter_hash):
    source = json.loads((root / "artifacts" / "experiment" / "shapley_relative_2026-08-02" / "shapley_summary.json").read_text(encoding="utf-8"))
    summaries = {0.04: source}
    for allowance in ALLOWED:
        tag = f"{int(round(allowance * 100)):02d}"
        path = root / "artifacts" / "experiment" / f"adversarial_budget_{tag}_2026-08-02" / "budget_ablation_summary.json"
        summaries[allowance] = json.loads(path.read_text(encoding="utf-8"))
    if not all(summary["passed"] for summary in summaries.values()):
        raise RuntimeError("one or more budget tables failed completion gates")

    shapley_rows = []
    interaction_rows = []
    for allowance in sorted(summaries):
        attribution = summaries[allowance]["attribution"]["emissions_utility_kgco2e"]
        total = attribution["full_minus_empty_utility"]
        for feature, value in attribution["shapley"].items():
            shapley_rows.append({"allowance_fraction": allowance, "feature": feature, "shapley_kgco2e": value, "share_fraction": value / total})
        for term, value in attribution["harsanyi_dividends"].items():
            interaction_rows.append({"allowance_fraction": allowance, "term": term, "dividend_kgco2e": value})
    shapley_frame = pd.DataFrame(shapley_rows)
    interaction_frame = pd.DataFrame(interaction_rows)
    output = root / "artifacts" / "experiment" / "adversarial_budget_curve_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    shapley_frame.to_csv(output / "shapley_budget_curve.csv", index=False)
    interaction_frame.to_csv(output / "interaction_budget_curve.csv", index=False)

    dominant = {
        str(allowance): shapley_frame[shapley_frame.allowance_fraction == allowance].iloc[
            shapley_frame[shapley_frame.allowance_fraction == allowance].shapley_kgco2e.abs().argmax()
        ].feature
        for allowance in sorted(summaries)
    }
    sign_changes = {
        feature: len(set(np.sign(shapley_frame[shapley_frame.feature == feature].shapley_kgco2e))) > 1
        for feature in ("A", "B", "C")
    }
    combined = {
        "status": "budget_sensitivity_curve",
        "allowances": sorted(summaries),
        "dominant_feature": dominant,
        "same_dominant_feature_across_allowances": len(set(dominant.values())) == 1,
        "feature_sign_changes": sign_changes,
        "attribution": {
            str(allowance): summaries[allowance]["attribution"]
            for allowance in sorted(summaries)
        },
        "source_4pct": "shapley_relative_2026-08-02/shapley_summary.json",
        "gates": {
            "E2_all_tables_passed": all(summary["passed"] for summary in summaries.values()),
            "E2_three_prespecified_allowances": sorted(summaries) == [0.01, 0.04, 0.08],
            "E2_curve_finite": bool(np.isfinite(shapley_frame.select_dtypes(include=[np.number]).to_numpy()).all()),
        },
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "command": "python -m scripts.run_budget_ablation --summarize",
    }
    combined["passed"] = all(combined["gates"].values())
    (output / "budget_curve_summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: combined[key] for key in ("dominant_feature", "same_dominant_feature_across_allowances", "feature_sign_changes", "gates", "passed")}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowance", type=float, choices=ALLOWED)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if (args.allowance is None) == (not args.summarize):
        parser.error("choose exactly one of --allowance or --summarize")
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    if args.summarize:
        _summarize(root, contract_hash, parameter_hash)
    else:
        _run_allowance(root, args.allowance, contract_hash, parameter_hash)


if __name__ == "__main__":
    main()
