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
from scripts.run_pareto import _reference_cost_from_f5_cap
from src.data_pipeline import file_sha256, load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch, exact_shapley, harsanyi_dividends
from src.model import build_hvac_envelope


COALITIONS = [f"A{a}B{b}C{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]
FULL = "A1B1C1"


def _members(coalition):
    return frozenset(feature for feature, enabled in zip(("A", "B", "C"), (coalition[1], coalition[3], coalition[5])) if enabled == "1")


def _resource_key(coalition):
    return f"B{coalition[3]}C{coalition[5]}"


def _relative_cap(reference_cost):
    return reference_cost + 0.04 * abs(reference_cost)


def _run_day(day_number, frame, envelope, parameters, full_reference_cost, annual_carbon_mean, coalitions, include_b2):
    indices = np.arange(day_number * 24, (day_number + 1) * 24)
    previous = (indices - 24) % 8760
    actual = frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
    forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    resource_keys = {_resource_key(coalition) for coalition in coalitions}
    if include_b2:
        resource_keys.add("B1C1")
    references = {}
    reference_solver = {}
    for key in sorted(resource_keys):
        b, c = key[1] == "1", key[3] == "1"
        try:
            result = solve_day(
                forecast,
                forecast_envelope,
                parameters,
                objective="cost",
                hvac_flexible=b,
                bess_flexible=c,
            )
        except RuntimeError as error:
            raise RuntimeError(f"day {day_number} economic reference {key} failed: {error}") from error
        chosen = full_reference_cost if key == "B1C1" else result.cost_usd
        difference = abs(result.cost_usd - chosen)
        if key == "B1C1" and difference > 1e-8:
            raise RuntimeError(f"day {day_number} attached full-resource reference differs by {difference}")
        references[key] = {
            "recomputed_cost_usd": result.cost_usd,
            "used_cost_usd": chosen,
            "absolute_attachment_difference_usd": difference,
            "planned_cost_cap_usd": _relative_cap(chosen),
        }
        reference_solver[key] = result.solver
    dispatch = {}
    coalition_solver = {}
    for coalition in coalitions:
        a, b, c = coalition[1] == "1", coalition[3] == "1", coalition[5] == "1"
        try:
            plan = solve_day(
                forecast,
                forecast_envelope,
                parameters,
                objective="emissions" if a else "emissions_lossless",
                cost_cap_usd=references[_resource_key(coalition)]["planned_cost_cap_usd"],
                hvac_flexible=b,
                bess_flexible=c,
            )
        except RuntimeError as error:
            raise RuntimeError(f"day {day_number} coalition {coalition} failed: {error}") from error
        dispatch[coalition] = replay_day(actual, actual_envelope, parameters, _controls(plan.frame))
        coalition_solver[coalition] = plan.solver
    b2_solver = None
    if include_b2:
        average = forecast.copy()
        average["carbon_kg_per_kwh"] = annual_carbon_mean
        try:
            b2 = solve_day(
                average,
                forecast_envelope,
                parameters,
                objective="emissions",
                cost_cap_usd=references["B1C1"]["planned_cost_cap_usd"],
            )
        except RuntimeError as error:
            raise RuntimeError(f"day {day_number} comparator B2 failed: {error}") from error
        dispatch["B2"] = replay_day(actual, actual_envelope, parameters, _controls(b2.frame))
        b2_solver = b2.solver
    return {
        "day": day_number,
        "dispatch": dispatch,
        "references": references,
        "solver": {"economic_reference": reference_solver, "coalition": coalition_solver, "B2": b2_solver},
    }


def _metric_utility(metrics, key):
    return {_members(coalition): -metrics[coalition][key] for coalition in COALITIONS}


def _attribution(metrics, key):
    values = _metric_utility(metrics, key)
    shapley = exact_shapley(values)
    dividends = harsanyi_dividends(values)
    total = values[frozenset(("A", "B", "C"))] - values[frozenset()]
    closure = abs(sum(shapley.values()) - total) / max(abs(total), 1e-12)
    return {"shapley": shapley, "harsanyi_dividends": dividends, "full_minus_empty_utility": total, "closure_relative_error": closure}


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    pareto_root = root / "artifacts" / "experiment" / "pareto_2026-08-02"
    pareto = json.loads((pareto_root / "pareto_summary.json").read_text(encoding="utf-8"))
    if not pareto["passed_for_shapley"] or pareto["practical_policy"] != "F4":
        raise RuntimeError("F4 Shapley gate is not open")
    smoke = json.loads((root / "artifacts" / "experiment" / "shapley_smoke_2026-08-02" / "smoke_summary.json").read_text(encoding="utf-8"))
    if smoke["contract_sha256"] != contract_hash or not smoke["gates"]["C0_all_coalitions_feasible"] or not smoke["gates"]["C1_F4_identity"]:
        raise RuntimeError("coalition smoke gate is not open")
    source_records = json.loads((root / "artifacts" / "experiment" / "main_2026-08-02" / "solver_records.json").read_text(encoding="utf-8"))
    full_reference_costs = [_reference_cost_from_f5_cap(record["planned_cost_cap"]) for record in source_records]
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    annual_carbon_mean = float(inputs.frame["carbon_kg_per_kwh"].mean())
    output = root / "artifacts" / "experiment" / "shapley_relative_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            completed = pickle.load(handle)
    else:
        completed = []
    annual_coalitions = [coalition for coalition in COALITIONS if coalition != FULL]
    for day_number in range(len(completed), 365):
        try:
            item = _run_day(
                day_number,
                inputs.frame,
                envelope,
                parameters,
                full_reference_costs[day_number],
                annual_carbon_mean,
                annual_coalitions,
                True,
            )
        except Exception as error:
            failure = {"day": day_number, "error": str(error), "completed_days": len(completed)}
            (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        completed.append(item)
        if (day_number + 1) % 5 == 0 or day_number == 364:
            with checkpoint.open("wb") as handle:
                pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"completed_days={day_number + 1}/365", flush=True)
    shutil.copy2(pareto_root / "F4_dispatch.csv", output / f"{FULL}_dispatch.csv")
    for method in annual_coalitions + ["B2"]:
        pd.concat([item["dispatch"][method] for item in completed], ignore_index=True).to_csv(output / f"{method}_dispatch.csv", index=False)
    metrics = {}
    daily_rows = []
    integrity = {}
    for method in COALITIONS + ["B2"]:
        dispatch = pd.read_csv(output / f"{method}_dispatch.csv")
        metrics[method] = evaluate_dispatch(dispatch)
        numeric = dispatch.select_dtypes(include=[np.number]).to_numpy()
        integrity[method] = {"rows": len(dispatch), "nonfinite_values": int((~np.isfinite(numeric)).sum())}
        grid_net = dispatch["grid_import_kw"] - dispatch["grid_export_delivered_kw"]
        daily = pd.DataFrame({
            "day": dispatch.index // 24,
            "cost_usd": dispatch["cost_usd_per_kwh"] * grid_net,
            "emissions_kgco2e": dispatch["carbon_kg_per_kwh"] * grid_net,
        }).groupby("day", as_index=False).sum()
        daily["method"] = method
        daily_rows.append(daily)
    daily_metrics = pd.concat(daily_rows, ignore_index=True)
    daily_metrics.to_csv(output / "daily_metrics.csv", index=False)
    attribution = {
        "emissions_utility_kgco2e": _attribution(metrics, "annual_emissions_kgco2e"),
        "cost_utility_usd": _attribution(metrics, "annual_operating_cost_usd"),
    }
    comparator = {}
    full_daily = daily_metrics[daily_metrics.method == FULL].sort_values("day")
    for method in ("B2", "A0B1C1"):
        other_daily = daily_metrics[daily_metrics.method == method].sort_values("day")
        ci = _block_bootstrap_upper(full_daily.emissions_kgco2e.to_numpy() - other_daily.emissions_kgco2e.to_numpy())
        comparator[method] = {
            "full_minus_method_cost_fraction": metrics[FULL]["annual_operating_cost_usd"] / metrics[method]["annual_operating_cost_usd"] - 1,
            "full_minus_method_emissions_fraction": metrics[FULL]["annual_emissions_kgco2e"] / metrics[method]["annual_emissions_kgco2e"] - 1,
            "full_minus_method_daily_emissions_95ci_kg": list(ci),
        }
    physical = all(
        value["energy_closure_relative_error"] <= 0.005
        and value["carbon_closure_relative_error"] <= 0.005
        and value["comfort_violation_hours"] == 0
        for value in metrics.values()
    )
    pareto_verification = json.loads((pareto_root / "verification_report.json").read_text(encoding="utf-8"))
    expected_f4_hash = pareto_verification["sha256"]["F4_dispatch.csv"].lower()
    attached_f4_hash = file_sha256(output / f"{FULL}_dispatch.csv").lower()
    solver_records = [
        {"day": item["day"], "references": item["references"], "solver": item["solver"]}
        for item in completed
    ]
    solver_leaves = []
    for item in completed:
        solver_leaves.extend(item["solver"]["economic_reference"].values())
        solver_leaves.extend(item["solver"]["coalition"].values())
        solver_leaves.append(item["solver"]["B2"])
    pareto_records = json.loads((pareto_root / "solver_records.json").read_text(encoding="utf-8"))
    attached_full_solvers = [record["solver"]["F4"]["plan"] for record in pareto_records]
    all_solvers = solver_leaves + attached_full_solvers
    complete = (
        len(completed) == 365
        and len(COALITIONS) == 8
        and all(value["rows"] == 8760 and value["nonfinite_values"] == 0 for value in integrity.values())
        and all(solver is not None and solver["status"] == 0 for solver in all_solvers)
    )
    gates = {
        "C0_all_coalitions_feasible": smoke["gates"]["C0_all_coalitions_feasible"],
        "C1_F4_identity": smoke["gates"]["C1_F4_identity"] and attached_f4_hash == expected_f4_hash,
        "C2_physical": physical,
        "C3_emissions_shapley_closure": attribution["emissions_utility_kgco2e"]["closure_relative_error"] <= 0.005,
        "C3_cost_shapley_closure": attribution["cost_utility_usd"]["closure_relative_error"] <= 0.005,
        "C4_complete_denominator": complete,
    }
    with (output / "solver_records.json").open("w", encoding="utf-8") as handle:
        json.dump(solver_records, handle, ensure_ascii=False)
    summary = {
        "status": "relative_budget_exact_shapley",
        "hours": 8760,
        "practical_policy": "F4",
        "coalitions": COALITIONS,
        "features": {"A": "loss-aware carbon objective", "B": "HVAC flexibility", "C": "BESS flexibility"},
        "estimand": "relative-budget mechanism attribution with 4% above each resource set's own economic minimum",
        "attached_read_only": {FULL: "pareto_2026-08-02/F4_dispatch.csv"},
        "F4_sha256": {"expected": expected_f4_hash, "attached": attached_f4_hash},
        "integrity": integrity,
        "new_solver_records": len(solver_leaves),
        "attached_full_solver_records": len(attached_full_solvers),
        "nonoptimal_solver_records": sum(solver["status"] != 0 for solver in all_solvers),
        "maximum_full_reference_attachment_difference_usd": max(
            item["references"]["B1C1"]["absolute_attachment_difference_usd"] for item in completed
        ),
        "metrics": metrics,
        "matched_comparators": comparator,
        "attribution": attribution,
        "gates": gates,
        "passed": all(gates.values()),
        "evaluation_summary": {
            "outcome_summary": "eight-coalition relative-budget exact Shapley and F4 matched comparators evaluated",
            "claim_update": "supported" if all(gates.values()) else "inconclusive",
            "baseline_relation": "each resource set uses 4% above its own economic minimum; full coalition is attached byte-for-byte from verified F4",
            "failure_mode": None if all(gates.values()) else "evaluation_pipeline_failure",
            "next_action": "run_bounded_sensitivity" if all(gates.values()) else "inspect_attribution_failure",
            "comparability": "same annual denominator, data, forecast, physical model and daily cadence",
        },
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "command": "python -m scripts.run_shapley",
    }
    with (output / "shapley_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"matched_comparators": comparator, "attribution": attribution, "gates": gates, "passed": summary["passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
