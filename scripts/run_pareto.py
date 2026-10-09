from pathlib import Path
import json
import pickle
import platform
import shutil
import sys

import numpy as np
import pandas as pd
import scipy

from scripts.run_hourly_audit import POLICIES, _plan
from scripts.run_main import _block_bootstrap_upper, _controls, _slice_envelope, _verify_hash
from src.data_pipeline import file_sha256, load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import distinct_nondominated_count, evaluate_dispatch, pareto_status, select_knee, select_practical_policy
from src.model import build_hvac_envelope


NEW_POLICIES = {policy: allowance for policy, allowance in POLICIES.items() if policy not in ("F0", "F5")}


def _reference_cost_from_f5_cap(cap):
    return cap / (1.05 if cap >= 0 else 0.95)


def _run_day(day_number, frame, envelope, parameters, reference_cost):
    indices = np.arange(day_number * 24, (day_number + 1) * 24)
    previous = (indices - 24) % 8760
    actual = frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
    forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    result = {}
    solver = {}
    for policy, allowance in NEW_POLICIES.items():
        if np.isinf(allowance):
            plan, extra = _plan(forecast, forecast_envelope, parameters, allowance)
            cap = None
        else:
            cap = reference_cost + allowance * abs(reference_cost)
            plan = solve_day(forecast, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap)
            extra = None
        result[policy] = replay_day(actual, actual_envelope, parameters, _controls(plan.frame))
        solver[policy] = {"plan": plan.solver, "planned_cost_cap_usd": cap, "extra": extra}
    return {"day": day_number, "dispatch": result, "solver": solver, "reference_cost_usd": reference_cost}


def _verify_attached_reference(frame, envelope, parameters, source_records):
    checks = []
    for day_number in (0, 182, 364):
        indices = np.arange(day_number * 24, (day_number + 1) * 24)
        previous = (indices - 24) % 8760
        forecast = frame.iloc[indices].reset_index(drop=True).copy()
        forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
        forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
        forecast_envelope = _slice_envelope(envelope, indices, previous)
        result = solve_day(forecast, forecast_envelope, parameters, objective="cost")
        attached = _reference_cost_from_f5_cap(source_records[day_number]["planned_cost_cap"])
        difference = abs(result.cost_usd - attached)
        checks.append({"day": day_number, "recomputed_cost_usd": result.cost_usd, "attached_cost_usd": attached, "absolute_difference": difference})
        if difference > 1e-8:
            raise RuntimeError(f"attached B1 plan cost mismatch on day {day_number}")
    return checks


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    p0 = json.loads((root / "artifacts" / "experiment" / "p0_hourly_audit_2026-08-02" / "p0_summary.json").read_text(encoding="utf-8"))
    if not p0["passed"]:
        raise RuntimeError("P0 did not pass")
    source = root / "artifacts" / "experiment" / "main_2026-08-02"
    source_records = json.loads((source / "solver_records.json").read_text(encoding="utf-8"))
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    reference_checks = _verify_attached_reference(inputs.frame, envelope, parameters, source_records)
    reference_costs = [_reference_cost_from_f5_cap(record["planned_cost_cap"]) for record in source_records]
    output = root / "artifacts" / "experiment" / "pareto_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            completed = pickle.load(handle)
    else:
        completed = []
    for day_number in range(len(completed), 365):
        completed.append(_run_day(day_number, inputs.frame, envelope, parameters, reference_costs[day_number]))
        if (day_number + 1) % 5 == 0 or day_number == 364:
            with checkpoint.open("wb") as handle:
                pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"completed_days={day_number + 1}/365", flush=True)
    shutil.copy2(source / "B1_dispatch.csv", output / "F0_dispatch.csv")
    shutil.copy2(source / "P1_dispatch.csv", output / "F5_dispatch.csv")
    for policy in NEW_POLICIES:
        pd.concat([item["dispatch"][policy] for item in completed], ignore_index=True).to_csv(output / f"{policy}_dispatch.csv", index=False)
    metrics = {}
    daily_rows = []
    for policy in POLICIES:
        dispatch = pd.read_csv(output / f"{policy}_dispatch.csv")
        metrics[policy] = evaluate_dispatch(dispatch)
        grid_net = dispatch["grid_import_kw"] - dispatch["grid_export_delivered_kw"]
        daily = pd.DataFrame({
            "day": dispatch.index // 24,
            "cost_usd": dispatch["cost_usd_per_kwh"] * grid_net,
            "emissions_kgco2e": dispatch["carbon_kg_per_kwh"] * grid_net,
        }).groupby("day", as_index=False).sum()
        daily["policy"] = policy
        daily_rows.append(daily)
    daily_metrics = pd.concat(daily_rows, ignore_index=True)
    daily_metrics.to_csv(output / "daily_metrics.csv", index=False)
    nondominated = pareto_status(metrics)
    distinct_count = distinct_nondominated_count(metrics, nondominated)
    practical = select_practical_policy(metrics)
    knee = select_knee(metrics, nondominated)
    comparisons = {}
    baseline_daily = daily_metrics[daily_metrics.policy == "F0"].sort_values("day")
    for policy in POLICIES:
        policy_daily = daily_metrics[daily_metrics.policy == policy].sort_values("day")
        ci = _block_bootstrap_upper(policy_daily.emissions_kgco2e.to_numpy() - baseline_daily.emissions_kgco2e.to_numpy())
        comparisons[policy] = {
            "cost_fraction_vs_F0": metrics[policy]["annual_operating_cost_usd"] / metrics["F0"]["annual_operating_cost_usd"] - 1,
            "emissions_fraction_vs_F0": metrics[policy]["annual_emissions_kgco2e"] / metrics["F0"]["annual_emissions_kgco2e"] - 1,
            "daily_emissions_difference_block_bootstrap_95ci_kg": list(ci),
            "nondominated": nondominated[policy],
        }
    practical_comparison = comparisons[practical] if practical else None
    gates = {
        "P0_cadence": True,
        "P1_physical": all(
            value["energy_closure_relative_error"] <= 0.005
            and value["carbon_closure_relative_error"] <= 0.005
            and value["comfort_violation_hours"] == 0
            for value in metrics.values()
        ),
        "P2_frontier": distinct_count >= 4,
        "P3_practical_policy_exists": practical is not None,
        "P3_emissions_reduction_at_least_2pct": practical is not None and practical_comparison["emissions_fraction_vs_F0"] <= -0.02,
        "P3_emissions_ci_upper_below_zero": practical is not None and practical_comparison["daily_emissions_difference_block_bootstrap_95ci_kg"][1] < 0,
    }
    solver_records = [
        {"day": item["day"], "reference_cost_usd": item["reference_cost_usd"], "solver": item["solver"]}
        for item in completed
    ]
    with (output / "solver_records.json").open("w", encoding="utf-8") as handle:
        json.dump(solver_records, handle, ensure_ascii=False)
    summary = {
        "status": "annual_pareto_frontier",
        "hours": 8760,
        "policies": list(POLICIES),
        "attached_read_only": {"F0": "main_2026-08-02/B1_dispatch.csv", "F5": "main_2026-08-02/P1_dispatch.csv"},
        "reference_cost_checks": reference_checks,
        "metrics": metrics,
        "comparisons": comparisons,
        "practical_policy": practical,
        "descriptive_knee": knee,
        "distinct_nondominated_points": distinct_count,
        "gates": gates,
        "passed_for_shapley": all(gates.values()),
        "evaluation_summary": {
            "outcome_summary": "annual seven-policy cost-carbon frontier evaluated",
            "claim_update": "supported" if all(gates.values()) else "refuted",
            "baseline_relation": "F0 and F5 attached after exact P0-window match; other policies newly run",
            "failure_mode": None if all(gates.values()) else "direction_underperforming",
            "next_action": "run_matched_comparators_and_shapley" if all(gates.values()) else "inspect_direction_failure",
            "comparability": "same data, physical model, forecast, daily cadence and annual denominator",
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
        "command": "python -m scripts.run_pareto",
    }
    with (output / "pareto_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"practical_policy": practical, "descriptive_knee": knee, "distinct_nondominated_points": distinct_count, "comparisons": comparisons, "gates": gates, "passed_for_shapley": summary["passed_for_shapley"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
