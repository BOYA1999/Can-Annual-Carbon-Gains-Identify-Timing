from copy import deepcopy
from pathlib import Path
import json
import pickle
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from scripts.run_main import _controls, _slice_envelope, _verify_hash
from src.data_pipeline import load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import HvacEnvelope, build_hvac_envelope


METHODS = ("F0", "F4", "B2")
WINDOWS = {
    "winter": "2023-01-09 00:00:00",
    "spring": "2023-04-10 00:00:00",
    "summer": "2023-07-10 00:00:00",
    "autumn": "2023-10-09 00:00:00",
}
SCENARIOS = [
    {"id": "nominal", "factor": "nominal", "level": 0.0, "seed": None},
    {"id": "pv_0p40", "factor": "pv_fraction", "level": 0.40, "seed": None},
    {"id": "pv_0p80", "factor": "pv_fraction", "level": 0.80, "seed": None},
    {"id": "bess_1h", "factor": "bess_duration_h", "level": 1.0, "seed": None},
    {"id": "bess_4h", "factor": "bess_duration_h", "level": 4.0, "seed": None},
    {"id": "eff_minus_2pp", "factor": "efficiency_shift", "level": -0.02, "seed": None},
    {"id": "eff_plus_2pp", "factor": "efficiency_shift", "level": 0.02, "seed": None},
]
for sigma in (0.05, 0.10, 0.20):
    for seed in (20260802, 20260803, 20260804):
        SCENARIOS.append({"id": f"forecast_{int(100 * sigma):02d}pct_seed_{seed}", "factor": "forecast_error_sd", "level": sigma, "seed": seed})


def _scenario_inputs(base_frame, base_parameters, scenario):
    frame = base_frame.copy()
    parameters = deepcopy(base_parameters)
    if scenario["factor"] == "pv_fraction":
        scale = scenario["level"] / 0.60
        frame["pv_dc_kw"] *= scale
        parameters["network"]["ratings_kw"]["pv"] *= scale
    elif scenario["factor"] == "bess_duration_h":
        parameters["bess"]["energy_kwh"] = parameters["network"]["ratings_kw"]["bess"] * scenario["level"]
    elif scenario["factor"] == "efficiency_shift":
        parameters["network"]["converter_efficiency_shift_fraction"] = scenario["level"]
    return frame, parameters


def _multiplier(seed, day, channel, sigma):
    rng = np.random.default_rng(np.random.SeedSequence([seed, day, channel]))
    return np.maximum(0.0, 1.0 + sigma * rng.standard_normal(24))


def _forecast_context(day, frame, envelope, scenario):
    indices = np.arange(day * 24, (day + 1) * 24)
    previous = (indices - 24) % 8760
    actual = frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
    forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    if scenario["factor"] == "forecast_error_sd":
        sigma, seed = scenario["level"], scenario["seed"]
        forecast["pv_dc_kw"] *= _multiplier(seed, day, 0, sigma)
        forecast["carbon_kg_per_kwh"] *= _multiplier(seed, day, 1, sigma)
        forecast_envelope = HvacEnvelope(
            baseline_kw=forecast_envelope.baseline_kw * _multiplier(seed, day, 2, sigma),
            fixed_kw=forecast_envelope.fixed_kw * _multiplier(seed, day, 3, sigma),
            lower_kw=forecast_envelope.lower_kw,
            upper_kw=forecast_envelope.upper_kw,
            state_a=forecast_envelope.state_a,
            state_b_c_per_kwh=forecast_envelope.state_b_c_per_kwh,
            comfort_delta_c=forecast_envelope.comfort_delta_c,
        )
    return actual, forecast, actual_envelope, forecast_envelope


def _run_scenario(scenario, days, base_frame, base_parameters, annual_carbon_mean):
    frame, parameters = _scenario_inputs(base_frame, base_parameters, scenario)
    envelope = build_hvac_envelope(frame)
    dispatch = {method: [] for method in METHODS}
    solver = []
    for day in days:
        actual, forecast, actual_envelope, forecast_envelope = _forecast_context(day, frame, envelope, scenario)
        try:
            f0 = solve_day(forecast, forecast_envelope, parameters, objective="cost")
        except RuntimeError as error:
            return {
                "status": "failed",
                "failure": {"day": day, "timestamp": str(actual["timestamp"].iloc[0]), "method": "F0", "error": str(error)},
                "dispatch": {method: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for method, parts in dispatch.items()},
                "solver": solver,
            }
        cap = f0.cost_usd + 0.04 * abs(f0.cost_usd)
        try:
            f4 = solve_day(forecast, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap)
        except RuntimeError as error:
            return {
                "status": "failed",
                "failure": {"day": day, "timestamp": str(actual["timestamp"].iloc[0]), "method": "F4", "error": str(error)},
                "dispatch": {method: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for method, parts in dispatch.items()},
                "solver": solver,
            }
        average = forecast.copy()
        average["carbon_kg_per_kwh"] = annual_carbon_mean
        try:
            b2 = solve_day(average, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap)
        except RuntimeError as error:
            return {
                "status": "failed",
                "failure": {"day": day, "timestamp": str(actual["timestamp"].iloc[0]), "method": "B2", "error": str(error)},
                "dispatch": {method: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for method, parts in dispatch.items()},
                "solver": solver,
            }
        plans = {"F0": f0, "F4": f4, "B2": b2}
        for method, plan in plans.items():
            dispatch[method].append(replay_day(actual, actual_envelope, parameters, _controls(plan.frame)))
        solver.append({
            "day": day,
            "reference_cost_usd": f0.cost_usd,
            "planned_cost_cap_usd": cap,
            "solver": {method: plan.solver for method, plan in plans.items()},
        })
    return {"status": "completed", "dispatch": {method: pd.concat(parts, ignore_index=True) for method, parts in dispatch.items()}, "solver": solver}


def _seasonal_metrics(frame):
    evaluations = [evaluate_dispatch(frame.iloc[start:start + 168].reset_index(drop=True)) for start in range(0, 672, 168)]
    return {
        "hours": len(frame),
        "cost_usd": sum(value["annual_operating_cost_usd"] for value in evaluations),
        "emissions_kgco2e": sum(value["annual_emissions_kgco2e"] for value in evaluations),
        "pv_self_consumption_fraction": float(frame["pv_used_kw"].sum() / max(frame["pv_dc_kw"].sum(), 1e-12)),
        "battery_throughput_kwh": sum(value["battery_throughput_kwh"] for value in evaluations),
        "peak_grid_import_kw": max(value["peak_grid_import_kw"] for value in evaluations),
        "maximum_energy_closure_relative_error": max(value["energy_closure_relative_error"] for value in evaluations),
        "maximum_carbon_closure_relative_error": max(value["carbon_closure_relative_error"] for value in evaluations),
        "comfort_violation_hours": sum(value["comfort_violation_hours"] for value in evaluations),
    }


def _comparison(left, right):
    return {
        "cost_fraction": (left["cost_usd"] - right["cost_usd"]) / max(abs(right["cost_usd"]), 1e-12),
        "emissions_fraction": (left["emissions_kgco2e"] - right["emissions_kgco2e"]) / max(abs(right["emissions_kgco2e"]), 1e-12),
    }


def _write_attached_rows(source, mask, destination):
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = [lines[0], *(lines[index + 1] for index in np.flatnonzero(mask))]
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    starts = {name: int(inputs.frame.index[inputs.frame["timestamp"] == pd.Timestamp(timestamp)][0]) for name, timestamp in WINDOWS.items()}
    days = [hour // 24 + offset for hour in starts.values() for offset in range(7)]
    mask = pd.to_datetime(inputs.frame["timestamp"]).isin(
        pd.DatetimeIndex(np.concatenate([pd.date_range(timestamp, periods=168, freq="h").to_numpy() for timestamp in WINDOWS.values()]))
    )
    output = root / "artifacts" / "experiment" / "sensitivity_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)

    pareto_root = root / "artifacts" / "experiment" / "pareto_2026-08-02"
    shapley_root = root / "artifacts" / "experiment" / "shapley_relative_2026-08-02"
    nominal_sources = {"F0": pareto_root / "F0_dispatch.csv", "F4": pareto_root / "F4_dispatch.csv", "B2": shapley_root / "B2_dispatch.csv"}
    nominal = {method: pd.read_csv(path).loc[mask].reset_index(drop=True) for method, path in nominal_sources.items()}
    if any(len(frame) != 672 for frame in nominal.values()):
        raise RuntimeError("nominal seasonal attachment has wrong denominator")

    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            completed = pickle.load(handle)
        for result in completed.values():
            result.setdefault("status", "completed")
    else:
        completed = {}
    annual_carbon_mean = float(inputs.frame["carbon_kg_per_kwh"].mean())
    for scenario in SCENARIOS[1:]:
        if scenario["id"] in completed:
            continue
        completed[scenario["id"]] = _run_scenario(scenario, days, inputs.frame, parameters, annual_carbon_mean)
        temporary = output / "checkpoint.tmp"
        with temporary.open("wb") as handle:
            pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint)
        print(f"represented_scenarios={len(completed)}/15 id={scenario['id']} status={completed[scenario['id']]['status']}", flush=True)

    results = {"nominal": {"status": "completed", "dispatch": nominal, "solver": []}, **completed}
    metrics = {}
    comparisons = {}
    slice_rows = []
    season_rows = []
    failures = []
    for scenario in SCENARIOS:
        scenario_id = scenario["id"]
        if results[scenario_id]["status"] == "failed":
            failure = {"scenario": scenario, "completed_hours": len(results[scenario_id]["dispatch"]["F0"]), **results[scenario_id]["failure"]}
            failures.append(failure)
            for method in METHODS:
                results[scenario_id]["dispatch"][method].to_csv(output / f"{scenario_id}_{method}_partial_dispatch.csv", index=False)
            continue
        metrics[scenario_id] = {}
        for method in METHODS:
            frame = results[scenario_id]["dispatch"][method]
            destination = output / f"{scenario_id}_{method}_dispatch.csv"
            if scenario_id == "nominal":
                _write_attached_rows(nominal_sources[method], mask, destination)
            else:
                frame.to_csv(destination, index=False)
            value = _seasonal_metrics(frame)
            metrics[scenario_id][method] = value
            slice_rows.append({"scenario": scenario_id, "factor": scenario["factor"], "level": scenario["level"], "seed": scenario["seed"], "method": method, **value})
            for index, season in enumerate(WINDOWS):
                season_value = evaluate_dispatch(frame.iloc[index * 168:(index + 1) * 168].reset_index(drop=True))
                season_rows.append({
                    "scenario": scenario_id,
                    "factor": scenario["factor"],
                    "level": scenario["level"],
                    "seed": scenario["seed"],
                    "season": season,
                    "method": method,
                    "cost_usd": season_value["annual_operating_cost_usd"],
                    "emissions_kgco2e": season_value["annual_emissions_kgco2e"],
                    "energy_closure_relative_error": season_value["energy_closure_relative_error"],
                    "carbon_closure_relative_error": season_value["carbon_closure_relative_error"],
                    "comfort_violation_hours": season_value["comfort_violation_hours"],
                })
        comparisons[scenario_id] = {
            "F4_vs_F0": _comparison(metrics[scenario_id]["F4"], metrics[scenario_id]["F0"]),
            "F4_vs_B2": _comparison(metrics[scenario_id]["F4"], metrics[scenario_id]["B2"]),
        }
    pd.DataFrame(slice_rows).to_csv(output / "slice_metrics.csv", index=False)
    pd.DataFrame(season_rows).to_csv(output / "season_metrics.csv", index=False)

    solver_records = [
        {"scenario": scenario["id"], **record}
        for scenario in SCENARIOS[1:]
        for record in results[scenario["id"]]["solver"]
    ]
    (output / "solver_records.json").write_text(json.dumps(solver_records, ensure_ascii=False), encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    solver_leaves = [record["solver"][method] for record in solver_records for method in METHODS]
    successful = [scenario for scenario in SCENARIOS if results[scenario["id"]]["status"] == "completed"]
    frames = [results[scenario["id"]]["dispatch"][method] for scenario in successful for method in METHODS]
    finite = all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all() for frame in frames)
    physical = all(
        value["maximum_energy_closure_relative_error"] <= 0.005
        and value["maximum_carbon_closure_relative_error"] <= 0.005
        and value["comfort_violation_hours"] == 0
        for scenario_metrics in metrics.values()
        for value in scenario_metrics.values()
    )
    complete = (
        len(results) == len(SCENARIOS)
        and all(results[scenario["id"]]["status"] in ("completed", "failed") for scenario in SCENARIOS)
        and all(len(results[scenario["id"]]["dispatch"][method]) == 672 for scenario in successful for method in METHODS)
        and all("failure" in results[scenario["id"]] for scenario in SCENARIOS if results[scenario["id"]]["status"] == "failed")
    )
    nominal_identity = all(
        pd.read_csv(output / f"nominal_{method}_dispatch.csv").equals(nominal[method])
        for method in METHODS
    )
    gates = {
        "D0_nominal_attachment_identity": nominal_identity,
        "D1_complete_denominator": complete,
        "D2_physical": physical,
        "D3_finite": finite,
        "D4_solver_denominator_recorded": all(value["status"] == 0 for value in solver_leaves) and len(failures) == sum(results[scenario["id"]]["status"] == "failed" for scenario in SCENARIOS),
    }
    reversals = [
        {"scenario": scenario["id"], "factor": scenario["factor"], "level": scenario["level"], "seed": scenario["seed"], "comparison": name}
        for scenario in successful
        for name, value in comparisons[scenario["id"]].items()
        if value["emissions_fraction"] >= 0
    ]
    summary = {
        "status": "bounded_seasonal_ofat",
        "hours_per_scenario": 672,
        "scenarios": SCENARIOS,
        "methods": METHODS,
        "metrics": metrics,
        "comparisons": comparisons,
        "emissions_rank_reversals": reversals,
        "direction_counts": {
            "F4_lower_emissions_than_F0": sum(comparisons[scenario["id"]]["F4_vs_F0"]["emissions_fraction"] < 0 for scenario in successful),
            "F4_lower_emissions_than_B2": sum(comparisons[scenario["id"]]["F4_vs_B2"]["emissions_fraction"] < 0 for scenario in successful),
            "evaluable_scenarios": len(successful),
            "failed_scenarios": len(failures),
            "total_scenarios": len(SCENARIOS),
        },
        "failures": failures,
        "new_solver_records": len(solver_leaves),
        "nonoptimal_solver_records": sum(value["status"] != 0 for value in solver_leaves),
        "maximum_mip_gap": max(value["mip_gap"] for value in solver_leaves),
        "gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "command": "python -m scripts.run_sensitivity",
    }
    (output / "sensitivity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"direction_counts": summary["direction_counts"], "emissions_rank_reversals": reversals, "gates": gates, "passed": summary["passed"]}, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("sensitivity recording gate failed")


if __name__ == "__main__":
    main()
