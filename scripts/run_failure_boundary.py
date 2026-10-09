from pathlib import Path
import json
import pickle
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from scripts.run_main import _verify_hash
from scripts.run_sensitivity import METHODS, WINDOWS, _comparison, _run_scenario, _seasonal_metrics
from src.data_pipeline import load_inputs
from src.dispatch import load_parameters
from src.evaluate import evaluate_dispatch


LEVELS = (0.125, 0.15, 0.175)
SEEDS = (20260802, 20260803, 20260804)
SCENARIOS = [
    {"id": f"forecast_{int(round(1000 * level)):03d}permille_seed_{seed}", "factor": "forecast_error_sd", "level": level, "seed": seed}
    for level in LEVELS
    for seed in SEEDS
]


def _season_rows(scenario, frames):
    rows = []
    for method, frame in frames.items():
        for index, season in enumerate(WINDOWS):
            value = evaluate_dispatch(frame.iloc[index * 168:(index + 1) * 168].reset_index(drop=True))
            rows.append({
                "scenario": scenario["id"],
                "level": scenario["level"],
                "seed": scenario["seed"],
                "season": season,
                "method": method,
                "cost_usd": value["annual_operating_cost_usd"],
                "emissions_kgco2e": value["annual_emissions_kgco2e"],
                "energy_closure_relative_error": value["energy_closure_relative_error"],
                "carbon_closure_relative_error": value["carbon_closure_relative_error"],
                "comfort_violation_hours": value["comfort_violation_hours"],
            })
    return rows


def _reversals(frame):
    result = []
    for keys, group in frame.groupby(["scenario", "level", "seed", "season"], dropna=False):
        values = group.set_index("method")["emissions_kgco2e"]
        if not all(method in values for method in METHODS):
            continue
        for comparator in ("F0", "B2"):
            fraction = values["F4"] / values[comparator] - 1
            if fraction >= 0:
                result.append({
                    "scenario": keys[0], "level": keys[1], "seed": keys[2], "season": keys[3],
                    "comparison": f"F4_vs_{comparator}", "emissions_fraction": float(fraction),
                })
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    source_root = root / "artifacts" / "experiment" / "sensitivity_2026-08-02"
    source = json.loads((source_root / "sensitivity_summary.json").read_text(encoding="utf-8"))
    if not source["passed"]:
        raise RuntimeError("verified sensitivity source did not pass")

    inputs = load_inputs(root)
    parameters = load_parameters(root)
    starts = {name: int(inputs.frame.index[inputs.frame["timestamp"] == pd.Timestamp(timestamp)][0]) for name, timestamp in WINDOWS.items()}
    days = [hour // 24 + offset for hour in starts.values() for offset in range(7)]
    annual_carbon_mean = float(inputs.frame["carbon_kg_per_kwh"].mean())
    output = root / "artifacts" / "experiment" / "forecast_boundary_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            completed = pickle.load(handle)
    else:
        completed = {}

    for scenario in SCENARIOS:
        if scenario["id"] in completed:
            continue
        completed[scenario["id"]] = _run_scenario(scenario, days, inputs.frame, parameters, annual_carbon_mean)
        temporary = output / "checkpoint.tmp"
        with temporary.open("wb") as handle:
            pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint)
        print(f"represented_scenarios={len(completed)}/9 id={scenario['id']} status={completed[scenario['id']]['status']}", flush=True)

    metrics = {}
    comparisons = {}
    failures = []
    season_rows = []
    for scenario in SCENARIOS:
        result = completed[scenario["id"]]
        if result["status"] == "failed":
            failures.append({
                "scenario": scenario,
                "completed_hours": len(result["dispatch"]["F0"]),
                **result["failure"],
            })
            for method in METHODS:
                result["dispatch"][method].to_csv(output / f"{scenario['id']}_{method}_partial_dispatch.csv", index=False)
            continue
        metrics[scenario["id"]] = {}
        for method in METHODS:
            frame = result["dispatch"][method]
            frame.to_csv(output / f"{scenario['id']}_{method}_dispatch.csv", index=False)
            metrics[scenario["id"]][method] = _seasonal_metrics(frame)
        comparisons[scenario["id"]] = {
            "F4_vs_F0": _comparison(metrics[scenario["id"]]["F4"], metrics[scenario["id"]]["F0"]),
            "F4_vs_B2": _comparison(metrics[scenario["id"]]["F4"], metrics[scenario["id"]]["B2"]),
        }
        season_rows.extend(_season_rows(scenario, result["dispatch"]))
    new_season_frame = pd.DataFrame(season_rows)
    new_season_frame.to_csv(output / "new_season_metrics.csv", index=False)
    (output / "new_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [{"level": 0.0, "seed": None, "scenario": "nominal", "status": "completed", "completed_hours": 672}]
    source_failures = {item["scenario"]["id"]: item for item in source["failures"]}
    for level in (0.05, 0.10, 0.20):
        for seed in SEEDS:
            scenario_id = f"forecast_{int(100 * level):02d}pct_seed_{seed}"
            if scenario_id in source_failures:
                failure = source_failures[scenario_id]
                rows.append({
                    "level": level, "seed": seed, "scenario": scenario_id, "status": "failed",
                    "completed_hours": failure["completed_hours"], "failure_method": failure["method"],
                    "failure_day": failure["day"], "failure_timestamp": failure["timestamp"],
                })
            else:
                rows.append({"level": level, "seed": seed, "scenario": scenario_id, "status": "completed", "completed_hours": 672})
    new_failures = {item["scenario"]["id"]: item for item in failures}
    for scenario in SCENARIOS:
        if scenario["id"] in new_failures:
            failure = new_failures[scenario["id"]]
            rows.append({
                "level": scenario["level"], "seed": scenario["seed"], "scenario": scenario["id"], "status": "failed",
                "completed_hours": failure["completed_hours"], "failure_method": failure["method"],
                "failure_day": failure["day"], "failure_timestamp": failure["timestamp"],
            })
        else:
            rows.append({"level": scenario["level"], "seed": scenario["seed"], "scenario": scenario["id"], "status": "completed", "completed_hours": 672})
    boundary = pd.DataFrame(rows).sort_values(["level", "seed"], na_position="first")

    comparison_lookup = {"nominal": source["comparisons"]["nominal"]}
    comparison_lookup.update({key: value for key, value in source["comparisons"].items() if key.startswith("forecast_")})
    comparison_lookup.update(comparisons)
    for name in ("F4_vs_F0", "F4_vs_B2"):
        boundary[f"{name}_cost_fraction"] = boundary.scenario.map(lambda key: comparison_lookup.get(key, {}).get(name, {}).get("cost_fraction"))
        boundary[f"{name}_emissions_fraction"] = boundary.scenario.map(lambda key: comparison_lookup.get(key, {}).get(name, {}).get("emissions_fraction"))
    boundary.to_csv(output / "forecast_boundary_curve.csv", index=False)

    first_failure = {}
    for seed in SEEDS:
        seed_rows = boundary[boundary.seed == seed].sort_values("level")
        failed = seed_rows[seed_rows.status == "failed"]
        if failed.empty:
            first_failure[str(seed)] = None
            continue
        first = failed.iloc[0]
        lower = seed_rows[(seed_rows.level < first.level) & (seed_rows.status == "completed")].level.max()
        first_failure[str(seed)] = {
            "last_completed_level": None if pd.isna(lower) else float(lower),
            "first_failed_level": float(first.level),
            "completed_hours_before_failure": int(first.completed_hours),
            "failure_method": first.failure_method,
            "failure_day": int(first.failure_day),
        }

    source_season = pd.read_csv(source_root / "season_metrics.csv")
    source_season = source_season[(source_season.scenario == "nominal") | (source_season.factor == "forecast_error_sd")].copy()
    source_season = source_season.rename(columns={"factor": "source_factor"})
    combined_season = pd.concat([
        source_season[["scenario", "level", "seed", "season", "method", "cost_usd", "emissions_kgco2e", "energy_closure_relative_error", "carbon_closure_relative_error", "comfort_violation_hours"]],
        new_season_frame,
    ], ignore_index=True)
    reversals = _reversals(combined_season)
    pd.DataFrame(reversals).to_csv(output / "season_rank_reversals.csv", index=False)

    solver_records = [
        {"scenario": scenario["id"], **record}
        for scenario in SCENARIOS
        for record in completed[scenario["id"]]["solver"]
    ]
    (output / "solver_records.json").write_text(json.dumps(solver_records, ensure_ascii=False), encoding="utf-8")
    solver_leaves = [record["solver"][method] for record in solver_records for method in METHODS]
    successful = [scenario for scenario in SCENARIOS if completed[scenario["id"]]["status"] == "completed"]
    frames = [completed[scenario["id"]]["dispatch"][method] for scenario in successful for method in METHODS]
    finite = all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all() for frame in frames)
    physical = all(
        value["maximum_energy_closure_relative_error"] <= 0.005
        and value["maximum_carbon_closure_relative_error"] <= 0.005
        and value["comfort_violation_hours"] == 0
        for scenario_metrics in metrics.values()
        for value in scenario_metrics.values()
    )
    gates = {
        "E3_source_passed": source["passed"],
        "E3_complete_new_scenario_denominator": len(completed) == 9 and len(successful) + len(failures) == 9,
        "E3_complete_boundary_grid": len(boundary) == 19,
        "E3_failure_denominators_recorded": all("completed_hours" in failure for failure in failures),
        "E3_finite_completed_dispatch": finite,
        "E3_physical_completed_dispatch": physical,
        "E3_successful_solvers_optimal": all(value["status"] == 0 for value in solver_leaves),
    }
    summary = {
        "status": "forecast_feasibility_boundary",
        "hours_per_scenario": 672,
        "new_levels": LEVELS,
        "seeds": SEEDS,
        "new_metrics": metrics,
        "new_comparisons": comparisons,
        "new_failures": failures,
        "first_failure_by_seed": first_failure,
        "season_rank_reversals": reversals,
        "new_completed_scenarios": len(successful),
        "new_failed_scenarios": len(failures),
        "new_solver_records": len(solver_leaves),
        "gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "command": "python -m scripts.run_failure_boundary",
    }
    (output / "failure_boundary_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"first_failure_by_seed": first_failure, "new_completed_scenarios": len(successful), "new_failed_scenarios": len(failures), "season_rank_reversals": reversals, "gates": gates}, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("E3 completion gate failed")


if __name__ == "__main__":
    main()
