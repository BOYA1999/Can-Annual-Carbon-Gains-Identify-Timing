import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scripts.run_main import _controls, _slice_envelope
from scripts.run_sensitivity import WINDOWS
from src.data_pipeline import load_inputs
from src.dispatch import DispatchError, load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope
from scripts.run_network_voltage_audit import node_voltages_pu

def case(root_text, previous_text, season, offset, output_text):
    root, previous, output = map(Path, (root_text, previous_text, output_text))
    records = json.loads((previous / "solver_records.json").read_text())
    aligned_record = next(r for r in records if r["season"] == season and r["soc_fraction"] == 0.5)
    row = {"season": season, "offset_days": offset, "soc_fraction": 0.5,
           "time_limit_seconds": 300, "mip_rel_gap": 1e-8, "aligned_record": aligned_record}
    if aligned_record["status"] != "completed":
        return {**row, "status": "baseline_not_certified", "new_solve_count": 0}
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    parameters["bess"]["soc_terminal_fraction"] = 0.5
    envelope = build_hvac_envelope(inputs.frame)
    first = int(inputs.frame.index[inputs.frame.timestamp == pd.Timestamp(WINDOWS[season])][0])
    indices = np.arange(first, first + 168)
    history = (indices - 24) % 8760
    actual = inputs.frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = inputs.frame.iloc[history].pv_dc_kw.to_numpy()
    aligned_carbon = inputs.frame.iloc[history].carbon_kg_per_kwh.to_numpy()
    mapping = (np.arange(7) - offset) % 7
    forecast["carbon_kg_per_kwh"] = aligned_carbon.reshape(7, 24)[mapping].ravel()
    assert np.array_equal(np.sort(aligned_carbon), np.sort(forecast.carbon_kg_per_kwh))
    row.update(cost_cap_usd=aligned_record["cost_cap_usd"], day_mapping=mapping.tolist(),
               same_sorted_carbon_values=True, new_solve_count=1)
    started = time.perf_counter()
    try:
        plan = solve_day(forecast, _slice_envelope(envelope, indices, history), parameters,
                         objective="emissions", cost_cap_usd=row["cost_cap_usd"],
                         initial_soc_kwh=0.5 * parameters["bess"]["energy_kwh"],
                         solver_time_limit_seconds=300.0)
    except DispatchError as error:
        return {**row, "status": "planning_not_certified", "solver": error.solver,
                "elapsed_seconds": time.perf_counter() - started}
    row.update(solver=plan.solver, elapsed_seconds=time.perf_counter() - started)
    try:
        replay = replay_day(actual, _slice_envelope(envelope, indices), parameters, _controls(plan.frame),
                            initial_soc_kwh=0.5 * parameters["bess"]["energy_kwh"])
    except ValueError as error:
        return {**row, "status": "replay_failed", "error": str(error)}
    aligned = pd.read_csv(previous / f"trajectory_{season}_0.5_F4.csv")
    metrics, baseline = evaluate_dispatch(replay), evaluate_dispatch(aligned)
    volts = node_voltages_pu(replay, parameters)
    replay.to_csv(output / f"trajectory_{season}_SHIFT_{offset}D.csv", index=False)
    row.update(status="completed", hours=len(replay), metrics=metrics, aligned_metrics=baseline,
               penalty_fraction=metrics["annual_emissions_kgco2e"] / baseline["annual_emissions_kgco2e"] - 1,
               minimum_voltage_pu=float(volts.min().min()), maximum_voltage_pu=float(volts.max().max()),
               violating_node_hours=int(((volts < .95) | (volts > 1.05)).to_numpy().sum()),
               soc_closure_kwh=float(replay.soc_end_kwh.iloc[-1] - replay.soc_start_kwh.iloc[0]),
               temperature_closure_c=float(replay.temperature_end_c.iloc[-1] - replay.temperature_start_c.iloc[0]),
               hvac_energy_closure_kwh=float(replay.hvac_adjustment_kw.sum()))
    return row

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, default=root/"artifacts/weekly_electrical_diagnostic")
    parser.add_argument("--output", type=Path, default=root/"artifacts/weekly_matched_timing")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(case, str(root), str(args.previous), season, offset, str(args.output))
                   for season in WINDOWS for offset in (1, 3)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (args.output / "solver_records_partial.json").write_text(json.dumps(results, indent=2))
            print(result["season"], result["offset_days"], result["status"], result.get("penalty_fraction"), flush=True)
    results.sort(key=lambda r: (r["season"], r["offset_days"]))
    (args.output / "solver_records.json").write_text(json.dumps(results, indent=2))
    table = pd.DataFrame([{k: v for k, v in row.items() if k not in ("aligned_record", "metrics", "aligned_metrics", "solver", "day_mapping")} for row in results])
    table.to_csv(args.output / "comparisons.csv", index=False)
    completed = [r for r in results if r["status"] == "completed"]
    summary = {"prespecified_pairs": 8, "certified_pairs": len(completed),
               "positive_pairs": sum(r["penalty_fraction"] > 0 for r in completed),
               "minimum_penalty_fraction": min((r["penalty_fraction"] for r in completed), default=None),
               "maximum_penalty_fraction": max((r["penalty_fraction"] for r in completed), default=None),
               "statuses": table.status.value_counts().to_dict(),
               "scope": "Four fixed seasonal weeks, SOC 0.5, week-local whole-day shifts 1 and 3; not an annual ensemble."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
