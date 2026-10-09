from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
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


SOC_FRACTIONS = (0.40, 0.50, 0.60)
TIME_LIMIT_SECONDS = 300.0


def run_case(root_text, season, soc_fraction):
    root = Path(root_text)
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    first = int(inputs.frame.index[inputs.frame.timestamp == pd.Timestamp(WINDOWS[season])][0])
    indices = np.arange(first, first + 168)
    previous = (indices - 24) % 8760
    actual = inputs.frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = inputs.frame.iloc[previous].pv_dc_kw.to_numpy()
    forecast["carbon_kg_per_kwh"] = inputs.frame.iloc[previous].carbon_kg_per_kwh.to_numpy()
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    parameters["bess"]["soc_terminal_fraction"] = soc_fraction
    initial_soc = soc_fraction * parameters["bess"]["energy_kwh"]
    started = time.perf_counter()
    try:
        f0 = solve_day(
            forecast,
            forecast_envelope,
            parameters,
            objective="cost",
            initial_soc_kwh=initial_soc,
            solver_time_limit_seconds=TIME_LIMIT_SECONDS,
        )
        cap = f0.cost_usd + 0.04 * abs(f0.cost_usd)
        f4 = solve_day(
            forecast,
            forecast_envelope,
            parameters,
            objective="emissions",
            cost_cap_usd=cap,
            initial_soc_kwh=initial_soc,
            solver_time_limit_seconds=TIME_LIMIT_SECONDS,
        )
    except DispatchError as error:
        return {
            "season": season,
            "soc_fraction": soc_fraction,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "solver": error.solver,
        }
    voltage = 380.0
    resistance = parameters["network"]["hvac_line_resistance_ohm"]
    rows = []
    trajectories = {}
    for method, plan in (("F0", f0), ("F4", f4)):
        realized = replay_day(
            actual,
            actual_envelope,
            parameters,
            _controls(plan.frame),
            initial_soc_kwh=initial_soc,
        )
        realized["hvac_linearized_drop_v"] = 1000.0 * realized["hvac_edge_kw"] * resistance / voltage
        realized["hvac_linearized_receiving_voltage_v"] = voltage - realized["hvac_linearized_drop_v"]
        metrics = evaluate_dispatch(realized)
        rows.append({
            "season": season,
            "soc_fraction": soc_fraction,
            "method": method,
            "annual_operating_cost_usd": metrics["annual_operating_cost_usd"],
            "annual_emissions_kgco2e": metrics["annual_emissions_kgco2e"],
            "battery_throughput_kwh": metrics["battery_throughput_kwh"],
            "energy_closure_relative_error": metrics["energy_closure_relative_error"],
            "maximum_hvac_edge_kw": float(realized["hvac_edge_kw"].max()),
            "maximum_linearized_drop_v": float(realized["hvac_linearized_drop_v"].max()),
            "maximum_linearized_drop_percent": float(100.0 * realized["hvac_linearized_drop_v"].max() / voltage),
            "minimum_linearized_receiving_voltage_v": float(realized["hvac_linearized_receiving_voltage_v"].min()),
        })
        trajectories[method] = realized
    return {
        "season": season,
        "soc_fraction": soc_fraction,
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "cost_cap_usd": cap,
        "F0": f0.solver,
        "F4": f4.solver,
        "rows": rows,
        "trajectories": trajectories,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "weekly_electrical_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    cases = [(season, soc) for soc in SOC_FRACTIONS for season in WINDOWS]
    records = []
    metric_rows = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_case, str(root), season, soc): (season, soc) for season, soc in cases}
        for future in as_completed(futures):
            result = future.result()
            records.append({key: value for key, value in result.items() if key not in ("rows", "trajectories")})
            if result["status"] == "completed":
                metric_rows.extend(result["rows"])
                for method, trajectory in result["trajectories"].items():
                    trajectory.to_csv(output / f"trajectory_{result['season']}_{result['soc_fraction']:.1f}_{method}.csv", index=False)
            (output / "solver_records_partial.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{result['season']} soc={result['soc_fraction']:.1f} {result['status']} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["season", "soc_fraction", "method"])
    metrics.to_csv(output / "scenario_metrics.csv", index=False)
    comparisons = []
    for (season, soc), group in metrics.groupby(["season", "soc_fraction"]):
        indexed = group.set_index("method")
        comparisons.append({
            "season": season,
            "soc_fraction": soc,
            "F4_minus_F0_emissions_fraction": indexed.loc["F4", "annual_emissions_kgco2e"] / indexed.loc["F0", "annual_emissions_kgco2e"] - 1.0,
            "F4_minus_F0_cost_fraction_of_abs_F0": (indexed.loc["F4", "annual_operating_cost_usd"] - indexed.loc["F0", "annual_operating_cost_usd"]) / abs(indexed.loc["F0", "annual_operating_cost_usd"]),
        })
    comparisons = pd.DataFrame(comparisons).sort_values(["season", "soc_fraction"])
    comparisons.to_csv(output / "comparisons.csv", index=False)
    records.sort(key=lambda item: (item["season"], item["soc_fraction"]))
    (output / "solver_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    completed = sum(item["status"] == "completed" for item in records)
    summary = {
        "status": "weekly_and_voltage_diagnostic",
        "case_count": len(cases),
        "completed_cases": completed,
        "failed_cases": len(cases) - completed,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "mip_rel_gap": 1e-8,
        "all_completed": completed == len(cases),
        "maximum_linearized_drop_percent": None if metrics.empty else float(metrics["maximum_linearized_drop_percent"].max()),
        "minimum_linearized_receiving_voltage_v": None if metrics.empty else float(metrics["minimum_linearized_receiving_voltage_v"].min()),
        "voltage_diagnostic_scope": "Only the configured 380 V HVAC branch with R=0.001 ohm; not a whole-network power-flow validation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
