from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.run_main import _controls, _slice_envelope
from scripts.run_sensitivity import WINDOWS
from src.data_pipeline import load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "spring_integrality_relaxation"
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    first = int(inputs.frame.index[inputs.frame.timestamp == pd.Timestamp(WINDOWS["spring"])][0])
    indices = np.arange(first, first + 168)
    previous = (indices - 24) % 8760
    actual = inputs.frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = inputs.frame.iloc[previous].pv_dc_kw.to_numpy()
    forecast["carbon_kg_per_kwh"] = inputs.frame.iloc[previous].carbon_kg_per_kwh.to_numpy()
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    rows = []
    solvers = []
    for soc_fraction in (0.40, 0.50, 0.60):
        parameters["bess"]["soc_terminal_fraction"] = soc_fraction
        initial_soc = soc_fraction * parameters["bess"]["energy_kwh"]
        f0 = solve_day(
            forecast,
            forecast_envelope,
            parameters,
            objective="cost",
            initial_soc_kwh=initial_soc,
            relax_integrality=True,
        )
        cap = f0.cost_usd + 0.04 * abs(f0.cost_usd)
        f4 = solve_day(
            forecast,
            forecast_envelope,
            parameters,
            objective="emissions",
            cost_cap_usd=cap,
            initial_soc_kwh=initial_soc,
            relax_integrality=True,
        )
        solvers.append({"soc_fraction": soc_fraction, "F0": f0.solver, "F4": f4.solver, "cost_cap_usd": cap})
        for method, plan in (("F0", f0), ("F4", f4)):
            realized = replay_day(actual, actual_envelope, parameters, _controls(plan.frame), initial_soc_kwh=initial_soc)
            metrics = evaluate_dispatch(realized)
            simultaneous_grid = (plan.frame["grid_import_kw"] > 1e-6) & (plan.frame["grid_export_bus_kw"] > 1e-6)
            simultaneous_battery = (plan.frame["battery_discharge_kw"] > 1e-6) & (plan.frame["battery_charge_bus_kw"] > 1e-6)
            rows.append({
                "season": "spring",
                "soc_fraction": soc_fraction,
                "method": method,
                "annual_operating_cost_usd": metrics["annual_operating_cost_usd"],
                "annual_emissions_kgco2e": metrics["annual_emissions_kgco2e"],
                "battery_throughput_kwh": metrics["battery_throughput_kwh"],
                "energy_closure_relative_error": metrics["energy_closure_relative_error"],
                "simultaneous_grid_hours": int(simultaneous_grid.sum()),
                "simultaneous_battery_hours": int(simultaneous_battery.sum()),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "scenario_metrics.csv", index=False)
    comparisons = []
    for soc, group in metrics.groupby("soc_fraction"):
        indexed = group.set_index("method")
        comparisons.append({
            "season": "spring",
            "soc_fraction": soc,
            "F4_minus_F0_emissions_fraction": indexed.loc["F4", "annual_emissions_kgco2e"] / indexed.loc["F0", "annual_emissions_kgco2e"] - 1.0,
            "F4_minus_F0_cost_fraction_of_abs_F0": (indexed.loc["F4", "annual_operating_cost_usd"] - indexed.loc["F0", "annual_operating_cost_usd"]) / abs(indexed.loc["F0", "annual_operating_cost_usd"]),
        })
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / "comparisons.csv", index=False)
    summary = {
        "status": "spring_continuous_integrality_relaxation",
        "completed_cases": len(solvers),
        "all_solvers_optimal": all(item[method]["status"] == 0 for item in solvers for method in ("F0", "F4")),
        "F4_minus_F0_emissions_fraction_min": float(comparisons["F4_minus_F0_emissions_fraction"].min()),
        "F4_minus_F0_emissions_fraction_max": float(comparisons["F4_minus_F0_emissions_fraction"].max()),
        "maximum_simultaneous_grid_hours": int(metrics["simultaneous_grid_hours"].max()),
        "maximum_simultaneous_battery_hours": int(metrics["simultaneous_battery_hours"].max()),
        "interpretation": "Diagnostic convex relaxation only; it does not replace unresolved mixed-integer spring cases.",
    }
    (output / "solver_records.json").write_text(json.dumps(solvers, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
