from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.run_main import _controls, _slice_envelope, _verify_hash
from scripts.run_sensitivity import WINDOWS
from src.data_pipeline import load_inputs
from src.dispatch import DispatchError, load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope


SOC_FRACTIONS = (0.40, 0.50, 0.60)
THROUGHPUT_PRICES = (0.0, 0.01, 0.03, 0.05)


def _inputs(frame, envelope, indices):
    previous = (indices - 24) % 8760
    actual = frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = frame.iloc[previous].pv_dc_kw.to_numpy()
    forecast["carbon_kg_per_kwh"] = frame.iloc[previous].carbon_kg_per_kwh.to_numpy()
    return actual, forecast, _slice_envelope(envelope, indices), _slice_envelope(envelope, indices, previous)


def _metrics(frame, energy_kwh, throughput_price):
    values = evaluate_dispatch(frame)
    fee = float(frame.throughput_cost_usd.sum())
    return {
        **values,
        "energy_cost_usd": values["annual_operating_cost_usd"],
        "throughput_cost_usd": fee,
        "total_cost_usd": values["annual_operating_cost_usd"] + fee,
        "equivalent_full_cycles": values["battery_throughput_kwh"] / energy_kwh,
        "throughput_price_usd_per_kwh": throughput_price,
    }


def _plans(forecast, forecast_envelope, parameters, throughput_price, initial_soc):
    common = {
        "initial_soc_kwh": initial_soc,
        "throughput_cost_usd_per_kwh": throughput_price,
    }
    try:
        f0 = solve_day(forecast, forecast_envelope, parameters, objective="cost", **common)
    except DispatchError as error:
        return None, None, None, {"stage": "F0", "solver": error.solver}
    cap = f0.cost_usd + 0.04 * abs(f0.cost_usd)
    try:
        f4 = solve_day(forecast, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap, **common)
    except DispatchError as error:
        return f0, None, cap, {"stage": "F4", "solver": error.solver}
    return f0, f4, cap, None


def _weekly(frame, envelope, base_parameters, output):
    rows, solver = [], []
    for soc_fraction in SOC_FRACTIONS:
        parameters = deepcopy(base_parameters)
        parameters["bess"]["soc_terminal_fraction"] = soc_fraction
        initial_soc = soc_fraction * parameters["bess"]["energy_kwh"]
        for season, timestamp in WINDOWS.items():
            first = int(frame.index[frame.timestamp == pd.Timestamp(timestamp)][0])
            indices = np.arange(first, first + 168)
            actual, forecast, actual_envelope, forecast_envelope = _inputs(frame, envelope, indices)
            f0, f4, cap, failure = _plans(forecast, forecast_envelope, parameters, 0.0, initial_soc)
            if failure:
                solver.append({"season": season, "soc_fraction": soc_fraction, "status": "failed", **failure})
                print(f"weekly_state season={season} soc={soc_fraction:.2f} status=failed stage={failure['stage']}", flush=True)
                continue
            for method, plan in (("F0", f0), ("F4", f4)):
                realized = replay_day(actual, actual_envelope, parameters, _controls(plan.frame), initial_soc_kwh=initial_soc)
                rows.append({"cadence": "weekly", "season": season, "soc_fraction": soc_fraction, "method": method, **_metrics(realized, parameters["bess"]["energy_kwh"], 0.0)})
            solver.append({"season": season, "soc_fraction": soc_fraction, "status": "completed", "cost_cap_usd": cap, "F0": f0.solver, "F4": f4.solver})
            pd.DataFrame(rows).to_csv(output / "weekly_state_metrics_partial.csv", index=False)
            print(f"weekly_state season={season} soc={soc_fraction:.2f} status=completed", flush=True)
    return pd.DataFrame(rows), solver


def _daily_degradation(frame, envelope, base_parameters, output):
    rows, solver = [], []
    initial_soc = base_parameters["bess"]["soc_initial_fraction"] * base_parameters["bess"]["energy_kwh"]
    for price in THROUGHPUT_PRICES:
        for season, timestamp in WINDOWS.items():
            first = int(frame.index[frame.timestamp == pd.Timestamp(timestamp)][0])
            parts = {"F0": [], "F4": []}
            scenario_failed = False
            for offset in range(7):
                indices = np.arange(first + 24 * offset, first + 24 * (offset + 1))
                actual, forecast, actual_envelope, forecast_envelope = _inputs(frame, envelope, indices)
                f0, f4, cap, failure = _plans(forecast, forecast_envelope, base_parameters, price, initial_soc)
                if failure:
                    scenario_failed = True
                    solver.append({"season": season, "day": offset, "throughput_price": price, "status": "failed", **failure})
                    continue
                for method, plan in (("F0", f0), ("F4", f4)):
                    parts[method].append(replay_day(
                        actual,
                        actual_envelope,
                        base_parameters,
                        _controls(plan.frame),
                        initial_soc_kwh=initial_soc,
                        throughput_cost_usd_per_kwh=price,
                    ))
                solver.append({"season": season, "day": offset, "throughput_price": price, "status": "completed", "cost_cap_usd": cap, "F0": f0.solver, "F4": f4.solver})
            if not scenario_failed:
                for method, values in parts.items():
                    result = pd.concat(values, ignore_index=True)
                    rows.append({"cadence": "daily", "season": season, "soc_fraction": 0.50, "method": method, **_metrics(result, base_parameters["bess"]["energy_kwh"], price)})
                pd.DataFrame(rows).to_csv(output / "daily_degradation_metrics_partial.csv", index=False)
            print(f"daily_degradation season={season} price={price:.2f} status={'failed' if scenario_failed else 'completed'}", flush=True)
    return pd.DataFrame(rows), solver


def _comparisons(metrics, keys):
    rows = []
    for values, group in metrics.groupby(keys):
        indexed = group.set_index("method")
        cost_difference = float(indexed.loc["F4", "total_cost_usd"] - indexed.loc["F0", "total_cost_usd"])
        rows.append({
            **dict(zip(keys, values if isinstance(values, tuple) else (values,))),
            "F4_minus_F0_total_cost_usd": cost_difference,
            "F4_minus_F0_total_cost_fraction_of_abs_F0": cost_difference / abs(float(indexed.loc["F0", "total_cost_usd"])),
            "F4_minus_F0_emissions_fraction": float(indexed.loc["F4", "annual_emissions_kgco2e"] / indexed.loc["F0", "annual_emissions_kgco2e"] - 1),
            "F4_minus_F0_throughput_kwh": float(indexed.loc["F4", "battery_throughput_kwh"] - indexed.loc["F0", "battery_throughput_kwh"]),
        })
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(
        root / "config" / "extended_run_contract.md",
        root / "config" / "extended_run_contract.sha256",
    )
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    output = root / "artifacts" / "experiment" / "state_sensitivity_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)

    weekly, weekly_solver = _weekly(inputs.frame, envelope, parameters, output)
    daily, daily_solver = _daily_degradation(inputs.frame, envelope, parameters, output)
    weekly.to_csv(output / "weekly_state_metrics.csv", index=False)
    daily.to_csv(output / "daily_degradation_metrics.csv", index=False)
    weekly_comparison = _comparisons(weekly, ["season", "soc_fraction"])
    daily_comparison = _comparisons(daily, ["season", "throughput_price_usd_per_kwh"])
    weekly_comparison.to_csv(output / "weekly_state_comparisons.csv", index=False)
    daily_comparison.to_csv(output / "daily_degradation_comparisons.csv", index=False)
    (output / "solver_records.json").write_text(json.dumps({"weekly": weekly_solver, "daily": daily_solver}, ensure_ascii=False), encoding="utf-8")

    weekly_failures = [item for item in weekly_solver if item["status"] == "failed"]
    daily_failures = [item for item in daily_solver if item["status"] == "failed"]
    all_solver = [item[method] for item in weekly_solver + daily_solver if item["status"] == "completed" for method in ("F0", "F4")]
    gates = {
        "weekly_grid_accounted": len(weekly) // 2 + len(weekly_failures) == len(WINDOWS) * len(SOC_FRACTIONS),
        "degradation_grid_accounted": len(daily) // 2 + len({(item["season"], item["throughput_price"]) for item in daily_failures}) == len(WINDOWS) * len(THROUGHPUT_PRICES),
        "all_solvers_optimal": not weekly_failures and not daily_failures and all(item["status"] == 0 for item in all_solver),
        "all_completed_physical_closure": bool(pd.concat([weekly, daily]).energy_closure_relative_error.le(0.005).all()),
    }
    summary = {
        "status": "state_horizon_and_throughput_sensitivity",
        "weekly_completed_scenarios": len(weekly) // 2,
        "weekly_failed_scenarios": len(weekly_failures),
        "degradation_completed_scenarios": len(daily) // 2,
        "degradation_failed_records": len(daily_failures),
        "weekly_F4_lower_emissions_fraction": float((weekly_comparison.F4_minus_F0_emissions_fraction < 0).mean()) if len(weekly_comparison) else None,
        "degradation_F4_lower_emissions_fraction": float((daily_comparison.F4_minus_F0_emissions_fraction < 0).mean()) if len(daily_comparison) else None,
        "cost_change_normalization": "(F4 total cost - F0 total cost) / abs(F0 total cost)",
        "gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "command": "python -m scripts.run_state_sensitivity",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("state sensitivity gate failed")


if __name__ == "__main__":
    main()
