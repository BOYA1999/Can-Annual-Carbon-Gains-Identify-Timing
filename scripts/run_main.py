from pathlib import Path
import json
import pickle
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from src.data_pipeline import file_sha256, load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import HvacEnvelope, build_hvac_envelope


def _slice_envelope(source: HvacEnvelope, indices: np.ndarray, baseline_indices: np.ndarray | None = None) -> HvacEnvelope:
    baseline_indices = indices if baseline_indices is None else baseline_indices
    return HvacEnvelope(
        baseline_kw=source.baseline_kw[baseline_indices],
        fixed_kw=source.fixed_kw[indices],
        lower_kw=source.lower_kw[indices],
        upper_kw=source.upper_kw[indices],
        state_a=source.state_a,
        state_b_c_per_kwh=source.state_b_c_per_kwh,
        comfort_delta_c=source.comfort_delta_c,
    )


def _controls(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "hvac_adjustment_kw": frame["hvac_adjustment_kw"].to_numpy(float),
        "battery_charge_stored_kw": (frame["battery_charge_bus_kw"] - frame["battery_charge_loss_kw"]).to_numpy(float),
        "battery_discharge_kw": frame["battery_discharge_kw"].to_numpy(float),
    }


def _run_day(payload):
    day_number, frame, envelope, parameters, annual_carbon_mean = payload
    indices = np.arange(day_number * 24, (day_number + 1) * 24)
    previous = (indices - 24) % 8760
    actual = frame.iloc[indices].reset_index(drop=True)
    forecast = actual.copy()
    forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
    forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
    actual_envelope = _slice_envelope(envelope, indices)
    forecast_envelope = _slice_envelope(envelope, indices, previous)
    b1 = solve_day(forecast, forecast_envelope, parameters, objective="cost")
    cap = b1.cost_usd + 0.05 * abs(b1.cost_usd)
    p1 = solve_day(forecast, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap)
    average = forecast.copy()
    average["carbon_kg_per_kwh"] = annual_carbon_mean
    b2 = solve_day(average, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap)
    b3 = solve_day(forecast, forecast_envelope, parameters, objective="emissions_lossless", cost_cap_usd=cap)
    plans = {"B1": b1, "B2": b2, "B3": b3, "P1": p1}
    replay = {method: replay_day(actual, actual_envelope, parameters, _controls(result.frame)) for method, result in plans.items()}
    return {
        "day": day_number,
        "dispatch": replay,
        "solver": {method: {"plan": plans[method].solver, "replay": "analytic_realized_day_balance"} for method in plans},
        "planned_cost_cap": cap,
    }


def _block_bootstrap_upper(daily_difference: np.ndarray, block=7, samples=10000, seed=20260802) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    count = len(daily_difference)
    starts = np.arange(count)
    estimates = np.empty(samples)
    blocks_needed = int(np.ceil(count / block))
    for sample in range(samples):
        chosen = rng.choice(starts, blocks_needed, replace=True)
        indices = np.concatenate([(np.arange(start, start + block) % count) for start in chosen])[:count]
        estimates[sample] = daily_difference[indices].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _verify_hash(path: Path, record: Path):
    expected = record.read_text(encoding="utf-8").split()[0].lower()
    actual = file_sha256(path).lower()
    if actual != expected:
        raise RuntimeError(f"locked hash mismatch: {path}")
    return actual


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    output = root / "artifacts" / "experiment" / "main_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            completed = pickle.load(handle)
    else:
        completed = []
    start_day = len(completed)
    annual_carbon_mean = float(inputs.frame["carbon_kg_per_kwh"].mean())
    for day in range(start_day, 365):
        completed.append(_run_day((day, inputs.frame, envelope, parameters, annual_carbon_mean)))
        if (day + 1) % 5 == 0 or day == 364:
            with checkpoint.open("wb") as handle:
                pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"completed_days={day + 1}/365", flush=True)
    metrics = {}
    daily_rows = []
    solver_records = []
    for method in ("B1", "B2", "B3", "P1"):
        dispatch = pd.concat([item["dispatch"][method] for item in completed], ignore_index=True)
        dispatch.to_csv(output / f"{method}_dispatch.csv", index=False)
        metrics[method] = evaluate_dispatch(dispatch)
        daily = dispatch.assign(day=dispatch.index // 24).groupby("day", as_index=False).apply(
            lambda group: pd.Series({
                "cost_usd": float((group["cost_usd_per_kwh"] * (group["grid_import_kw"] - group["grid_export_delivered_kw"])).sum()),
                "emissions_kgco2e": float((group["carbon_kg_per_kwh"] * (group["grid_import_kw"] - group["grid_export_delivered_kw"])).sum()),
            }), include_groups=False
        ).reset_index(drop=True)
        daily["method"] = method
        daily_rows.append(daily)
    daily_metrics = pd.concat(daily_rows, ignore_index=True)
    daily_metrics.to_csv(output / "daily_metrics.csv", index=False)
    b1_daily = daily_metrics[daily_metrics.method == "B1"].sort_values("day")
    p1_daily = daily_metrics[daily_metrics.method == "P1"].sort_values("day")
    ci = _block_bootstrap_upper(p1_daily.emissions_kgco2e.to_numpy() - b1_daily.emissions_kgco2e.to_numpy())
    cost_change = metrics["P1"]["annual_operating_cost_usd"] / metrics["B1"]["annual_operating_cost_usd"] - 1
    emission_change = metrics["P1"]["annual_emissions_kgco2e"] / metrics["B1"]["annual_emissions_kgco2e"] - 1
    gates = {
        "G1_energy_closure": max(value["energy_closure_relative_error"] for value in metrics.values()) <= 0.005,
        "G1_carbon_closure": max(value["carbon_closure_relative_error"] for value in metrics.values()) <= 0.005,
        "G1_comfort": all(value["comfort_violation_hours"] == 0 for value in metrics.values()),
        "G2_nodal_signal": metrics["P1"]["nodal_p95_abs_diff_kg_per_kwh"] >= 0.01,
        "G3_emissions_ci_upper_below_zero": ci[1] < 0,
        "G3_cost_increase_at_most_5pct": cost_change <= 0.05,
    }
    forecast_error = {
        column: float(np.mean(np.abs(inputs.frame[column].to_numpy() - np.roll(inputs.frame[column].to_numpy(), 24))))
        for column in ("load_baseline_kw", "pv_dc_kw", "carbon_kg_per_kwh")
    }
    for item in completed:
        solver_records.append({"day": item["day"], "planned_cost_cap": item["planned_cost_cap"], "solver": item["solver"]})
    with (output / "solver_records.json").open("w", encoding="utf-8") as handle:
        json.dump(solver_records, handle, ensure_ascii=False)
    summary = {
        "status": "main_annual_daily_rolling_with_previous_day_forecast",
        "hours": 8760,
        "methods": ["B1", "B2", "B3", "P1"],
        "metrics": metrics,
        "comparison": {
            "P1_vs_B1_cost_fraction": cost_change,
            "P1_vs_B1_emissions_fraction": emission_change,
            "daily_emissions_difference_block_bootstrap_95ci_kg": list(ci),
        },
        "forecast_mae": forecast_error,
        "gates": gates,
        "passed": all(gates.values()),
        "cadence_note": "24-hour horizon re-optimized daily; the independent hourly equivalence audit is required",
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "workers": 1},
        "command": "python -m scripts.run_main"
    }
    with (output / "main_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"comparison": summary["comparison"], "gates": gates, "passed": summary["passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
