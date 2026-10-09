from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.run_main import _slice_envelope, _verify_hash
from scripts.run_sensitivity import WINDOWS
from src.data_pipeline import load_inputs
from src.dispatch import DispatchError, _loss_value, load_parameters, solve_day
from src.model import HvacEnvelope, build_hvac_envelope


LEVELS = (0.125, 0.15, 0.175, 0.20)
SEEDS = (20260802, 20260803, 20260804)
CHANNELS = ("carbon", "pv", "hvac", "fixed", "joint")


def _forecast_metrics(actual, forecast, mask):
    left = actual[mask]
    right = forecast[mask]
    error = right - left
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(right) > 0 else None
    daily = []
    for start in range(0, len(left) - 23, 24):
        x, y = left[start:start + 24], right[start:start + 24]
        if np.std(x) > 0 and np.std(y) > 0:
            daily.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "hours": int(mask.sum()),
        "mae_kg_per_kwh": float(np.mean(np.abs(error))),
        "rmse_kg_per_kwh": float(np.sqrt(np.mean(error**2))),
        "nrmse_by_realized_mean": float(np.sqrt(np.mean(error**2)) / np.mean(left)),
        "pearson_r": correlation,
        "daily_pearson_q05": float(np.quantile(daily, 0.05)) if daily else None,
        "daily_pearson_median": float(np.median(daily)) if daily else None,
        "daily_pearson_q95": float(np.quantile(daily, 0.95)) if daily else None,
        "finite_daily_correlations": len(daily),
    }


def _diagnose_forecasts(frame):
    actual = frame.carbon_kg_per_kwh.to_numpy()
    hour_of_week = np.arange(len(actual)) % 168
    sums = np.bincount(hour_of_week, weights=actual, minlength=168)
    counts = np.bincount(hour_of_week, minlength=168)
    forecasts = {
        "previous_day": np.roll(actual, 24),
        "previous_week": np.roll(actual, 168),
        "leave_one_week_out_hour_of_week": (sums[hour_of_week] - actual) / (counts[hour_of_week] - 1),
        "annual_mean": np.full(len(actual), actual.mean()),
    }
    timestamps = pd.to_datetime(frame.timestamp)
    masks = {
        "annual_with_wrap": np.ones(len(actual), dtype=bool),
        "annual_without_first_day": np.arange(len(actual)) >= 24,
        "winter": timestamps.dt.month.isin((12, 1, 2)).to_numpy(),
        "spring": timestamps.dt.month.isin((3, 4, 5)).to_numpy(),
        "summer": timestamps.dt.month.isin((6, 7, 8)).to_numpy(),
        "autumn": timestamps.dt.month.isin((9, 10, 11)).to_numpy(),
    }
    rows = []
    for method, values in forecasts.items():
        for period, mask in masks.items():
            rows.append({"method": method, "period": period, **_forecast_metrics(actual, values, mask)})
    scatter = pd.DataFrame({"timestamp": frame.timestamp, "realized": actual, **forecasts})
    return pd.DataFrame(rows), scatter


def _multiplier(seed, day, channel, sigma):
    raw = 1.0 + sigma * np.random.default_rng(np.random.SeedSequence([seed, day, channel])).standard_normal(24)
    return np.maximum(0.0, raw), int((raw < 0).sum())


def _perturbed_day(day, frame, envelope, level, seed, channel):
    indices = np.arange(day * 24, (day + 1) * 24)
    previous = (indices - 24) % 8760
    forecast = frame.iloc[indices].reset_index(drop=True).copy()
    forecast["pv_dc_kw"] = frame.iloc[previous].pv_dc_kw.to_numpy()
    forecast["carbon_kg_per_kwh"] = frame.iloc[previous].carbon_kg_per_kwh.to_numpy()
    source = _slice_envelope(envelope, indices, previous)
    baseline, fixed = source.baseline_kw.copy(), source.fixed_kw.copy()
    clipped = 0
    if channel in ("pv", "joint"):
        values, count = _multiplier(seed, day, 0, level)
        forecast["pv_dc_kw"] *= values
        clipped += count
    if channel in ("carbon", "joint"):
        values, count = _multiplier(seed, day, 1, level)
        forecast["carbon_kg_per_kwh"] *= values
        clipped += count
    if channel in ("hvac", "joint"):
        values, count = _multiplier(seed, day, 2, level)
        baseline *= values
        clipped += count
    if channel in ("fixed", "joint"):
        values, count = _multiplier(seed, day, 3, level)
        fixed *= values
        clipped += count
    perturbed = HvacEnvelope(
        baseline_kw=baseline,
        fixed_kw=fixed,
        lower_kw=source.lower_kw,
        upper_kw=source.upper_kw,
        state_a=source.state_a,
        state_b_c_per_kwh=source.state_b_c_per_kwh,
        comfort_delta_c=source.comfort_delta_c,
    )
    return forecast, perturbed, clipped


def _feasible(forecast, envelope, parameters):
    try:
        solve_day(forecast, envelope, parameters, objective="cost")
        return True, {"status": 0, "message": "Optimal"}
    except DispatchError as error:
        return False, error.solver


def _minimum_fixed_rating_relaxation(forecast, envelope, parameters):
    base = parameters["network"]["ratings_kw"]["fixed_converter"]
    high = 2.0 * base
    trial = deepcopy(parameters)
    trial["network"]["ratings_kw"]["fixed_converter"] = high
    if not _feasible(forecast, envelope, trial)[0]:
        return None
    low = base
    for _ in range(24):
        middle = 0.5 * (low + high)
        trial["network"]["ratings_kw"]["fixed_converter"] = middle
        if _feasible(forecast, envelope, trial)[0]:
            high = middle
        else:
            low = middle
    return high - base


def _failure_diagnostic(forecast, envelope, parameters, clipped, solver):
    ratings = parameters["network"]["ratings_kw"]
    fixed_deliverable = ratings["fixed_converter"] - _loss_value(ratings["fixed_converter"], ratings["fixed_converter"])
    hvac_deliverable = ratings["hvac_line"] - _loss_value(ratings["hvac_line"], ratings["hvac_line"], line=True)
    fixed_shortfall = float(max(0.0, envelope.fixed_kw.max() - fixed_deliverable))
    hvac_shortfall = float(max(0.0, np.max(envelope.baseline_kw + envelope.lower_kw) - hvac_deliverable))
    if fixed_shortfall > 0:
        implicated = "fixed_converter_output_capacity"
    elif hvac_shortfall > 0:
        implicated = "hvac_line_output_capacity"
    else:
        implicated = "undetermined_without_IIS"
    return {
        "solver": solver,
        "iis_available_from_scipy_milp": False,
        "implicated_constraint": implicated,
        "negative_multiplier_clips": clipped,
        "pv_kw_range": [float(forecast.pv_dc_kw.min()), float(forecast.pv_dc_kw.max())],
        "carbon_kg_per_kwh_range": [float(forecast.carbon_kg_per_kwh.min()), float(forecast.carbon_kg_per_kwh.max())],
        "hvac_baseline_kw_range": [float(envelope.baseline_kw.min()), float(envelope.baseline_kw.max())],
        "fixed_load_kw_range": [float(envelope.fixed_kw.min()), float(envelope.fixed_kw.max())],
        "hvac_lower_le_upper": bool(np.all(envelope.lower_kw <= envelope.upper_kw)),
        "fixed_converter_deliverable_kw": float(fixed_deliverable),
        "fixed_output_shortfall_kw": fixed_shortfall,
        "hvac_minimum_output_shortfall_kw": hvac_shortfall,
        "minimum_fixed_converter_rating_relaxation_kw": _minimum_fixed_rating_relaxation(forecast, envelope, parameters) if fixed_shortfall > 0 else None,
    }


def _feasibility_grid(frame, envelope, parameters):
    starts = [int(frame.index[frame.timestamp == pd.Timestamp(timestamp)][0]) // 24 for timestamp in WINDOWS.values()]
    days = [start + offset for start in starts for offset in range(7)]
    scenarios, failures = [], []
    for channel in CHANNELS:
        for level in LEVELS:
            for seed in SEEDS:
                failed = []
                for day in days:
                    forecast, perturbed, clipped = _perturbed_day(day, frame, envelope, level, seed, channel)
                    feasible, solver = _feasible(forecast, perturbed, parameters)
                    if not feasible:
                        record = {
                            "channel": channel,
                            "level": level,
                            "seed": seed,
                            "day": day,
                            "timestamp": str(frame.timestamp.iloc[day * 24]),
                            **_failure_diagnostic(forecast, perturbed, parameters, clipped, solver),
                        }
                        failed.append(record)
                        failures.append(record)
                scenarios.append({
                    "channel": channel,
                    "level": level,
                    "seed": seed,
                    "evaluated_days": len(days),
                    "feasible_days": len(days) - len(failed),
                    "failed_days": len(failed),
                    "first_failed_day": failed[0]["day"] if failed else None,
                })
                print(f"forecast_grid channel={channel} level={level:.3f} seed={seed} failures={len(failed)}/28", flush=True)
    return pd.DataFrame(scenarios), failures


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(
        root / "config" / "extended_run_contract.md",
        root / "config" / "extended_run_contract.sha256",
    )
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    output = root / "artifacts" / "experiment" / "forecast_diagnostics_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)

    metrics, scatter = _diagnose_forecasts(inputs.frame)
    metrics.to_csv(output / "forecast_metrics.csv", index=False)
    scatter.to_csv(output / "forecast_scatter_data.csv", index=False)
    scenarios, failures = _feasibility_grid(inputs.frame, envelope, parameters)
    scenarios.to_csv(output / "channel_feasibility_grid.csv", index=False)
    (output / "failure_diagnostics.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    fixed_failures = [item for item in failures if item["implicated_constraint"] == "fixed_converter_output_capacity"]
    gates = {
        "forecast_metrics_complete": len(metrics) == 24,
        "channel_grid_complete": len(scenarios) == len(CHANNELS) * len(LEVELS) * len(SEEDS),
        "all_failures_have_raw_status": all("status" in item["solver"] and "message" in item["solver"] for item in failures),
        "all_failures_have_input_ranges": all("pv_kw_range" in item and "fixed_load_kw_range" in item for item in failures),
        "fixed_capacity_failures_have_relaxation": all(item["minimum_fixed_converter_rating_relaxation_kw"] is not None for item in fixed_failures),
    }
    summary = {
        "status": "forecast_diagnostics_and_failure_cause",
        "scenario_count": len(scenarios),
        "failure_count": len(failures),
        "failure_counts_by_channel": scenarios.groupby("channel").failed_days.sum().astype(int).to_dict(),
        "implicated_constraint_counts": pd.Series([item["implicated_constraint"] for item in failures]).value_counts().to_dict(),
        "gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "command": "python -m scripts.run_forecast_diagnostics",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("forecast diagnostic gate failed")


if __name__ == "__main__":
    main()
