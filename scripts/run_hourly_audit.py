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


POLICIES = {"F0": None, "F1": 0.01, "F2": 0.02, "F3": 0.03, "F4": 0.04, "F5": 0.05, "F6": np.inf}


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


def _controls(frame: pd.DataFrame, hours: int | None = None) -> dict[str, np.ndarray]:
    view = frame if hours is None else frame.iloc[:hours]
    return {
        "hvac_adjustment_kw": view["hvac_adjustment_kw"].to_numpy(float),
        "battery_charge_stored_kw": (view["battery_charge_bus_kw"] - view["battery_charge_loss_kw"]).to_numpy(float),
        "battery_discharge_kw": view["battery_discharge_kw"].to_numpy(float),
    }


def _plan(
    forecast,
    envelope,
    parameters,
    allowance,
    initial_soc=None,
    initial_temp=0.0,
    terminal_step=None,
    hvac_energy_target=0.0,
):
    common = {
        "initial_soc_kwh": initial_soc,
        "initial_temp_deviation_c": initial_temp,
        "terminal_step": terminal_step,
        "hvac_energy_steps": terminal_step,
        "hvac_energy_target_kwh": hvac_energy_target,
    }
    if allowance is None:
        return solve_day(
            forecast,
            envelope,
            parameters,
            objective="cost",
            **common,
        ), None
    if np.isinf(allowance):
        return solve_day(
            forecast,
            envelope,
            parameters,
            objective="emissions",
            **common,
        ), None
    reference = solve_day(
        forecast,
        envelope,
        parameters,
        objective="cost",
        **common,
    )
    cap = reference.cost_usd + allowance * abs(reference.cost_usd)
    result = solve_day(
        forecast,
        envelope,
        parameters,
        objective="emissions",
        cost_cap_usd=cap,
        **common,
    )
    return result, {"cost_cap_usd": cap, "reference_solver": reference.solver}


def _daily(frame, envelope, parameters, first):
    dispatch = {policy: [] for policy in POLICIES}
    solvers = []
    for day_number in range(7):
        indices = np.arange(first + 24 * day_number, first + 24 * (day_number + 1))
        previous = (indices - 24) % 8760
        actual = frame.iloc[indices].reset_index(drop=True)
        forecast = actual.copy()
        forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
        forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
        actual_envelope = _slice_envelope(envelope, indices)
        forecast_envelope = _slice_envelope(envelope, indices, previous)
        day_record = {"day": day_number, "policies": {}}
        for policy, allowance in POLICIES.items():
            plan, extra = _plan(forecast, forecast_envelope, parameters, allowance)
            dispatch[policy].append(replay_day(actual, actual_envelope, parameters, _controls(plan.frame)))
            day_record["policies"][policy] = {"plan": plan.solver, "cap": extra}
        solvers.append(day_record)
    return {policy: pd.concat(parts, ignore_index=True) for policy, parts in dispatch.items()}, solvers


def _hourly(frame, envelope, parameters, first, checkpoint):
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["version"] != "calendar_day_terminal_v1":
            raise RuntimeError("incompatible hourly checkpoint")
        dispatch = saved["dispatch"]
        states = saved["states"]
        cumulative_hvac = saved["cumulative_hvac"]
        solvers = saved["solvers"]
        start_offset = saved["next_offset"]
    else:
        dispatch = {policy: [] for policy in POLICIES}
        initial_soc = parameters["bess"]["soc_initial_fraction"] * parameters["bess"]["energy_kwh"]
        states = {policy: (initial_soc, 0.0) for policy in POLICIES}
        cumulative_hvac = {policy: 0.0 for policy in POLICIES}
        solvers = []
        start_offset = 0
    for offset in range(start_offset, 168):
        current = first + offset
        indices = np.arange(current, current + 24) % 8760
        previous = (indices - 24) % 8760
        actual_index = np.array([current % 8760])
        actual = frame.iloc[actual_index].reset_index(drop=True)
        forecast = frame.iloc[indices].reset_index(drop=True).copy()
        forecast["pv_dc_kw"] = frame.iloc[previous]["pv_dc_kw"].to_numpy()
        forecast["carbon_kg_per_kwh"] = frame.iloc[previous]["carbon_kg_per_kwh"].to_numpy()
        actual_envelope = _slice_envelope(envelope, actual_index)
        forecast_envelope = _slice_envelope(envelope, indices, previous)
        hour_record = {"hour": offset, "timestamp": str(actual["timestamp"].iloc[0]), "policies": {}}
        terminal_step = 24 - int(actual["timestamp"].iloc[0].hour)
        for policy, allowance in POLICIES.items():
            soc, temp = states[policy]
            try:
                plan, extra = _plan(
                    forecast,
                    forecast_envelope,
                    parameters,
                    allowance,
                    soc,
                    temp,
                    terminal_step,
                    -cumulative_hvac[policy],
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"hourly planning failed at hour={offset}, timestamp={actual['timestamp'].iloc[0]}, "
                    f"policy={policy}, initial_soc_kwh={soc}, initial_temp_deviation_c={temp}, "
                    f"terminal_step={terminal_step}, cumulative_hvac_kwh={cumulative_hvac[policy]}"
                ) from error
            realized = replay_day(
                actual,
                actual_envelope,
                parameters,
                _controls(plan.frame, 1),
                initial_soc_kwh=soc,
                initial_temp_deviation_c=temp,
            )
            dispatch[policy].append(realized)
            states[policy] = (float(realized["soc_end_kwh"].iloc[0]), float(realized["temperature_end_c"].iloc[0] - 24.0))
            cumulative_hvac[policy] += float(realized["hvac_adjustment_kw"].iloc[0])
            hour_record["policies"][policy] = {"plan": plan.solver, "cap": extra}
        solvers.append(hour_record)
        if int(actual["timestamp"].iloc[0].hour) == 23:
            cumulative_hvac = {policy: 0.0 for policy in POLICIES}
        if (offset + 1) % 6 == 0 or offset == 167:
            with checkpoint.open("wb") as handle:
                pickle.dump(
                    {
                        "version": "calendar_day_terminal_v1",
                        "next_offset": offset + 1,
                        "dispatch": dispatch,
                        "states": states,
                        "cumulative_hvac": cumulative_hvac,
                        "solvers": solvers,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        if (offset + 1) % 12 == 0 or offset == 167:
            print(f"hourly_completed={offset + 1}/168", flush=True)
    return {policy: pd.concat(parts, ignore_index=True) for policy, parts in dispatch.items()}, solvers


def _totals(frame):
    grid_net = frame["grid_import_kw"] - frame["grid_export_delivered_kw"]
    return {
        "cost_usd": float((frame["cost_usd_per_kwh"] * grid_net).sum()),
        "emissions_kgco2e": float((frame["carbon_kg_per_kwh"] * grid_net).sum()),
    }


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
    start = pd.Timestamp("2023-07-10 00:00:00")
    first = int(inputs.frame.index[inputs.frame["timestamp"] == start][0])
    output = root / "artifacts" / "experiment" / "p0_hourly_audit_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    daily_checkpoint = output / "daily_checkpoint.pkl"
    if daily_checkpoint.exists():
        with daily_checkpoint.open("rb") as handle:
            daily, daily_solver = pickle.load(handle)
    else:
        daily, daily_solver = _daily(inputs.frame, envelope, parameters, first)
        with daily_checkpoint.open("wb") as handle:
            pickle.dump((daily, daily_solver), handle, protocol=pickle.HIGHEST_PROTOCOL)
    hourly, hourly_solver = _hourly(
        inputs.frame,
        envelope,
        parameters,
        first,
        output / "hourly_checkpoint.pkl",
    )
    rows = []
    for cadence, results in (("daily", daily), ("hourly", hourly)):
        for policy, result in results.items():
            result.to_csv(output / f"{cadence}_{policy}_dispatch.csv", index=False)
            metrics = evaluate_dispatch(result)
            rows.append({"cadence": cadence, "policy": policy, **_totals(result), **metrics})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "cadence_metrics.csv", index=False)
    comparisons = []
    for policy in POLICIES:
        daily_row = metrics[(metrics.cadence == "daily") & (metrics.policy == policy)].iloc[0]
        hourly_row = metrics[(metrics.cadence == "hourly") & (metrics.policy == policy)].iloc[0]
        cost_difference = abs(hourly_row.cost_usd - daily_row.cost_usd) / max(abs(hourly_row.cost_usd), abs(daily_row.cost_usd), 1.0)
        emissions_difference = abs(hourly_row.emissions_kgco2e - daily_row.emissions_kgco2e) / max(abs(hourly_row.emissions_kgco2e), abs(daily_row.emissions_kgco2e), 1.0)
        daily_f0 = metrics[(metrics.cadence == "daily") & (metrics.policy == "F0")].iloc[0]
        hourly_f0 = metrics[(metrics.cadence == "hourly") & (metrics.policy == "F0")].iloc[0]
        sign_same = np.sign(daily_row.emissions_kgco2e - daily_f0.emissions_kgco2e) == np.sign(hourly_row.emissions_kgco2e - hourly_f0.emissions_kgco2e)
        physical = all(
            row.energy_closure_relative_error <= 0.005
            and row.carbon_closure_relative_error <= 0.005
            and row.comfort_violation_hours == 0
            for row in (daily_row, hourly_row)
        )
        comparisons.append({
            "policy": policy,
            "cost_relative_difference": float(cost_difference),
            "emissions_relative_difference": float(emissions_difference),
            "emissions_sign_same": bool(sign_same),
            "physical_gates_pass": bool(physical),
            "passed": bool(cost_difference <= 0.005 and emissions_difference <= 0.005 and sign_same and physical),
        })
    with (output / "solver_records.json").open("w", encoding="utf-8") as handle:
        json.dump({"daily": daily_solver, "hourly": hourly_solver}, handle, ensure_ascii=False)
    summary = {
        "status": "P0_hourly_equivalence_audit",
        "period": {"start": str(start), "hours": 168},
        "policies": list(POLICIES),
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons),
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "command": "python -m scripts.run_hourly_audit",
    }
    with (output / "p0_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
