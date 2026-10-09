from pathlib import Path

from src.data_pipeline import load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import HvacEnvelope, build_hvac_envelope


ROOT = Path(__file__).resolve().parents[1]


def test_one_day_cost_dispatch_is_feasible_and_closed():
    inputs = load_inputs(ROOT)
    parameters = load_parameters(ROOT)
    envelope = build_hvac_envelope(inputs.frame)
    left = int(inputs.frame.index[inputs.frame["timestamp"] == "2023-07-10 00:00:00"][0])
    day = inputs.frame.iloc[left:left + 24].reset_index(drop=True)
    envelope.baseline_kw = envelope.baseline_kw[left:left + 24]
    envelope.fixed_kw = envelope.fixed_kw[left:left + 24]
    envelope.lower_kw = envelope.lower_kw[left:left + 24]
    envelope.upper_kw = envelope.upper_kw[left:left + 24]
    result = solve_day(day, envelope, parameters, objective="cost")
    metrics = evaluate_dispatch(result.frame)
    assert metrics["energy_closure_relative_error"] < 1e-8
    assert metrics["carbon_closure_relative_error"] < 1e-8
    assert metrics["comfort_violation_hours"] == 0
    controls = {
        "hvac_adjustment_kw": result.frame["hvac_adjustment_kw"].to_numpy(),
        "battery_charge_stored_kw": (result.frame["battery_charge_bus_kw"] - result.frame["battery_charge_loss_kw"]).to_numpy(),
        "battery_discharge_kw": result.frame["battery_discharge_kw"].to_numpy(),
    }
    replay = replay_day(day, envelope, parameters, controls)
    replay_metrics = evaluate_dispatch(replay)
    assert replay_metrics["energy_closure_relative_error"] < 1e-8
    assert replay_metrics["carbon_closure_relative_error"] < 1e-8
    exact_replay = replay_day(day, envelope, parameters, controls, loss_model="exact")
    exact_metrics = evaluate_dispatch(exact_replay)
    assert exact_metrics["energy_closure_relative_error"] < 1e-8
    assert exact_metrics["carbon_closure_relative_error"] < 1e-8


def test_nondefault_initial_states_are_propagated():
    inputs = load_inputs(ROOT)
    parameters = load_parameters(ROOT)
    envelope = build_hvac_envelope(inputs.frame)
    left = int(inputs.frame.index[inputs.frame["timestamp"] == "2023-07-10 00:00:00"][0])
    day = inputs.frame.iloc[left:left + 24].reset_index(drop=True)
    envelope.baseline_kw = envelope.baseline_kw[left:left + 24]
    envelope.fixed_kw = envelope.fixed_kw[left:left + 24]
    envelope.lower_kw = envelope.lower_kw[left:left + 24]
    envelope.upper_kw = envelope.upper_kw[left:left + 24]
    initial_soc = 0.55 * parameters["bess"]["energy_kwh"]
    initial_temp = 0.1
    result = solve_day(
        day,
        envelope,
        parameters,
        objective="cost",
        initial_soc_kwh=initial_soc,
        initial_temp_deviation_c=initial_temp,
    )
    assert abs(result.frame["soc_start_kwh"].iloc[0] - initial_soc) < 1e-8
    assert abs(result.frame["temperature_start_c"].iloc[0] - (24.0 + initial_temp)) < 1e-8
    controls = {
        "hvac_adjustment_kw": result.frame["hvac_adjustment_kw"].iloc[:1].to_numpy(),
        "battery_charge_stored_kw": (result.frame["battery_charge_bus_kw"] - result.frame["battery_charge_loss_kw"]).iloc[:1].to_numpy(),
        "battery_discharge_kw": result.frame["battery_discharge_kw"].iloc[:1].to_numpy(),
    }
    replay = replay_day(
        day.iloc[:1],
        HvacEnvelope(
            baseline_kw=envelope.baseline_kw[:1],
            fixed_kw=envelope.fixed_kw[:1],
            lower_kw=envelope.lower_kw[:1],
            upper_kw=envelope.upper_kw[:1],
            state_a=envelope.state_a,
            state_b_c_per_kwh=envelope.state_b_c_per_kwh,
            comfort_delta_c=envelope.comfort_delta_c,
        ),
        parameters,
        controls,
        initial_soc_kwh=initial_soc,
        initial_temp_deviation_c=initial_temp,
    )
    assert abs(replay["soc_start_kwh"].iloc[0] - initial_soc) < 1e-8
    assert abs(replay["temperature_start_c"].iloc[0] - (24.0 + initial_temp)) < 1e-8


def test_calendar_day_terminal_and_remaining_hvac_balance():
    inputs = load_inputs(ROOT)
    parameters = load_parameters(ROOT)
    envelope = build_hvac_envelope(inputs.frame)
    left = int(inputs.frame.index[inputs.frame["timestamp"] == "2023-07-10 01:00:00"][0])
    day = inputs.frame.iloc[left:left + 24].reset_index(drop=True)
    envelope.baseline_kw = envelope.baseline_kw[left:left + 24]
    envelope.fixed_kw = envelope.fixed_kw[left:left + 24]
    envelope.lower_kw = envelope.lower_kw[left:left + 24]
    envelope.upper_kw = envelope.upper_kw[left:left + 24]
    initial_soc = 0.55 * parameters["bess"]["energy_kwh"]
    result = solve_day(
        day,
        envelope,
        parameters,
        objective="cost",
        initial_soc_kwh=initial_soc,
        terminal_step=23,
        hvac_energy_steps=23,
        hvac_energy_target_kwh=0.0,
    )
    terminal_soc = parameters["bess"]["soc_terminal_fraction"] * parameters["bess"]["energy_kwh"]
    assert abs(result.frame["soc_end_kwh"].iloc[22] - terminal_soc) < 1e-8
    assert abs(result.frame["temperature_end_c"].iloc[22] - 24.0) < 1e-8
    assert abs(result.frame["hvac_adjustment_kw"].iloc[:23].sum()) < 1e-8
