from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from src.model import HvacEnvelope, converter_tangents, line_loss_tangents


class _Builder:
    def __init__(self):
        self.names = {}
        self.lb = []
        self.ub = []
        self.integrality = []
        self.rows = []
        self.row_lb = []
        self.row_ub = []

    def variables(self, name, size, lower=0.0, upper=np.inf, integer=False):
        start = len(self.lb)
        index = np.arange(start, start + size)
        self.names[name] = index
        self.lb.extend(np.broadcast_to(lower, size).astype(float))
        self.ub.extend(np.broadcast_to(upper, size).astype(float))
        self.integrality.extend([int(integer)] * size)
        return index

    def constraint(self, terms, lower=-np.inf, upper=np.inf):
        row = {}
        for index, value in terms:
            row[int(index)] = row.get(int(index), 0.0) + float(value)
        self.rows.append(row)
        self.row_lb.append(float(lower))
        self.row_ub.append(float(upper))

    def matrix(self):
        rr, cc, vv = [], [], []
        for r, row in enumerate(self.rows):
            for c, value in row.items():
                rr.append(r)
                cc.append(c)
                vv.append(value)
        return coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.lb))).tocsr()


@dataclass
class DispatchResult:
    frame: pd.DataFrame
    objective_value: float
    cost_usd: float
    emissions_kgco2: float
    solver: dict


def _loss_constraints(builder, power, loss, rating, active, loss_aware, line=False, efficiency_shift=0.0):
    for t in range(len(power)):
        if not loss_aware:
            builder.constraint([(loss[t], 1)], lower=0, upper=0)
            if active is not None:
                builder.constraint([(power[t], 1), (active[t], -rating)], upper=0)
            continue
        if active is not None:
            builder.constraint([(power[t], 1), (active[t], -rating)], upper=0)
        lines = line_loss_tangents(rating) if line else converter_tangents(rating, efficiency_shift)
        for slope, intercept in lines:
            terms = [(loss[t], 1), (power[t], -slope)]
            if active is None:
                builder.constraint(terms, lower=intercept)
            else:
                terms.append((active[t], -intercept))
                builder.constraint(terms, lower=0)


def solve_day(
    day: pd.DataFrame,
    envelope: HvacEnvelope,
    parameters: dict,
    objective: str,
    loss_aware: bool = True,
    carbon_signal: str = "dynamic",
    cost_cap_usd: float | None = None,
    fixed_controls: dict[str, np.ndarray] | None = None,
    initial_soc_kwh: float | None = None,
    initial_temp_deviation_c: float = 0.0,
    terminal_step: int | None = None,
    hvac_energy_steps: int | None = None,
    hvac_energy_target_kwh: float = 0.0,
    hvac_flexible: bool = True,
    bess_flexible: bool = True,
) -> DispatchResult:
    hours = len(day)
    if hours != 24:
        raise ValueError("solve_day requires 24 hours")
    ratings = parameters["network"]["ratings_kw"]
    efficiency_shift = parameters["network"].get("converter_efficiency_shift_fraction", 0.0)
    bess = parameters["bess"]
    terminal_step = hours if terminal_step is None else terminal_step
    hvac_energy_steps = hours if hvac_energy_steps is None else hvac_energy_steps
    if not 1 <= terminal_step <= hours or not 1 <= hvac_energy_steps <= hours:
        raise ValueError("state and HVAC balance steps must be within the horizon")
    builder = _Builder()
    p_gi = builder.variables("grid_import_kw", hours, upper=ratings["pcc"])
    p_ge = builder.variables("grid_export_bus_kw", hours, upper=ratings["pcc"])
    p_pv = builder.variables("pv_used_kw", hours, upper=day["pv_dc_kw"].to_numpy())
    p_bd = builder.variables("battery_discharge_kw", hours, upper=ratings["bess"])
    p_bc = builder.variables("battery_charge_bus_kw", hours, upper=ratings["bess"])
    p_main = builder.variables("main_flow_kw", hours, upper=ratings["main"])
    p_hvac = builder.variables("hvac_edge_kw", hours, upper=ratings["hvac_line"])
    p_fixed = builder.variables("fixed_edge_kw", hours, upper=ratings["fixed_converter"])
    loss_names = ("grid_import_loss_kw", "grid_export_loss_kw", "pv_loss_kw", "battery_discharge_loss_kw", "battery_charge_loss_kw", "main_loss_kw", "hvac_line_loss_kw", "fixed_loss_kw")
    losses = {name: builder.variables(name, hours) for name in loss_names}
    z_gi = builder.variables("grid_import_on", hours, upper=1, integer=True)
    z_ge = builder.variables("grid_export_on", hours, upper=1, integer=True)
    z_pv = builder.variables("pv_on", hours, upper=1, integer=True)
    z_bd = builder.variables("battery_discharge_on", hours, upper=1, integer=True)
    z_bc = builder.variables("battery_charge_on", hours, upper=1, integer=True)
    u = builder.variables("hvac_adjustment_kw", hours, lower=envelope.lower_kw, upper=envelope.upper_kw)
    soc_lb = np.full(hours + 1, bess["soc_min_fraction"] * bess["energy_kwh"])
    soc_ub = np.full(hours + 1, bess["soc_max_fraction"] * bess["energy_kwh"])
    initial_soc = bess["soc_initial_fraction"] * bess["energy_kwh"] if initial_soc_kwh is None else initial_soc_kwh
    soc_lb[0] = soc_ub[0] = initial_soc
    soc_lb[terminal_step] = soc_ub[terminal_step] = bess["soc_terminal_fraction"] * bess["energy_kwh"]
    soc = builder.variables("soc_kwh", hours + 1, lower=soc_lb, upper=soc_ub)
    temp_lb = np.full(hours + 1, -envelope.comfort_delta_c)
    temp_ub = np.full(hours + 1, envelope.comfort_delta_c)
    temp_lb[0] = temp_ub[0] = initial_temp_deviation_c
    temp_lb[terminal_step] = temp_ub[terminal_step] = 0
    temp = builder.variables("temperature_deviation_c", hours + 1, lower=temp_lb, upper=temp_ub)

    _loss_constraints(builder, p_gi, losses[loss_names[0]], ratings["pcc"], z_gi, loss_aware, efficiency_shift=efficiency_shift)
    _loss_constraints(builder, p_ge, losses[loss_names[1]], ratings["pcc"], z_ge, loss_aware, efficiency_shift=efficiency_shift)
    _loss_constraints(builder, p_pv, losses[loss_names[2]], ratings["pv"], z_pv, loss_aware, efficiency_shift=efficiency_shift)
    _loss_constraints(builder, p_bd, losses[loss_names[3]], ratings["bess"], z_bd, loss_aware, efficiency_shift=efficiency_shift)
    _loss_constraints(builder, p_bc, losses[loss_names[4]], ratings["bess"], z_bc, loss_aware, efficiency_shift=efficiency_shift)
    _loss_constraints(builder, p_main, losses[loss_names[5]], ratings["main"], None, loss_aware, efficiency_shift=efficiency_shift)
    _loss_constraints(builder, p_hvac, losses[loss_names[6]], ratings["hvac_line"], None, loss_aware, line=True)
    _loss_constraints(builder, p_fixed, losses[loss_names[7]], ratings["fixed_converter"], None, loss_aware, efficiency_shift=efficiency_shift)

    for t in range(hours):
        builder.constraint([(z_gi[t], 1), (z_ge[t], 1)], upper=1)
        builder.constraint([(z_bd[t], 1), (z_bc[t], 1)], upper=1)
        builder.constraint(
            [(p_gi[t], 1), (losses[loss_names[0]][t], -1), (p_pv[t], 1), (losses[loss_names[2]][t], -1),
             (p_bd[t], 1), (losses[loss_names[3]][t], -1), (p_ge[t], -1), (p_bc[t], -1), (p_main[t], -1)],
            lower=0, upper=0,
        )
        builder.constraint(
            [(p_main[t], 1), (losses[loss_names[5]][t], -1), (p_hvac[t], -1), (p_fixed[t], -1)],
            lower=0, upper=0,
        )
        builder.constraint(
            [(p_hvac[t], 1), (losses[loss_names[6]][t], -1), (u[t], -1)],
            lower=envelope.baseline_kw[t], upper=envelope.baseline_kw[t],
        )
        builder.constraint(
            [(p_fixed[t], 1), (losses[loss_names[7]][t], -1)],
            lower=envelope.fixed_kw[t], upper=envelope.fixed_kw[t],
        )
        builder.constraint(
            [(soc[t + 1], 1), (soc[t], -1), (p_bc[t], -1), (losses[loss_names[4]][t], 1), (p_bd[t], 1)],
            lower=0, upper=0,
        )
        builder.constraint(
            [(temp[t + 1], 1), (temp[t], -envelope.state_a), (u[t], envelope.state_b_c_per_kwh)],
            lower=0, upper=0,
        )
    builder.constraint(
        [(index, 1) for index in u[:hvac_energy_steps]],
        lower=hvac_energy_target_kwh,
        upper=hvac_energy_target_kwh,
    )
    if not hvac_flexible:
        for index in u:
            builder.constraint([(index, 1)], lower=0, upper=0)
    if not bess_flexible:
        for discharge, charge in zip(p_bd, p_bc):
            builder.constraint([(discharge, 1)], lower=0, upper=0)
            builder.constraint([(charge, 1)], lower=0, upper=0)
    if fixed_controls is not None:
        for variable, index in (("hvac_adjustment_kw", u), ("battery_discharge_kw", p_bd)):
            values = np.asarray(fixed_controls[variable], dtype=float)
            if len(values) != hours:
                raise ValueError(f"invalid fixed control length: {variable}")
            for t, value in enumerate(values):
                builder.constraint([(index[t], 1)], lower=value, upper=value)
        stored = np.asarray(fixed_controls["battery_charge_stored_kw"], dtype=float)
        if len(stored) != hours:
            raise ValueError("invalid fixed control length: battery_charge_stored_kw")
        for t, value in enumerate(stored):
            builder.constraint([(p_bc[t], 1), (losses[loss_names[4]][t], -1)], lower=value, upper=value)

    price = day["cost_usd_per_kwh"].to_numpy(float)
    dynamic_carbon = day["carbon_kg_per_kwh"].to_numpy(float)
    carbon = dynamic_carbon if carbon_signal == "dynamic" else np.full(hours, dynamic_carbon.mean())
    cost_vector = np.zeros(len(builder.lb))
    carbon_vector = np.zeros(len(builder.lb))
    lossless_carbon_vector = np.zeros(len(builder.lb))
    for t in range(hours):
        cost_vector[p_gi[t]] += price[t]
        cost_vector[p_ge[t]] -= price[t]
        cost_vector[losses[loss_names[1]][t]] += price[t]
        carbon_vector[p_gi[t]] += carbon[t]
        carbon_vector[p_ge[t]] -= carbon[t]
        carbon_vector[losses[loss_names[1]][t]] += carbon[t]
        lossless_carbon_vector[u[t]] += carbon[t]
        lossless_carbon_vector[p_pv[t]] -= carbon[t]
        lossless_carbon_vector[p_bd[t]] -= carbon[t]
        lossless_carbon_vector[p_bc[t]] += carbon[t]
    if cost_cap_usd is not None:
        builder.constraint([(i, value) for i, value in enumerate(cost_vector) if value], upper=cost_cap_usd)
    target = {"cost": cost_vector, "emissions": carbon_vector, "emissions_lossless": lossless_carbon_vector}[objective]
    target = target + 1e-10 * np.array([0 if i in soc or i in temp or i in u else 1 for i in range(len(target))])
    matrix = builder.matrix()
    result = milp(
        target,
        integrality=np.asarray(builder.integrality),
        bounds=Bounds(np.asarray(builder.lb), np.asarray(builder.ub)),
        constraints=LinearConstraint(matrix, np.asarray(builder.row_lb), np.asarray(builder.row_ub)),
        options={"time_limit": 60, "mip_rel_gap": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"dispatch failed: {result.message}")
    x = result.x
    output = day[["timestamp", "carbon_kg_per_kwh", "cost_usd_per_kwh", "pv_dc_kw"]].reset_index(drop=True).copy()
    for name, index in builder.names.items():
        if len(index) == hours:
            output[name] = x[index]
    output["soc_start_kwh"] = x[soc[:-1]]
    output["soc_end_kwh"] = x[soc[1:]]
    output["temperature_start_c"] = 24.0 + x[temp[:-1]]
    output["temperature_end_c"] = 24.0 + x[temp[1:]]
    output["hvac_baseline_kw"] = envelope.baseline_kw
    output["fixed_load_kw"] = envelope.fixed_kw
    output["hvac_load_kw"] = envelope.baseline_kw + output["hvac_adjustment_kw"]
    output["grid_export_delivered_kw"] = output["grid_export_bus_kw"] - output["grid_export_loss_kw"]
    cost_value = float(output["cost_usd_per_kwh"].dot(output["grid_import_kw"] - output["grid_export_delivered_kw"]))
    emissions = float(output["carbon_kg_per_kwh"].dot(output["grid_import_kw"] - output["grid_export_delivered_kw"]))
    solver = {"status": int(result.status), "message": result.message, "mip_gap": float(getattr(result, "mip_gap", np.nan)), "mip_node_count": int(getattr(result, "mip_node_count", 0))}
    return DispatchResult(output, float(result.fun), cost_value, emissions, solver)


def load_parameters(root: Path) -> dict:
    with (root / "config" / "system_parameters.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _loss_value(power: float, rating: float, line: bool = False, efficiency_shift: float = 0.0) -> float:
    if power <= 0:
        return 0.0
    lines = line_loss_tangents(rating) if line else converter_tangents(rating, efficiency_shift)
    return max(0.0, max(slope * power + intercept for slope, intercept in lines))


def _input_for_output(output: float, rating: float, line: bool = False, efficiency_shift: float = 0.0) -> float:
    if output <= 0:
        return 0.0
    low, high = output, rating
    if high - _loss_value(high, rating, line, efficiency_shift) < output:
        raise ValueError("required converter output exceeds rating")
    for _ in range(60):
        middle = 0.5 * (low + high)
        if middle - _loss_value(middle, rating, line, efficiency_shift) < output:
            low = middle
        else:
            high = middle
    return high


def replay_day(
    day: pd.DataFrame,
    envelope: HvacEnvelope,
    parameters: dict,
    controls: dict[str, np.ndarray],
    initial_soc_kwh: float | None = None,
    initial_temp_deviation_c: float = 0.0,
) -> pd.DataFrame:
    ratings = parameters["network"]["ratings_kw"]
    efficiency_shift = parameters["network"].get("converter_efficiency_shift_fraction", 0.0)
    bess = parameters["bess"]
    hours = len(day)
    u = np.asarray(controls["hvac_adjustment_kw"], dtype=float)
    stored_charge = np.asarray(controls["battery_charge_stored_kw"], dtype=float).copy()
    battery_discharge = np.asarray(controls["battery_discharge_kw"], dtype=float)
    output = day[["timestamp", "carbon_kg_per_kwh", "cost_usd_per_kwh", "pv_dc_kw"]].reset_index(drop=True).copy()
    records = {name: np.zeros(hours) for name in (
        "grid_import_kw", "grid_export_bus_kw", "pv_used_kw", "battery_discharge_kw", "battery_charge_bus_kw",
        "main_flow_kw", "hvac_edge_kw", "fixed_edge_kw", "grid_import_loss_kw", "grid_export_loss_kw", "pv_loss_kw",
        "battery_discharge_loss_kw", "battery_charge_loss_kw", "main_loss_kw", "hvac_line_loss_kw", "fixed_loss_kw",
    )}
    soc_start = np.zeros(hours)
    soc_end = np.zeros(hours)
    temp_start = np.zeros(hours)
    temp_end = np.zeros(hours)
    soc = bess["soc_initial_fraction"] * bess["energy_kwh"] if initial_soc_kwh is None else initial_soc_kwh
    state = initial_temp_deviation_c
    for t in range(hours):
        hvac_load = envelope.baseline_kw[t] + u[t]
        p_hvac = _input_for_output(hvac_load, ratings["hvac_line"], line=True)
        l_hvac = _loss_value(p_hvac, ratings["hvac_line"], line=True)
        p_fixed = _input_for_output(envelope.fixed_kw[t], ratings["fixed_converter"], efficiency_shift=efficiency_shift)
        l_fixed = _loss_value(p_fixed, ratings["fixed_converter"], efficiency_shift=efficiency_shift)
        p_main = _input_for_output(p_hvac + p_fixed, ratings["main"], efficiency_shift=efficiency_shift)
        l_main = _loss_value(p_main, ratings["main"], efficiency_shift=efficiency_shift)
        max_stored_charge = ratings["bess"] - _loss_value(ratings["bess"], ratings["bess"], efficiency_shift=efficiency_shift)
        if stored_charge[t] > max_stored_charge:
            if stored_charge[t] - max_stored_charge > 1e-6:
                raise ValueError("planned stored charge exceeds physical tolerance")
            stored_charge[t] = max_stored_charge
        p_bc = _input_for_output(stored_charge[t], ratings["bess"], efficiency_shift=efficiency_shift)
        l_bc = _loss_value(p_bc, ratings["bess"], efficiency_shift=efficiency_shift)
        p_bd = battery_discharge[t]
        l_bd = _loss_value(p_bd, ratings["bess"], efficiency_shift=efficiency_shift)
        pv_available = min(float(day["pv_dc_kw"].iloc[t]), ratings["pv"])
        l_pv_available = _loss_value(pv_available, ratings["pv"], efficiency_shift=efficiency_shift)
        pv_delivered_available = max(0.0, pv_available - l_pv_available)
        n3_demand = p_main + p_bc - (p_bd - l_bd)
        if pv_delivered_available <= n3_demand:
            p_pv = pv_available if pv_delivered_available > 0 else 0.0
            l_pv = l_pv_available if p_pv else 0.0
            p_gi = _input_for_output(n3_demand - pv_delivered_available, ratings["pcc"], efficiency_shift=efficiency_shift)
            l_gi = _loss_value(p_gi, ratings["pcc"], efficiency_shift=efficiency_shift)
            p_ge = l_ge = 0.0
        else:
            p_pv = pv_available
            l_pv = l_pv_available
            p_ge = pv_delivered_available - n3_demand
            l_ge = min(p_ge, _loss_value(p_ge, ratings["pcc"], efficiency_shift=efficiency_shift))
            p_gi = l_gi = 0.0
        values = (p_gi, p_ge, p_pv, p_bd, p_bc, p_main, p_hvac, p_fixed, l_gi, l_ge, l_pv, l_bd, l_bc, l_main, l_hvac, l_fixed)
        for name, value in zip(records, values):
            records[name][t] = value
        soc_start[t] = soc
        soc = soc + stored_charge[t] - p_bd
        soc_end[t] = soc
        temp_start[t] = 24.0 + state
        state = envelope.state_a * state - envelope.state_b_c_per_kwh * u[t]
        temp_end[t] = 24.0 + state
    for name, values in records.items():
        output[name] = values
    output["grid_import_on"] = (output["grid_import_kw"] > 0).astype(int)
    output["grid_export_on"] = (output["grid_export_bus_kw"] > 0).astype(int)
    output["pv_on"] = (output["pv_used_kw"] > 0).astype(int)
    output["battery_discharge_on"] = (output["battery_discharge_kw"] > 0).astype(int)
    output["battery_charge_on"] = (output["battery_charge_bus_kw"] > 0).astype(int)
    output["hvac_adjustment_kw"] = u
    output["soc_start_kwh"] = soc_start
    output["soc_end_kwh"] = soc_end
    output["temperature_start_c"] = temp_start
    output["temperature_end_c"] = temp_end
    output["hvac_baseline_kw"] = envelope.baseline_kw
    output["fixed_load_kw"] = envelope.fixed_kw
    output["hvac_load_kw"] = envelope.baseline_kw + u
    output["grid_export_delivered_kw"] = output["grid_export_bus_kw"] - output["grid_export_loss_kw"]
    return output
