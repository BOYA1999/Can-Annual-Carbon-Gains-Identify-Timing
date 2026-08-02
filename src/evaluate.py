import numpy as np
import pandas as pd
from itertools import combinations
from math import factorial


def evaluate_dispatch(frame: pd.DataFrame, initial_storage_carbon_intensity: float | None = None) -> dict:
    n3 = frame["grid_import_kw"] - frame["grid_import_loss_kw"] + frame["pv_used_kw"] - frame["pv_loss_kw"] + frame["battery_discharge_kw"] - frame["battery_discharge_loss_kw"] - frame["grid_export_bus_kw"] - frame["battery_charge_bus_kw"] - frame["main_flow_kw"]
    n4 = frame["main_flow_kw"] - frame["main_loss_kw"] - frame["hvac_edge_kw"] - frame["fixed_edge_kw"]
    n5 = frame["hvac_edge_kw"] - frame["hvac_line_loss_kw"] - frame["hvac_load_kw"]
    n6 = frame["fixed_edge_kw"] - frame["fixed_loss_kw"] - frame["fixed_load_kw"]
    soc = frame["soc_end_kwh"] - frame["soc_start_kwh"] - frame["battery_charge_bus_kw"] + frame["battery_charge_loss_kw"] + frame["battery_discharge_kw"]
    demand = float((frame["hvac_load_kw"] + frame["fixed_load_kw"]).sum())
    energy_error = float((n3.abs() + n4.abs() + n5.abs() + n6.abs() + soc.abs()).sum() / demand)

    initial_intensity = float(frame["carbon_kg_per_kwh"].iloc[0] if initial_storage_carbon_intensity is None else initial_storage_carbon_intensity)
    storage_carbon = float(frame["soc_start_kwh"].iloc[0] * initial_intensity)
    initial_storage_carbon = storage_carbon
    carbon_residuals = []
    node5_intensity = []
    node6_intensity = []
    attributed_load_carbon = 0.0
    attributed_export_carbon = 0.0
    for row in frame.itertuples(index=False):
        storage_intensity = storage_carbon / row.soc_start_kwh
        incoming_energy = row.grid_import_kw - row.grid_import_loss_kw + row.pv_used_kw - row.pv_loss_kw + row.battery_discharge_kw - row.battery_discharge_loss_kw
        incoming_carbon = row.grid_import_kw * row.carbon_kg_per_kwh + row.battery_discharge_kw * storage_intensity
        ci3 = incoming_carbon / incoming_energy
        ci4 = ci3 * row.main_flow_kw / (row.main_flow_kw - row.main_loss_kw)
        ci5 = ci4 * row.hvac_edge_kw / row.hvac_load_kw
        ci6 = ci4 * row.fixed_edge_kw / row.fixed_load_kw
        discharge_carbon = row.battery_discharge_kw * storage_intensity
        charge_carbon = row.battery_charge_bus_kw * ci3
        next_storage_carbon = storage_carbon - discharge_carbon + charge_carbon
        load_carbon = ci5 * row.hvac_load_kw + ci6 * row.fixed_load_kw
        export_carbon = ci3 * row.grid_export_bus_kw
        carbon_residuals.append(storage_carbon + row.grid_import_kw * row.carbon_kg_per_kwh - next_storage_carbon - load_carbon - export_carbon)
        node5_intensity.append(ci5)
        node6_intensity.append(ci6)
        attributed_load_carbon += load_carbon
        attributed_export_carbon += export_carbon
        storage_carbon = next_storage_carbon
    source_carbon = initial_storage_carbon + float((frame["grid_import_kw"] * frame["carbon_kg_per_kwh"]).sum())
    carbon_error = float(np.abs(carbon_residuals).sum() / source_carbon)
    node5_intensity = np.asarray(node5_intensity)
    node6_intensity = np.asarray(node6_intensity)
    grid_net = frame["grid_import_kw"] - frame["grid_export_delivered_kw"]
    return {
        "hours": int(len(frame)),
        "annual_operating_cost_usd": float((grid_net * frame["cost_usd_per_kwh"]).sum()),
        "annual_emissions_kgco2e": float((grid_net * frame["carbon_kg_per_kwh"]).sum()),
        "pv_self_consumption_fraction": float(frame["pv_used_kw"].sum() / frame["pv_dc_kw"].sum()),
        "comfort_violation_hours": int(((frame["temperature_end_c"] < 22 - 1e-7) | (frame["temperature_end_c"] > 26 + 1e-7)).sum()),
        "peak_grid_import_kw": float(frame["grid_import_kw"].max()),
        "battery_throughput_kwh": float(0.5 * (frame["battery_charge_bus_kw"].sum() + frame["battery_discharge_kw"].sum())),
        "energy_closure_relative_error": energy_error,
        "carbon_closure_relative_error": carbon_error,
        "nodal_p95_abs_diff_kg_per_kwh": float(np.quantile(np.abs(node5_intensity - frame["carbon_kg_per_kwh"].to_numpy()), 0.95)),
        "node5_carbon_intensity_mean": float(node5_intensity.mean()),
        "node6_carbon_intensity_mean": float(node6_intensity.mean()),
        "attributed_load_carbon_kg": float(attributed_load_carbon),
        "attributed_export_carbon_kg": float(attributed_export_carbon),
        "initial_storage_carbon_kg": float(initial_storage_carbon),
        "final_storage_carbon_kg": float(storage_carbon),
    }


def pareto_status(metrics: dict[str, dict]) -> dict[str, bool]:
    result = {}
    for policy, value in metrics.items():
        cost = value["annual_operating_cost_usd"]
        emissions = value["annual_emissions_kgco2e"]
        result[policy] = not any(
            other != policy
            and candidate["annual_operating_cost_usd"] <= cost
            and candidate["annual_emissions_kgco2e"] <= emissions
            and (
                candidate["annual_operating_cost_usd"] < cost
                or candidate["annual_emissions_kgco2e"] < emissions
            )
            for other, candidate in metrics.items()
        )
    return result


def distinct_nondominated_count(metrics: dict[str, dict], nondominated: dict[str, bool], threshold: float = 0.001) -> int:
    retained = []
    for policy in sorted((name for name, keep in nondominated.items() if keep), key=lambda name: metrics[name]["annual_operating_cost_usd"]):
        value = metrics[policy]
        if all(
            max(
                abs(value["annual_operating_cost_usd"] - metrics[other]["annual_operating_cost_usd"])
                / max(abs(value["annual_operating_cost_usd"]), abs(metrics[other]["annual_operating_cost_usd"]), 1.0),
                abs(value["annual_emissions_kgco2e"] - metrics[other]["annual_emissions_kgco2e"])
                / max(abs(value["annual_emissions_kgco2e"]), abs(metrics[other]["annual_emissions_kgco2e"]), 1.0),
            )
            >= threshold
            for other in retained
        ):
            retained.append(policy)
    return len(retained)


def select_practical_policy(metrics: dict[str, dict]) -> str | None:
    baseline_cost = metrics["F0"]["annual_operating_cost_usd"]
    eligible = [
        policy
        for policy in ("F1", "F2", "F3", "F4", "F5")
        if metrics[policy]["annual_operating_cost_usd"] / baseline_cost - 1 <= 0.05
    ]
    return eligible[-1] if eligible else None


def select_knee(metrics: dict[str, dict], nondominated: dict[str, bool]) -> str | None:
    policies = sorted((name for name, keep in nondominated.items() if keep), key=lambda name: metrics[name]["annual_operating_cost_usd"])
    if not policies:
        return None
    if len(policies) <= 2:
        return policies[0]
    costs = np.array([metrics[name]["annual_operating_cost_usd"] for name in policies])
    emissions = np.array([metrics[name]["annual_emissions_kgco2e"] for name in policies])
    x = (costs - costs.min()) / (costs.max() - costs.min())
    y = (emissions - emissions.min()) / (emissions.max() - emissions.min())
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    distances = np.abs(line[0] * (start[1] - y) - (start[0] - x) * line[1]) / np.linalg.norm(line)
    return policies[int(np.argmax(distances))]


def exact_shapley(values: dict[frozenset, float], features=("A", "B", "C")) -> dict[str, float]:
    count = len(features)
    result = {}
    universe = set(features)
    for feature in features:
        contribution = 0.0
        others = sorted(universe - {feature})
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            for subset in combinations(others, size):
                coalition = frozenset(subset)
                contribution += weight * (values[coalition | {feature}] - values[coalition])
        result[feature] = contribution
    return result


def harsanyi_dividends(values: dict[frozenset, float], features=("A", "B", "C")) -> dict[str, float]:
    dividends = {}
    ordered = [frozenset()] + [frozenset(group) for size in range(1, len(features) + 1) for group in combinations(features, size)]
    for coalition in ordered:
        dividends[coalition] = values[coalition] - sum(value for subset, value in dividends.items() if subset < coalition)
    return {"empty" if not coalition else "".join(sorted(coalition)): value for coalition, value in dividends.items()}
