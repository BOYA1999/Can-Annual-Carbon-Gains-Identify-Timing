from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.run_main import _controls, _slice_envelope, _verify_hash
from scripts.run_sensitivity import WINDOWS
from src.data_pipeline import load_inputs
from src.dispatch import _loss_value, load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope, converter_loss_exact, converter_tangents, line_loss_exact, line_loss_tangents


POINTS = {
    3: (0.1, 0.5, 1.0),
    5: tuple(np.linspace(0.1, 1.0, 5)),
    9: tuple(np.linspace(0.1, 1.0, 9)),
}


def _curve_rows():
    rows = []
    for fraction in np.linspace(0.0, 1.0, 10001):
        converter_exact = converter_loss_exact(fraction, 1.0)
        line_exact = line_loss_exact(fraction)
        for count, points in POINTS.items():
            converter_lines = converter_tangents(1.0, fractions=points)
            line_points = (0.1, 0.4, 0.7, 1.0) if count == 3 else points
            line_lines = line_loss_tangents(1.0, fractions=line_points)
            converter_tangent = 0.0 if fraction == 0 else max(slope * fraction + intercept for slope, intercept in converter_lines)
            line_tangent = 0.0 if fraction == 0 else max(0.0, max(slope * fraction + intercept for slope, intercept in line_lines))
            rows.append({
                "fraction": fraction,
                "points": count,
                "converter_exact_fraction": converter_exact,
                "converter_tangent_fraction": converter_tangent,
                "converter_underestimate_fraction": converter_exact - converter_tangent,
                "line_exact_fraction": line_exact,
                "line_tangent_fraction": line_tangent,
                "line_underestimate_fraction": line_exact - line_tangent,
            })
    return pd.DataFrame(rows)


def _curve_summary(curves):
    rows = []
    for count, group in curves.groupby("points"):
        active = group[group.fraction > 0]
        for kind in ("converter", "line"):
            exact = active[f"{kind}_exact_fraction"].to_numpy()
            error = active[f"{kind}_underestimate_fraction"].to_numpy()
            rows.append({
                "kind": kind,
                "points": int(count),
                "maximum_absolute_underestimate_per_unit_rating": float(error.max()),
                "maximum_relative_underestimate": float(np.max(error / np.maximum(exact, 1e-12))),
                "minimum_underestimate_per_unit_rating": float(error.min()),
            })
    return pd.DataFrame(rows)


def _run_seasonal(frame, envelope, base_parameters):
    output = []
    solver = []
    for count, points in POINTS.items():
        parameters = deepcopy(base_parameters)
        parameters["network"]["converter_tangent_fractions"] = list(points)
        parameters["network"]["line_tangent_fractions"] = list((0.1, 0.4, 0.7, 1.0) if count == 3 else points)
        for season, timestamp in WINDOWS.items():
            first = int(frame.index[frame.timestamp == pd.Timestamp(timestamp)][0])
            dispatch = {(method, loss): [] for method in ("F0", "F4") for loss in ("tangent", "exact")}
            for day in range(first // 24, first // 24 + 7):
                indices = np.arange(day * 24, (day + 1) * 24)
                previous = (indices - 24) % 8760
                actual = frame.iloc[indices].reset_index(drop=True)
                forecast = actual.copy()
                forecast["pv_dc_kw"] = frame.iloc[previous].pv_dc_kw.to_numpy()
                forecast["carbon_kg_per_kwh"] = frame.iloc[previous].carbon_kg_per_kwh.to_numpy()
                actual_envelope = _slice_envelope(envelope, indices)
                forecast_envelope = _slice_envelope(envelope, indices, previous)
                f0 = solve_day(forecast, forecast_envelope, parameters, objective="cost")
                cap = f0.cost_usd + 0.04 * abs(f0.cost_usd)
                f4 = solve_day(forecast, forecast_envelope, parameters, objective="emissions", cost_cap_usd=cap)
                for method, plan in (("F0", f0), ("F4", f4)):
                    controls = _controls(plan.frame)
                    dispatch[(method, "tangent")].append(replay_day(actual, actual_envelope, parameters, controls))
                    dispatch[(method, "exact")].append(replay_day(actual, actual_envelope, parameters, controls, loss_model="exact"))
                solver.append({"points": count, "season": season, "day": day, "F0": f0.solver, "F4": f4.solver})
            for (method, loss), parts in dispatch.items():
                result = pd.concat(parts, ignore_index=True)
                metrics = evaluate_dispatch(result)
                output.append({"points": count, "season": season, "method": method, "replay_loss": loss, **metrics})
    return pd.DataFrame(output), solver


def _annual_exact_replay(root, frame, envelope, parameters, output):
    rows = []
    frames = {}
    source_root = root / "artifacts" / "experiment" / "pareto_2026-08-02"
    for method in ("F0", "F4"):
        source = pd.read_csv(source_root / f"{method}_dispatch.csv")
        exact_parts = []
        for day in range(365):
            indices = np.arange(day * 24, (day + 1) * 24)
            exact_parts.append(replay_day(
                frame.iloc[indices].reset_index(drop=True),
                _slice_envelope(envelope, indices),
                parameters,
                _controls(source.iloc[indices].reset_index(drop=True)),
                loss_model="exact",
            ))
        exact = pd.concat(exact_parts, ignore_index=True)
        exact.to_csv(output / f"annual_{method}_exact_replay.csv", index=False)
        frames[(method, "tangent")] = source
        frames[(method, "exact")] = exact
        for loss, result in (("tangent", source), ("exact", exact)):
            rows.append({"method": method, "replay_loss": loss, **evaluate_dispatch(result)})
    metrics = pd.DataFrame(rows)
    f0 = metrics.set_index(["method", "replay_loss"])
    comparisons = {}
    for loss in ("tangent", "exact"):
        base = f0.loc[("F0", loss)]
        policy = f0.loc[("F4", loss)]
        comparisons[loss] = {
            "F4_minus_F0_cost_fraction": float(policy.annual_operating_cost_usd / base.annual_operating_cost_usd - 1),
            "F4_minus_F0_emissions_fraction": float(policy.annual_emissions_kgco2e / base.annual_emissions_kgco2e - 1),
        }
    return metrics, comparisons


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(
        root / "config" / "extended_run_contract.md",
        root / "config" / "extended_run_contract.sha256",
    )
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    output = root / "artifacts" / "experiment" / "loss_audit_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)

    curves = _curve_rows()
    curve_summary = _curve_summary(curves)
    curves.to_csv(output / "loss_curves.csv", index=False)
    curve_summary.to_csv(output / "loss_error_summary.csv", index=False)
    seasonal, solver = _run_seasonal(inputs.frame, envelope, parameters)
    seasonal.to_csv(output / "seasonal_tangent_sensitivity.csv", index=False)
    (output / "solver_records.json").write_text(json.dumps(solver, ensure_ascii=False), encoding="utf-8")
    annual, comparisons = _annual_exact_replay(root, inputs.frame, envelope, parameters, output)
    annual.to_csv(output / "annual_exact_replay_metrics.csv", index=False)

    zero_loss = {
        "converter_tangent": _loss_value(0.0, 100.0),
        "converter_exact": _loss_value(0.0, 100.0, exact=True),
        "line_tangent": _loss_value(0.0, 100.0, line=True),
        "line_exact": _loss_value(0.0, 100.0, line=True, exact=True),
    }
    gates = {
        "analytic_tangents_are_lower_bounds": bool((curve_summary.minimum_underestimate_per_unit_rating >= -1e-12).all()),
        "zero_power_losses_are_zero": all(value == 0 for value in zero_loss.values()),
        "seasonal_solvers_optimal": all(record[method]["status"] == 0 for record in solver for method in ("F0", "F4")),
        "seasonal_physical_closure": bool((seasonal.energy_closure_relative_error <= 0.005).all() and (seasonal.carbon_closure_relative_error <= 0.005).all()),
        "annual_exact_replay_complete": bool(len(annual) == 4 and (annual.hours == 8760).all()),
    }
    summary = {
        "status": "loss_approximation_audit",
        "curve_summary": curve_summary.to_dict("records"),
        "annual_metrics": annual.to_dict("records"),
        "annual_comparisons": comparisons,
        "zero_power_loss_kw": zero_loss,
        "F4_emissions_direction_stable": all(value["F4_minus_F0_emissions_fraction"] < 0 for value in comparisons.values()),
        "gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "command": "python -m scripts.run_loss_audit",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("loss-audit integrity gate failed")


if __name__ == "__main__":
    main()
