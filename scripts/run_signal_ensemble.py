import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from scripts.run_main import _controls, _slice_envelope, _verify_hash
from src.data_pipeline import load_inputs
from src.dispatch import load_parameters, replay_day, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope


SHIFTS = (1, 3, 7, 14, 28)
SEEDS = tuple(range(20260802, 20260822))
STRATA = ("month", "weekday", "month_weekday")


def _specs():
    result = [{"id": f"SHIFT_{days:02d}D", "kind": "shift", "days": days, "seed": None} for days in SHIFTS]
    result.extend(
        {"id": f"{kind.upper()}_{seed}", "kind": kind, "days": None, "seed": seed}
        for kind in STRATA
        for seed in SEEDS
    )
    return result


def _day_mapping(frame, spec):
    base = (np.arange(365) - 1) % 365
    if spec["kind"] == "shift":
        return (base - spec["days"]) % 365
    dates = pd.to_datetime(frame.iloc[base * 24].timestamp).reset_index(drop=True)
    labels = {
        "month": dates.dt.month.astype(str),
        "weekday": dates.dt.weekday.astype(str),
        "month_weekday": dates.dt.month.astype(str) + "_" + dates.dt.weekday.astype(str),
    }[spec["kind"]]
    mapping = base.copy()
    rng = np.random.default_rng(spec["seed"])
    for label in labels.unique():
        positions = np.flatnonzero(labels.to_numpy() == label)
        mapping[positions] = rng.permutation(base[positions])
    return mapping


def _daily_metrics(frame):
    net = frame.grid_import_kw - frame.grid_export_delivered_kw
    return {
        "cost_usd": float((frame.cost_usd_per_kwh * net).sum()),
        "emissions_kgco2": float((frame.carbon_kg_per_kwh * net).sum()),
    }


def _worker(root_text, spec):
    root = Path(root_text)
    output = root / "artifacts" / "experiment" / "signal_ensemble_2026-08-02" / spec["id"]
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    records = json.loads((root / "artifacts" / "experiment" / "shapley_relative_2026-08-02" / "solver_records.json").read_text(encoding="utf-8"))
    mapping = _day_mapping(inputs.frame, spec)
    carbon = inputs.frame.carbon_kg_per_kwh.to_numpy()
    planned = np.concatenate([carbon[day * 24:(day + 1) * 24] for day in mapping])
    target = carbon

    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        start, parts, daily, solver = saved["next_day"], saved["parts"], saved["daily"], saved["solver"]
    else:
        start, parts, daily, solver = 0, [], [], []
    for day in range(start, 365):
        indices = np.arange(day * 24, (day + 1) * 24)
        previous = (indices - 24) % 8760
        source = mapping[day] * 24 + np.arange(24)
        actual = inputs.frame.iloc[indices].reset_index(drop=True)
        forecast = actual.copy()
        forecast["pv_dc_kw"] = inputs.frame.iloc[previous].pv_dc_kw.to_numpy()
        forecast["carbon_kg_per_kwh"] = inputs.frame.iloc[source].carbon_kg_per_kwh.to_numpy()
        cap = records[day]["references"]["B1C1"]["planned_cost_cap_usd"]
        plan = solve_day(
            forecast,
            _slice_envelope(envelope, indices, previous),
            parameters,
            objective="emissions",
            cost_cap_usd=cap,
        )
        realized = replay_day(actual, _slice_envelope(envelope, indices), parameters, _controls(plan.frame))
        parts.append(realized)
        daily.append({"day": day, **_daily_metrics(realized)})
        solver.append({"day": day, **plan.solver})
        if (day + 1) % 25 == 0 or day == 364:
            temporary = output / "checkpoint.tmp"
            with temporary.open("wb") as handle:
                pickle.dump({"next_day": day + 1, "parts": parts, "daily": daily, "solver": solver}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(checkpoint)
            print(f"scenario={spec['id']} days={day + 1}/365", flush=True)

    dispatch = pd.concat(parts, ignore_index=True)
    metrics = evaluate_dispatch(dispatch)
    pd.DataFrame(daily).to_csv(output / "daily_metrics.csv", index=False)
    pd.DataFrame({"target_day": np.arange(365), "source_day": mapping}).to_csv(output / "source_day_mapping.csv", index=False)
    result = {
        **spec,
        "status": "completed",
        "hours": len(dispatch),
        "new_plan_solves": len(solver),
        "all_solvers_optimal": all(item["status"] == 0 for item in solver),
        "same_sorted_distribution_as_reference": bool(np.array_equal(np.sort(planned), np.sort(carbon))),
        "forecast_realized_pearson_r": float(np.corrcoef(planned, target)[0, 1]),
        "forecast_realized_mae_kg_per_kwh": float(np.mean(np.abs(planned - target))),
        "forecast_realized_rmse_kg_per_kwh": float(np.sqrt(np.mean((planned - target) ** 2))),
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _block_interval(values, block, samples=4000, seed=20260802):
    rng = np.random.default_rng(seed + block)
    count = len(values)
    starts = rng.integers(0, count, size=(samples, int(np.ceil(count / block))))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets) % count
    means = values[indices.reshape(samples, -1)[:, :count]].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(
        root / "config" / "extended_run_contract.md",
        root / "config" / "extended_run_contract.sha256",
    )
    output = root / "artifacts" / "experiment" / "signal_ensemble_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)
    specs = _specs()
    results = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_worker, str(root), spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failure = {**spec, "status": "failed", "error": str(error)}
                results.append(failure)
                (output / f"failure_{spec['id']}.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"ensemble_completed={len(results)}/{len(specs)} scenario={spec['id']} status={results[-1]['status']}", flush=True)

    source = json.loads((root / "artifacts" / "experiment" / "adversarial_signals_2026-08-02" / "signal_ablation_summary.json").read_text(encoding="utf-8"))
    dynamic_metrics = source["metrics"]["DYNAMIC"]
    dynamic_daily = pd.read_csv(root / "artifacts" / "experiment" / "adversarial_signals_2026-08-02" / "daily_metrics.csv")
    dynamic_daily = dynamic_daily[dynamic_daily.method == "DYNAMIC"].sort_values("day")
    rows = []
    daily_differences = []
    for result in sorted(results, key=lambda item: item["id"]):
        if result["status"] != "completed":
            rows.append(result)
            continue
        penalty = result["metrics"]["annual_emissions_kgco2e"] / dynamic_metrics["annual_emissions_kgco2e"] - 1
        cost = result["metrics"]["annual_operating_cost_usd"] / dynamic_metrics["annual_operating_cost_usd"] - 1
        daily = pd.read_csv(output / result["id"] / "daily_metrics.csv").sort_values("day")
        difference = daily.emissions_kgco2.to_numpy() - dynamic_daily.emissions_kgco2e.to_numpy()
        daily_differences.append(difference)
        rows.append({
            **{key: result[key] for key in ("id", "kind", "days", "seed", "status", "forecast_realized_pearson_r", "forecast_realized_mae_kg_per_kwh", "forecast_realized_rmse_kg_per_kwh", "same_sorted_distribution_as_reference", "all_solvers_optimal", "hours", "new_plan_solves")},
            "cost_penalty_fraction": cost,
            "emissions_penalty_fraction": penalty,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "ensemble_results.csv", index=False)
    completed = table[table.status == "completed"].copy()
    median_daily = np.median(np.vstack(daily_differences), axis=0) if daily_differences else np.array([])
    intervals = {str(block): _block_interval(median_daily, block) for block in (1, 7, 14, 28)} if len(median_daily) else {}
    positive_fraction = float((completed.emissions_penalty_fraction > 0).mean()) if len(completed) else 0.0
    support = bool(len(completed) == len(specs) and completed.emissions_penalty_fraction.median() > 0 and positive_fraction > 0.80)
    gates = {
        "complete_preregistered_denominator": len(completed) == len(specs),
        "all_distribution_invariants": bool(completed.same_sorted_distribution_as_reference.all()) if len(completed) else False,
        "all_solver_denominators_complete": bool(((completed.hours == 8760) & (completed.new_plan_solves == 365) & completed.all_solvers_optimal).all()) if len(completed) else False,
    }
    summary = {
        "status": "full_year_matched_signal_ensemble",
        "preregistered_surrogates": len(specs),
        "completed_surrogates": len(completed),
        "failed_surrogates": int((table.status == "failed").sum()),
        "emissions_penalty_fraction_distribution": {
            "minimum": float(completed.emissions_penalty_fraction.min()) if len(completed) else None,
            "q25": float(completed.emissions_penalty_fraction.quantile(0.25)) if len(completed) else None,
            "median": float(completed.emissions_penalty_fraction.median()) if len(completed) else None,
            "q75": float(completed.emissions_penalty_fraction.quantile(0.75)) if len(completed) else None,
            "maximum": float(completed.emissions_penalty_fraction.max()) if len(completed) else None,
            "positive_fraction": positive_fraction,
        },
        "median_daily_difference_95pct_circular_block_resampling_intervals_kgco2": intervals,
        "timing_support_gate": support,
        "integrity_gates": gates,
        "passed": all(gates.values()),
        "contract_sha256": contract_hash,
        "command": "python -m scripts.run_signal_ensemble",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("signal-ensemble integrity gate failed")


if __name__ == "__main__":
    main()
