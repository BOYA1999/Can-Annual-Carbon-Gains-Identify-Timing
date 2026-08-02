from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from src.data_pipeline import file_sha256, load_inputs
from src.dispatch import load_parameters, solve_day
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope


root = Path(__file__).resolve().parents[1]
inputs = load_inputs(root)
parameters = load_parameters(root)
envelope_all = build_hvac_envelope(inputs.frame)
start = pd.Timestamp(parameters["pilot"]["start"])
indices = inputs.frame.index[inputs.frame["timestamp"] == start]
if len(indices) != 1:
    raise RuntimeError("pilot start not found")
first = int(indices[0])
output = root / "artifacts" / "experiment" / "pilot_2023-07-10"
output.mkdir(parents=True, exist_ok=True)
results = {"B1": [], "P1": []}
solver_records = []
for day_number in range(7):
    left = first + 24 * day_number
    right = left + 24
    day = inputs.frame.iloc[left:right].reset_index(drop=True)
    envelope = build_hvac_envelope(inputs.frame)
    envelope.baseline_kw = envelope_all.baseline_kw[left:right]
    envelope.fixed_kw = envelope_all.fixed_kw[left:right]
    envelope.lower_kw = envelope_all.lower_kw[left:right]
    envelope.upper_kw = envelope_all.upper_kw[left:right]
    b1 = solve_day(day, envelope, parameters, objective="cost", loss_aware=True)
    cost_cap = b1.cost_usd + 0.05 * abs(b1.cost_usd)
    p1 = solve_day(day, envelope, parameters, objective="emissions", loss_aware=True, cost_cap_usd=cost_cap)
    results["B1"].append(b1.frame)
    results["P1"].append(p1.frame)
    solver_records.append({"day": day_number, "B1": b1.solver, "P1": p1.solver, "B1_cost": b1.cost_usd, "P1_cost_cap": cost_cap})
metrics = {}
for method, frames in results.items():
    dispatch = pd.concat(frames, ignore_index=True)
    dispatch.to_csv(output / f"{method}_dispatch.csv", index=False)
    metrics[method] = evaluate_dispatch(dispatch)
gates = {
    "G1_energy_closure": max(metrics[m]["energy_closure_relative_error"] for m in metrics) <= 0.005,
    "G1_carbon_closure": max(metrics[m]["carbon_closure_relative_error"] for m in metrics) <= 0.005,
    "G1_comfort": all(metrics[m]["comfort_violation_hours"] == 0 for m in metrics),
    "G2_nodal_signal": metrics["P1"]["nodal_p95_abs_diff_kg_per_kwh"] >= 0.01,
}
summary = {
    "status": "pilot_validation_only",
    "period": {"start": str(start), "hours": 168},
    "metrics": metrics,
    "gates": gates,
    "passed": all(gates.values()),
    "solver_records": solver_records,
    "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    "contract_sha256": file_sha256(root / "config" / "run_contract.json"),
    "parameter_sha256": file_sha256(root / "config" / "system_parameters.json"),
    "command": "python -m scripts.run_pilot"
}
with (output / "pilot_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
print(json.dumps({"metrics": metrics, "gates": gates, "passed": summary["passed"]}, ensure_ascii=False, indent=2))
