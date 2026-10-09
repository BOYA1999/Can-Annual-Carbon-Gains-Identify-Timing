from argparse import ArgumentParser
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from src.dispatch import load_parameters


def node_voltages_pu(frame, parameters, multiplier=1.0):
    resistance = parameters["network"]["branch_resistance_ohm"]

    def drop(power_kw, edge, voltage_v):
        return 1000.0 * multiplier * resistance[edge] * power_kw / voltage_v**2

    v3 = 1.0 - drop(frame["grid_import_kw"] - frame["grid_export_bus_kw"], "n0-n3", 750.0)
    v1 = v3 + drop(frame["pv_used_kw"], "n1-n3", 750.0)
    v2 = v3 + drop(frame["battery_discharge_kw"] - frame["battery_charge_bus_kw"], "n2-n3", 750.0)
    v4 = v3 - drop(frame["main_flow_kw"], "n3-n4", 380.0)
    v5 = v4 - drop(frame["hvac_edge_kw"], "n4-n5", 380.0)
    v6 = v4 - drop(frame["fixed_edge_kw"], "n4-n6", 48.0)
    return pd.DataFrame({"n0": np.ones(len(frame)), "n1": v1, "n2": v2, "n3": v3, "n4": v4, "n5": v5, "n6": v6})


def critical_multiplier(voltages):
    slopes = voltages.to_numpy(float) - 1.0
    lower = np.divide(-0.05, slopes, out=np.full_like(slopes, np.inf), where=slopes < 0)
    upper = np.divide(0.05, slopes, out=np.full_like(slopes, np.inf), where=slopes > 0)
    return float(np.min(np.minimum(lower, upper)))


def audit(input_dir, root):
    parameters = load_parameters(root)
    settings = parameters["network"]["voltage_audit"]
    lower = settings["lower_pu"]
    upper = settings["upper_pu"]
    pattern = re.compile(r"trajectory_(.+)_([0-9.]+)_(F[04])\.csv")
    cases = []
    sweeps = []
    all_base = []
    for path in sorted(input_dir.glob("trajectory_*_*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        frame = pd.read_csv(path)
        base = node_voltages_pu(frame, parameters)
        all_base.append(base)
        cases.append({
            "season": match.group(1),
            "soc_fraction": float(match.group(2)),
            "method": match.group(3),
            "hours": len(frame),
            "minimum_voltage_pu": float(base.min().min()),
            "maximum_voltage_pu": float(base.max().max()),
            "violating_node_hours": int(((base < lower) | (base > upper)).to_numpy().sum()),
            "critical_common_resistance_multiplier": critical_multiplier(base),
        })
        for multiplier in settings["resistance_multipliers"]:
            voltages = node_voltages_pu(frame, parameters, multiplier)
            sweeps.append({
                "season": match.group(1),
                "soc_fraction": float(match.group(2)),
                "method": match.group(3),
                "resistance_multiplier": multiplier,
                "minimum_voltage_pu": float(voltages.min().min()),
                "maximum_voltage_pu": float(voltages.max().max()),
                "violating_node_hours": int(((voltages < lower) | (voltages > upper)).to_numpy().sum()),
            })
    if not all_base:
        raise RuntimeError(f"No weekly trajectories found in {input_dir}")
    combined = pd.concat(all_base, ignore_index=True)
    summary = {
        "status": "posterior_full_topology_voltage_audit",
        "trajectory_count": len(cases),
        "schedule_hours": int(sum(item["hours"] for item in cases)),
        "node_count": 7,
        "branch_count": 6,
        "base_resistance_ohm_per_branch": 0.001,
        "voltage_interval_pu": [lower, upper],
        "minimum_voltage_pu": float(combined.min().min()),
        "minimum_voltage_node": str(combined.min().idxmin()),
        "maximum_voltage_pu": float(combined.max().max()),
        "maximum_voltage_node": str(combined.max().idxmax()),
        "violating_node_hours": int(((combined < lower) | (combined > upper)).to_numpy().sum()),
        "critical_common_resistance_multiplier": min(item["critical_common_resistance_multiplier"] for item in cases),
        "boundary": settings["boundary"],
    }
    return pd.DataFrame(cases), pd.DataFrame(sweeps), summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "artifacts" / "weekly_electrical_diagnostic")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "network_voltage_audit")
    args = parser.parse_args()
    cases, sweeps, summary = audit(args.input, root)
    args.output.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.output / "case_metrics.csv", index=False)
    sweeps.to_csv(args.output / "resistance_sweep.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
