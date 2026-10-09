from pathlib import Path
import json
import pickle
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from scripts.run_main import _verify_hash
from scripts.run_pareto import _reference_cost_from_f5_cap
from scripts.run_shapley import COALITIONS, FULL, _run_day
from src.data_pipeline import file_sha256, load_inputs
from src.dispatch import load_parameters
from src.evaluate import evaluate_dispatch
from src.model import build_hvac_envelope


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    pareto_root = root / "artifacts" / "experiment" / "pareto_2026-08-02"
    pareto = json.loads((pareto_root / "pareto_summary.json").read_text(encoding="utf-8"))
    verification = json.loads((pareto_root / "verification_report.json").read_text(encoding="utf-8"))
    expected_f4_hash = verification["sha256"]["F4_dispatch.csv"].lower()
    source_f4_hash = file_sha256(pareto_root / "F4_dispatch.csv").lower()
    if not pareto["passed_for_shapley"] or pareto["practical_policy"] != "F4" or source_f4_hash != expected_f4_hash:
        raise RuntimeError("verified F4 prerequisite failed")
    with (pareto_root / "checkpoint.pkl").open("rb") as handle:
        pareto_checkpoint = pickle.load(handle)
    if len(pareto_checkpoint) != 365:
        raise RuntimeError("incomplete Pareto checkpoint")
    source_records = json.loads((root / "artifacts" / "experiment" / "main_2026-08-02" / "solver_records.json").read_text(encoding="utf-8"))
    full_reference_costs = [_reference_cost_from_f5_cap(record["planned_cost_cap"]) for record in source_records]
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    envelope = build_hvac_envelope(inputs.frame)
    annual_carbon_mean = float(inputs.frame["carbon_kg_per_kwh"].mean())
    output = root / "artifacts" / "experiment" / "shapley_smoke_2026-08-02"
    output.mkdir(parents=True, exist_ok=True)

    reconstructed = pd.concat([item["dispatch"]["F4"] for item in pareto_checkpoint], ignore_index=True)
    reconstructed_path = output / "F4_reconstructed_from_checkpoint.csv"
    reconstructed.to_csv(reconstructed_path, index=False)
    reconstructed_hash = file_sha256(reconstructed_path).lower()

    c0_results = []
    for day in (0, 182, 364):
        c0_results.append(
            _run_day(
                day,
                inputs.frame,
                envelope,
                parameters,
                full_reference_costs[day],
                annual_carbon_mean,
                COALITIONS,
                False,
            )
        )
        print(f"C0_day={day}_complete", flush=True)
    for coalition in COALITIONS:
        pd.concat([item["dispatch"][coalition] for item in c0_results], ignore_index=True).to_csv(
            output / f"C0_{coalition}_dispatch.csv", index=False
        )
    c0_physical = {
        coalition: evaluate_dispatch(pd.concat([item["dispatch"][coalition] for item in c0_results], ignore_index=True))
        for coalition in COALITIONS
    }
    c0_full_identity = {
        str(item["day"]): item["dispatch"][FULL].equals(pareto_checkpoint[item["day"]]["dispatch"]["F4"])
        for item in c0_results
    }

    c1_results = []
    c1_identity = {}
    for day in range(190, 197):
        result = _run_day(
            day,
            inputs.frame,
            envelope,
            parameters,
            full_reference_costs[day],
            annual_carbon_mean,
            [FULL],
            False,
        )
        expected = pareto_checkpoint[day]["dispatch"]["F4"]
        actual = result["dispatch"][FULL]
        numeric_columns = actual.select_dtypes(include=[np.number]).columns
        maximum_difference = max(float(np.max(np.abs(actual[column].to_numpy() - expected[column].to_numpy()))) for column in numeric_columns)
        c1_identity[str(day)] = {"dataframe_bitwise_equal": actual.equals(expected), "maximum_absolute_numeric_difference": maximum_difference}
        c1_results.append(result)
        print(f"C1_day={day}_complete", flush=True)
    c1_actual = pd.concat([item["dispatch"][FULL] for item in c1_results], ignore_index=True)
    c1_expected = pd.concat([pareto_checkpoint[day]["dispatch"]["F4"] for day in range(190, 197)], ignore_index=True)
    c1_actual.to_csv(output / "C1_F4_rerun_dispatch.csv", index=False)
    c1_expected.to_csv(output / "C1_F4_expected_dispatch.csv", index=False)

    c0_feasible = len(c0_results) == 3 and all(set(item["dispatch"]) == set(COALITIONS) for item in c0_results)
    c1_exact = (
        all(c0_full_identity.values())
        and all(value["dataframe_bitwise_equal"] for value in c1_identity.values())
        and file_sha256(output / "C1_F4_rerun_dispatch.csv") == file_sha256(output / "C1_F4_expected_dispatch.csv")
        and reconstructed_hash == source_f4_hash == expected_f4_hash
    )
    gates = {"C0_all_coalitions_feasible": c0_feasible, "C1_F4_identity": c1_exact}
    solver_records = {
        "C0": [{"day": item["day"], "references": item["references"], "solver": item["solver"]} for item in c0_results],
        "C1": [{"day": item["day"], "references": item["references"], "solver": item["solver"]} for item in c1_results],
    }
    (output / "solver_records.json").write_text(json.dumps(solver_records, ensure_ascii=False), encoding="utf-8")
    summary = {
        "status": "coalition_smoke_test",
        "days": {"C0": [0, 182, 364], "C1_P0_window": list(range(190, 197))},
        "gates": gates,
        "passed": all(gates.values()),
        "C0_full_identity": c0_full_identity,
        "C0_physical_metrics": c0_physical,
        "C1_identity": c1_identity,
        "F4_sha256": {"verification_record": expected_f4_hash, "source": source_f4_hash, "checkpoint_reconstruction": reconstructed_hash},
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "command": "python -m scripts.run_shapley_smoke",
    }
    (output / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "passed": summary["passed"], "F4_sha256": summary["F4_sha256"]}, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise RuntimeError("coalition smoke gate failed")


if __name__ == "__main__":
    main()
