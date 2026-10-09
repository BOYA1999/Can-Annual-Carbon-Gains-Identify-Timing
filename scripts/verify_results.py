from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.run_main import _verify_hash
from scripts.run_shapley import COALITIONS
from src.data_pipeline import file_sha256


def _rows(path):
    with path.open("rb") as handle:
        return sum(1 for _ in handle) - 1


def main():
    root = Path(__file__).resolve().parents[1]
    contract_hash = _verify_hash(root / "config" / "run_contract.json", root / "config" / "run_contract.sha256")
    parameter_hash = _verify_hash(root / "config" / "system_parameters.json", root / "config" / "system_parameters.sha256")
    experiment = root / "artifacts" / "experiment"

    e1_root = experiment / "adversarial_signals_2026-08-02"
    e1 = json.loads((e1_root / "signal_ablation_summary.json").read_text(encoding="utf-8"))
    e1_rows = {method: _rows(e1_root / f"{method}_dispatch.csv") for method in e1["methods"]}
    planned = pd.read_csv(e1_root / "planned_carbon_signals.csv")
    e1_checks = {
        "summary_passed": e1["passed"],
        "contract_hash": e1["contract_sha256"] == contract_hash,
        "parameter_hash": e1["parameter_sha256"] == parameter_hash,
        "all_dispatch_rows_8760": all(value == 8760 for value in e1_rows.values()),
        "daily_rows_1460": _rows(e1_root / "daily_metrics.csv") == 4 * 365,
        "new_solver_count_730": e1["new_plan_solves"] == 730 and e1["nonoptimal_new_solver_records"] == 0,
        "attached_hashes_match": e1["source_sha256"] == e1["attached_sha256"],
        "shift_distribution_exact": bool(np.array_equal(np.sort(planned.DYNAMIC), np.sort(planned.SHIFT_7D))),
        "permutation_distribution_exact": bool(np.array_equal(np.sort(planned.DYNAMIC), np.sort(planned.PERMUTED_DAYS))),
    }

    e2_root = experiment / "adversarial_budget_curve_2026-08-02"
    e2 = json.loads((e2_root / "budget_curve_summary.json").read_text(encoding="utf-8"))
    e2_summaries = {}
    e2_rows = {}
    for allowance in (0.01, 0.08):
        tag = f"{int(100 * allowance):02d}"
        path = experiment / f"adversarial_budget_{tag}_2026-08-02"
        summary = json.loads((path / "budget_ablation_summary.json").read_text(encoding="utf-8"))
        e2_summaries[str(allowance)] = summary
        e2_rows[str(allowance)] = {coalition: _rows(path / f"{coalition}_dispatch.csv") for coalition in COALITIONS}
    curve = pd.read_csv(e2_root / "shapley_budget_curve.csv")
    e2_checks = {
        "combined_summary_passed": e2["passed"],
        "all_new_summaries_passed": all(value["passed"] for value in e2_summaries.values()),
        "contract_hash": all(value["contract_sha256"] == contract_hash for value in e2_summaries.values()),
        "parameter_hash": all(value["parameter_sha256"] == parameter_hash for value in e2_summaries.values()),
        "all_new_dispatch_rows_8760": all(rows == 8760 for table in e2_rows.values() for rows in table.values()),
        "new_solver_count_5840": sum(value["new_plan_solves"] for value in e2_summaries.values()) == 5840,
        "all_new_solvers_optimal": sum(value["nonoptimal_solver_records"] for value in e2_summaries.values()) == 0,
        "curve_has_9_rows": len(curve) == 9 and set(curve.allowance_fraction) == {0.01, 0.04, 0.08},
        "curve_finite": bool(np.isfinite(curve.select_dtypes(include=[np.number]).to_numpy()).all()),
        "shapley_closure": all(value["attribution"]["emissions_utility_kgco2e"]["closure_relative_error"] <= 0.005 for value in e2_summaries.values()),
    }

    e3_root = experiment / "forecast_boundary_2026-08-02"
    e3 = json.loads((e3_root / "failure_boundary_summary.json").read_text(encoding="utf-8"))
    boundary = pd.read_csv(e3_root / "forecast_boundary_curve.csv")
    expected_levels = {0.0, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20}
    failure_rows = boundary[boundary.status == "failed"]
    e3_checks = {
        "summary_passed": e3["passed"],
        "contract_hash": e3["contract_sha256"] == contract_hash,
        "parameter_hash": e3["parameter_sha256"] == parameter_hash,
        "boundary_has_19_rows": len(boundary) == 19,
        "boundary_levels_exact": set(boundary.level) == expected_levels,
        "new_scenario_count_9": e3["new_completed_scenarios"] + e3["new_failed_scenarios"] == 9,
        "new_failure_count_4": e3["new_failed_scenarios"] == 4,
        "all_failed_rows_have_denominators": bool(failure_rows.completed_hours.notna().all() and failure_rows.failure_method.notna().all()),
        "all_seed_boundaries_recorded": set(e3["first_failure_by_seed"]) == {"20260802", "20260803", "20260804"},
    }

    checks = {"E1_matched_signals": e1_checks, "E2_budget_structure": e2_checks, "E3_failure_boundary": e3_checks}
    passed = all(value for group in checks.values() for value in group.values())
    report = {
        "status": "independent_artifact_verification",
        "passed": passed,
        "checks": checks,
        "dispatch_rows": {"E1": e1_rows, "E2": e2_rows},
        "metric_directions_not_used_as_validation_gates": {
            "E1_timing_conclusion": e1["timing_conclusion"],
            "E2_feature_sign_changes": e2["feature_sign_changes"],
            "E3_first_failure_by_seed": e3["first_failure_by_seed"],
        },
        "sha256": {
            "E1_summary": file_sha256(e1_root / "signal_ablation_summary.json"),
            "E2_curve_summary": file_sha256(e2_root / "budget_curve_summary.json"),
            "E3_summary": file_sha256(e3_root / "failure_boundary_summary.json"),
        },
        "contract_sha256": contract_hash,
        "parameter_sha256": parameter_hash,
        "command": "python -m scripts.verify_results",
    }
    destination = experiment / "verification_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": checks}, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("independent result verification failed")


if __name__ == "__main__":
    main()
