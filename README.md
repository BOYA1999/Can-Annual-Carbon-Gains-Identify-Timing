# DC Microgrid Dispatch Reproducibility Package

This repository contains the public data acquisition code, fixed numerical configuration, dispatch implementation, evaluation routines, and verification tests for a synthetic DC microgrid benchmark.

The repository does not contain downloaded data. Run all commands from the repository root with Python 3.12.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data acquisition

The download script retrieves the California archive from NLR Dataset 205, records the public license page, requests the fixed hourly PVWatts V8 profile, verifies the archive checksum, and extracts the archive.

```powershell
python -m scripts.download_data
python -m scripts.audit_data
```

The PVWatts request uses the public `DEMO_KEY`. Replace it through the `NREL_API_KEY` environment variable only if the public quota is unavailable.

## Fast verification

```powershell
python -m pytest -q
python -m scripts.run_pilot
```

The test suite checks input dimensions, units, model constraints, dispatch feasibility, metric definitions, attribution identities, and fixed perturbation grids. The pilot executes 168 hourly records through the optimizer and evaluator.

## Complete computation

Run the following commands in order. Checkpoints are written under `artifacts/experiment` and allow interrupted numerical jobs to resume.

```powershell
python -m scripts.run_main
python -m scripts.run_hourly_audit
python -m scripts.run_pareto
python -m scripts.run_shapley_smoke
python -m scripts.run_shapley
python -m scripts.run_sensitivity
python -m scripts.run_adversarial_signals
python -m scripts.run_budget_ablation --allowance 0.01
python -m scripts.run_budget_ablation --allowance 0.08
python -m scripts.run_budget_ablation --summarize
python -m scripts.run_failure_boundary
python -m scripts.verify_phase2e
```

No GPU is required. The optimizer uses SciPy MILP with the HiGHS backend. Full execution is CPU intensive and writes intermediate CSV, JSON, and pickle files under the ignored `artifacts` directory.

## Integrity

`config/system_parameters.sha256` protects the numerical parameter file. `config/run_contract.sha256` protects the frozen data, denominator, threshold, random seed, and scenario definitions. `MANIFEST.sha256` protects the distributable files.

Public source addresses and license boundaries are recorded in `DATA_SOURCES.md`.

