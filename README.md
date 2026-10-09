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

Run the following commands in order. Long annual jobs write checkpoints under `artifacts/experiment`. Scripts without checkpoint files restart their fixed grid after interruption.

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
python -m scripts.run_loss_audit
python -m scripts.run_signal_ensemble
python -m scripts.run_forecast_diagnostics
python -m scripts.run_state_sensitivity
python -m scripts.run_weekly_electrical_diagnostic
python -m scripts.run_network_voltage_audit
python -m scripts.run_spring_integrality_relaxation
python -m scripts.run_annual_closure_audit
python -m scripts.run_weekly_matched_timing
python -m scripts.verify_results
```

No GPU is required. The optimizer uses SciPy MILP with the HiGHS backend. Full execution is CPU intensive and writes intermediate CSV, JSON, and pickle files under the ignored `artifacts` directory. Experiment entry points write their complete denominator before applying fixed result gates. A nonzero exit after artifact creation therefore indicates a failed numerical gate, not permission to change its grid or discard failed cases.

The weekly electrical diagnostic repeats the 12 fixed seasonal state cases with a 300 s solver limit. The subsequent network audit evaluates all seven nodes and six branches under the frozen synthetic 1 mOhm-per-branch resistance case, then scales every branch resistance to locate the first 0.95 to 1.05 p.u. voltage-boundary violation. This is a posterior check of the declared benchmark topology, not validation of a measured feeder. The spring integrality script relaxes only the converter-mode binaries for the three fixed spring cases. The scripts retain raw solver records and keep the mixed-integer, relaxed, and posterior voltage results separate.

## Integrity

`config/system_parameters.sha256` protects the numerical parameter file. `config/run_contract.sha256` protects the original run definitions. `config/extended_run_contract.sha256` protects the loss, surrogate, forecast, state-horizon, and throughput-price grids. `MANIFEST.sha256` protects the distributable files.

Public source addresses and license boundaries are recorded in `DATA_SOURCES.md`.

The canonical hash in the extended run contract records the original configuration before the posterior voltage fields were added. The current complete JSON is protected by `config/system_parameters.sha256`. Text checkouts use LF line endings, except the byte-frozen parameter JSON, which is preserved verbatim. This keeps integrity hashes reproducible on Windows as well as other platforms.

The annual trajectory audit uses saved controls from the full ensemble, checks every annual trajectory with the same synthetic voltage screen, and applies the exact quadratic losses without reoptimization. Its analytic inverse is tested against the scalar replay. The weekly timing entry point reuses certified 300 s economic baselines and aligned trajectories, fixes SOC at 0.5 and permutes the seven carbon day profiles by offsets 1 and 3. A missing certified baseline remains an explicit status in the eight-pair denominator. Solver settings and counts are matched, not wall time or search-node counts. No GPU is needed. These outputs are generated locally and are not distributed inside this code archive.
