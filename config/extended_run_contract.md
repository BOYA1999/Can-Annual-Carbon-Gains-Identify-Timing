# Extended numerical run contract

Date frozen: 2026-08-02

Canonical system-parameter SHA256: `6973b110a5c41ddab971ce73c31a239474e3852f8472ccb6b4b9364e58bcbd9a`

## Loss approximation audit

- Evaluate exact converter and line quadratics on 10,001 equally spaced loading fractions.
- Compare 3, 5, and 9 fixed tangent sets.
- Replay fixed F0 and F4 controls with exact losses.
- Record absolute and relative approximation errors, closure, cost, emissions, and zero-power behavior.

## Matched signal ensemble

- Use 8,760 hours and 365 daily solves per surrogate.
- Hold resources, cost caps, input dimensions, solver settings, and realized replay fixed.
- Use circular shifts of 1, 3, 7, 14, and 28 complete days.
- Use seeds 20260802 through 20260821 for permutations within month, weekday, and joint month-by-weekday strata.
- Preserve the exact sorted annual carbon values and retain every failed surrogate.
- Compute circular block-resampling intervals with 1, 7, 14, and 28 day blocks.
- The fixed support gate requires all 65 surrogates, a positive median annual emissions penalty, and more than 80% positive penalties.

## Forecast diagnostics

- Compare previous-day, previous-week, leave-one-week-out hour-of-week climatology, and annual-mean carbon predictors.
- Report annual and seasonal MAE, RMSE, RMSE divided by the realized mean, Pearson correlation, and daily-correlation quantiles.
- Evaluate levels 0.125, 0.15, 0.175, and 0.20 with seeds 20260802, 20260803, and 20260804 over the four fixed seasonal weeks.
- Perturb carbon, PV, HVAC baseline, fixed load, and all four channels separately.
- Retain raw solver status, input ranges, clipping counts, bound checks, implicated capacity, and minimum fixed-converter rating relaxation for every failed day.

## State horizon and throughput price

- Evaluate one 168-hour optimization for each fixed seasonal week.
- Use initial and terminal battery fractions 0.40, 0.50, and 0.60, zero terminal temperature deviation, and zero weekly HVAC adjustment sum.
- Compare F0 with emissions minimization at 4% above the corresponding economic minimum.
- Evaluate half-cycle throughput prices of 0, 0.01, 0.03, and 0.05 USD/kWh in both the economic reference and cost cap.
- Record cost, emissions, throughput, equivalent full cycles, physical closure, every solver failure, and all incomplete denominators.

## Fixed stop rules

Solver failures, sign changes, approximation errors, and null surrogate distributions remain results. Seeds, levels, thresholds, block lengths, signal definitions, time limits, and solver settings cannot be changed after inspecting outputs.
