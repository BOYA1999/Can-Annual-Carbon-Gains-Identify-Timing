from dataclasses import dataclass
from math import exp

import numpy as np
import pandas as pd


@dataclass
class HvacEnvelope:
    baseline_kw: np.ndarray
    fixed_kw: np.ndarray
    lower_kw: np.ndarray
    upper_kw: np.ndarray
    state_a: float
    state_b_c_per_kwh: float
    comfort_delta_c: float


def _center_daily(values: np.ndarray) -> np.ndarray:
    return (values.reshape(-1, 24) - values.reshape(-1, 24).mean(axis=1, keepdims=True)).reshape(-1)


def _states(adjustment: np.ndarray, state_a: float) -> np.ndarray:
    result = np.zeros_like(adjustment, dtype=float)
    for day in range(len(adjustment) // 24):
        start = day * 24
        for hour in range(23):
            i = start + hour
            result[i + 1] = state_a * result[i] - adjustment[i]
    return result


def build_hvac_envelope(frame: pd.DataFrame, tau_hours: float = 8.0, comfort_delta_c: float = 2.0) -> HvacEnvelope:
    baseline = frame["load_baseline_kw"].to_numpy(float)
    dr = _center_daily(frame["load_hvac_dr_kw"].to_numpy(float) - baseline)
    precool = _center_daily(frame["load_precool_kw"].to_numpy(float) - baseline)
    lower = np.minimum.reduce([np.zeros(len(frame)), dr, precool])
    upper = np.maximum.reduce([np.zeros(len(frame)), dr, precool])
    state_a = exp(-1.0 / tau_hours)
    raw_max = max(np.abs(_states(dr, state_a)).max(), np.abs(_states(precool, state_a)).max())
    state_b = comfort_delta_c / raw_max
    fixed_level = 0.95 * np.min(baseline + lower)
    fixed = np.full(len(frame), fixed_level)
    hvac_baseline = baseline - fixed
    if np.min(hvac_baseline + lower) <= 0:
        raise ValueError("HVAC envelope creates a non-positive branch load")
    return HvacEnvelope(
        baseline_kw=hvac_baseline,
        fixed_kw=fixed,
        lower_kw=lower,
        upper_kw=upper,
        state_a=state_a,
        state_b_c_per_kwh=state_b,
        comfort_delta_c=comfort_delta_c,
    )


def converter_tangents(rating_kw: float, efficiency_shift: float = 0.0) -> list[tuple[float, float]]:
    if efficiency_shift == 0:
        fixed, linear, quadratic = 0.0033333333333333335, 0.01, 0.016666666666666666
    else:
        fractions = np.array([0.1, 0.5, 1.0])
        efficiency = np.clip(np.array([0.955, 0.975, 0.97]) + efficiency_shift, 1e-6, 1 - 1e-6)
        quadratic, linear, fixed = np.polyfit(fractions, fractions * (1 - efficiency), 2)
    lines = []
    for fraction in (0.1, 0.5, 1.0):
        slope = linear + 2.0 * quadratic * fraction
        intercept = rating_kw * (fixed - quadratic * fraction * fraction)
        lines.append((slope, intercept))
    return lines


def line_loss_tangents(rating_kw: float, resistance_ohm: float = 0.001, voltage_v: float = 380.0) -> list[tuple[float, float]]:
    coefficient = 1000.0 * resistance_ohm / voltage_v**2
    lines = []
    for fraction in (0.1, 0.4, 0.7, 1.0):
        point = fraction * rating_kw
        lines.append((2.0 * coefficient * point, -coefficient * point**2))
    return lines
