from pathlib import Path

import numpy as np

from src.data_pipeline import load_inputs
from src.model import build_hvac_envelope, converter_loss_exact, converter_tangents, line_loss_exact


ROOT = Path(__file__).resolve().parents[1]


def test_hvac_envelope_is_bounded_and_energy_neutral_by_source_day():
    data = load_inputs(ROOT)
    envelope = build_hvac_envelope(data.frame)
    assert np.all(envelope.lower_kw <= 0)
    assert np.all(envelope.upper_kw >= 0)
    assert np.min(envelope.baseline_kw + envelope.lower_kw) > 0
    for source in (data.frame["load_hvac_dr_kw"], data.frame["load_precool_kw"]):
        centered = (source.to_numpy() - data.frame["load_baseline_kw"].to_numpy()).reshape(-1, 24)
        centered -= centered.mean(axis=1, keepdims=True)
        assert np.max(np.abs(centered.sum(axis=1))) < 1e-10


def test_converter_curve_matches_frozen_points():
    rating = 100.0
    lines = converter_tangents(rating)
    for fraction, efficiency in ((0.1, 0.955), (0.5, 0.975), (1.0, 0.970)):
        power = fraction * rating
        loss = max(slope * power + intercept for slope, intercept in lines)
        assert abs((power - loss) / power - efficiency) < 1e-10


def test_converter_efficiency_shift_matches_varied_points():
    rating = 100.0
    for shift in (-0.02, 0.02):
        lines = converter_tangents(rating, shift)
        for fraction, efficiency in ((0.1, 0.955), (0.5, 0.975), (1.0, 0.970)):
            power = fraction * rating
            loss = max(slope * power + intercept for slope, intercept in lines)
            assert abs((power - loss) / power - (efficiency + shift)) < 1e-10


def test_tangent_loss_is_a_lower_bound_and_switchable_zero_loss_is_zero():
    rating = 100.0
    lines = converter_tangents(rating)
    for power in np.linspace(0.1 * rating, rating, 101):
        tangent = max(slope * power + intercept for slope, intercept in lines)
        assert tangent <= converter_loss_exact(power, rating) + 1e-12
    assert converter_loss_exact(0.0, rating) == 0.0
    assert line_loss_exact(0.0) == 0.0
