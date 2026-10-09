from pathlib import Path
import numpy as np
import pytest
from scripts.run_main import _controls, _slice_envelope
from scripts.run_annual_closure_audit import exact_replay
from src.data_pipeline import load_inputs
from src.dispatch import load_parameters, replay_day
from src.model import build_hvac_envelope

@pytest.mark.parametrize("day", [0, 190])
def test_vector_exact_matches_scalar_replay(day):
    root = Path(__file__).resolve().parents[1]
    inputs = load_inputs(root)
    parameters = load_parameters(root)
    indices = np.arange(day*24, (day+1)*24)
    frame = inputs.frame.iloc[indices].reset_index(drop=True)
    envelope = _slice_envelope(build_hvac_envelope(inputs.frame), indices)
    controls = {"hvac_adjustment_kw": np.zeros(24),
                "battery_charge_stored_kw": np.r_[np.full(12,10.),np.zeros(12)],
                "battery_discharge_kw": np.r_[np.zeros(12),np.full(12,10.)]}
    tangent = replay_day(frame, envelope, parameters, controls)
    scalar = replay_day(frame, envelope, parameters, _controls(tangent), loss_model="exact")
    vector = exact_replay(tangent, parameters)
    numeric = scalar.select_dtypes("number").columns
    assert np.max(np.abs(scalar[numeric].to_numpy()-vector[numeric].to_numpy())) < 1e-8

def test_week_local_mapping_keeps_profiles_and_all_values():
    profiles = np.arange(168).reshape(7,24)
    for offset in (1,3):
        altered = profiles[(np.arange(7)-offset)%7].ravel()
        assert np.array_equal(np.sort(altered), profiles.ravel())
        assert np.all(np.diff(altered.reshape(7,24), axis=1)==1)
