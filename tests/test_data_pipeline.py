from pathlib import Path

from src.data_pipeline import audit_inputs, load_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_g0_data_gate():
    report = audit_inputs(ROOT)
    assert report["passed"], report["checks"]


def test_units_and_capacity_contract():
    data = load_inputs(ROOT)
    assert len(data.frame) == 8760
    assert data.frame["carbon_kg_per_kwh"].max() < 1.0
    assert data.frame["cost_usd_per_kwh"].max() < 3.0
    assert abs(data.metadata["pv_dc_energy_fraction"] - 0.60) < 1e-12
    assert data.metadata["bess_energy_kwh"] > 0
    assert not data.metadata["load_grid_same_source_year"]
    assert data.metadata["load_grid_calendar_positions_equal"]
    assert "tmy-2020" in data.metadata["pv_weather_data_source"].lower()
