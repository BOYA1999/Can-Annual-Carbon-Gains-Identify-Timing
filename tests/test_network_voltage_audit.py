import pandas as pd

from scripts.run_network_voltage_audit import critical_multiplier, node_voltages_pu


def test_full_topology_voltage_audit_uses_all_seven_nodes():
    frame = pd.DataFrame({
        "grid_import_kw": [100.0],
        "grid_export_bus_kw": [0.0],
        "pv_used_kw": [40.0],
        "battery_discharge_kw": [20.0],
        "battery_charge_bus_kw": [0.0],
        "main_flow_kw": [150.0],
        "hvac_edge_kw": [80.0],
        "fixed_edge_kw": [60.0],
    })
    parameters = {"network": {"branch_resistance_ohm": {edge: 0.001 for edge in ("n0-n3", "n1-n3", "n2-n3", "n3-n4", "n4-n5", "n4-n6")}}}
    voltages = node_voltages_pu(frame, parameters)
    assert list(voltages) == ["n0", "n1", "n2", "n3", "n4", "n5", "n6"]
    assert voltages.loc[0, "n6"] < voltages.loc[0, "n5"] < 1.0
    assert critical_multiplier(voltages) > 1.0
