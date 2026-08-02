from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass
class InputData:
    frame: pd.DataFrame
    metadata: dict


def _read_8760(path: Path, value_column: str, time_column: str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if len(data) != 8760 or data[value_column].isna().any():
        raise ValueError(f"invalid 8760 series: {path}")
    time = pd.to_datetime(data[time_column])
    if time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError(f"invalid timestamps: {path}")
    return data, time


def load_inputs(root: Path) -> InputData:
    raw = root / "data" / "raw"
    source = root / "data" / "extracted" / "california" / "California"
    baseline, timestamp = _read_8760(source / "annual_load_pattern_CAMX_baseline.csv", "load_data", "datetime")
    hvac, hvac_time = _read_8760(source / "annual_load_pattern_CAMX_hvac_dr_ee.csv", "load_data", "datetime")
    precool, precool_time = _read_8760(source / "annual_load_pattern_CAMX_hvac_dr_ee_precool.csv", "load_data", "datetime")
    carbon, carbon_time = _read_8760(source / "cambium_grid_data_California_cambium_co2_rate_lrmer.csv", "value", "timestamp")
    cost, cost_time = _read_8760(source / "cambium_grid_data_California_cambium_grid_value.csv", "value", "timestamp")
    if not timestamp.equals(hvac_time) or not timestamp.equals(precool_time):
        raise ValueError("load scenarios are not time aligned")
    if not np.all(np.diff(timestamp.values).astype("timedelta64[h]") == np.timedelta64(1, "h")):
        raise ValueError("load timeline is not hourly")
    with (raw / "pvwatts_pasadena_1kw.json").open(encoding="utf-8") as handle:
        pv = json.load(handle)
    if pv.get("errors"):
        raise ValueError(f"PVWatts errors: {pv['errors']}")
    outputs = pv["outputs"]
    required = ("dc", "ac", "tamb")
    if any(len(outputs[key]) != 8760 for key in required):
        raise ValueError("PVWatts output is not 8760 hourly values")

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "load_baseline_kw": baseline["load_data"].astype(float),
            "load_hvac_dr_kw": hvac["load_data"].astype(float),
            "load_precool_kw": precool["load_data"].astype(float),
            "carbon_kg_per_kwh": carbon["value"].astype(float) / 1000.0,
            "cost_usd_per_kwh": cost["value"].astype(float) / 1000.0,
            "pv_dc_per_kw": np.asarray(outputs["dc"], dtype=float) / 1000.0,
            "pv_ac_per_kw": np.asarray(outputs["ac"], dtype=float) / 1000.0,
            "ambient_c": np.asarray(outputs["tamb"], dtype=float),
        }
    )
    pv_capacity_kw = 0.60 * frame["load_baseline_kw"].sum() / frame["pv_dc_per_kw"].sum()
    frame["pv_dc_kw"] = frame["pv_dc_per_kw"] * pv_capacity_kw
    peak_net_kw = float((frame["load_baseline_kw"] - 0.97 * frame["pv_dc_kw"]).clip(lower=0).max())
    metadata = {
        "hours": 8760,
        "dataset205_url": "https://data.nlr.gov/submissions/205",
        "california_archive_url": "https://data.nlr.gov/system/files/205/California-1679428758.zip",
        "dataset205_license_url": "https://data.nlr.gov/node/205/license",
        "pvwatts_request_url": "https://developer.nlr.gov/api/pvwatts/v8.json",
        "load_year": int(timestamp.dt.year.iloc[0]),
        "grid_source_year": int(carbon_time.dt.year.iloc[0]),
        "grid_alignment": "positional standard-year alignment documented by Dataset 205 methodology",
        "carbon_source": "15-year normalized long-run marginal CO2 rate",
        "carbon_raw_unit": "kg_CO2_per_MWh",
        "cost_source": "combined short-run marginal grid value",
        "cost_raw_unit": "USD_per_MWh",
        "pvwatts_version": pv["version"],
        "pvwatts_request": pv["inputs"],
        "pv_station": pv["station_info"],
        "pv_capacity_kw": pv_capacity_kw,
        "pv_dc_energy_fraction": float(frame["pv_dc_kw"].sum() / frame["load_baseline_kw"].sum()),
        "bess_energy_kwh": 2.0 * peak_net_kw,
        "bess_power_kw": peak_net_kw,
    }
    return InputData(frame=frame, metadata=metadata)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_inputs(root: Path) -> dict:
    inputs = load_inputs(root)
    frame = inputs.frame
    raw = root / "data" / "raw"
    files = sorted(path for path in raw.iterdir() if path.is_file())
    checks = {
        "hours_equal_8760": len(frame) == 8760,
        "no_missing_values": not frame.isna().any().any(),
        "timestamps_unique": not frame["timestamp"].duplicated().any(),
        "pvwatts_has_no_errors": True,
        "pv_energy_ratio_is_0_60": abs(inputs.metadata["pv_dc_energy_fraction"] - 0.60) < 1e-12,
        "carbon_nonnegative": bool((frame["carbon_kg_per_kwh"] >= 0).all()),
        "cost_nonnegative": bool((frame["cost_usd_per_kwh"] >= 0).all()),
        "license_captured": (raw / "dataset205_license.html").exists(),
    }
    return {
        "gate": "G0",
        "passed": all(checks.values()),
        "checks": checks,
        "metadata": inputs.metadata,
        "ranges": {
            column: {"min": float(frame[column].min()), "max": float(frame[column].max()), "mean": float(frame[column].mean())}
            for column in frame.columns if column != "timestamp"
        },
        "raw_files": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in files
        ],
    }
