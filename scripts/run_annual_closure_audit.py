import argparse
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_signal_ensemble import _block_interval
from scripts.run_network_voltage_audit import node_voltages_pu, critical_multiplier
from scripts.run_main import _controls
from src.dispatch import load_parameters, _input_for_output, replay_day
from src.evaluate import evaluate_dispatch
from src.model import converter_loss_coefficients

def exact_replay(source, parameters):
    r = parameters["network"]["ratings_kw"]
    f, b, q = converter_loss_coefficients(parameters["network"].get("converter_efficiency_shift_fraction", 0))
    def loss(x, rating, line=False):
        x = np.asarray(x, float)
        return np.where(x > 0, 1000 * .001 * x**2 / 380**2 if line else rating*f + b*x + q*x**2/rating, 0)
    def inverse(y, rating, line=False):
        y = np.asarray(y, float)
        c, a, fixed = (1000*.001/380**2, 1., 0.) if line else (q/rating, 1-b, rating*f)
        if np.max(y - (rating - loss(rating, rating, line))) > 1e-6:
            raise ValueError("required converter output exceeds rating")
        value = 2*(y+fixed) / (a + np.sqrt(a*a - 4*c*(y+fixed)))
        return np.where(y > 0, value, 0.)
    out = source.copy()
    controls = _controls(source)
    stored = controls["battery_charge_stored_kw"].copy()
    maximum = r["bess"] - loss(r["bess"], r["bess"])
    if np.max(stored - maximum) > 1e-6:
        raise ValueError("planned stored charge exceeds physical tolerance")
    stored = np.minimum(stored, maximum)
    powers = {}
    powers["hvac_edge_kw"] = inverse(source.hvac_load_kw, r["hvac_line"], True)
    powers["fixed_edge_kw"] = inverse(source.fixed_load_kw, r["fixed_converter"])
    powers["main_flow_kw"] = inverse(powers["hvac_edge_kw"] + powers["fixed_edge_kw"], r["main"])
    powers["battery_charge_bus_kw"] = inverse(stored, r["bess"])
    powers["battery_discharge_kw"] = controls["battery_discharge_kw"]
    pv = np.minimum(source.pv_dc_kw.to_numpy(), r["pv"])
    pv_delivery = np.maximum(0, pv - loss(pv, r["pv"]))
    demand = powers["main_flow_kw"] + powers["battery_charge_bus_kw"] - (powers["battery_discharge_kw"] - loss(powers["battery_discharge_kw"], r["bess"]))
    powers["pv_used_kw"] = np.where((pv_delivery <= demand) & (pv_delivery <= 0), 0, pv)
    powers["grid_import_kw"] = inverse(np.maximum(demand - pv_delivery, 0), r["pcc"])
    powers["grid_export_bus_kw"] = np.maximum(pv_delivery - demand, 0)
    names = (("hvac_edge_kw","hvac_line_loss_kw","hvac_line",True), ("fixed_edge_kw","fixed_loss_kw","fixed_converter",False),
             ("main_flow_kw","main_loss_kw","main",False), ("battery_charge_bus_kw","battery_charge_loss_kw","bess",False),
             ("battery_discharge_kw","battery_discharge_loss_kw","bess",False), ("pv_used_kw","pv_loss_kw","pv",False),
             ("grid_import_kw","grid_import_loss_kw","pcc",False), ("grid_export_bus_kw","grid_export_loss_kw","pcc",False))
    for p, l, rating, line in names:
        out[p] = powers[p]
        out[l] = np.minimum(powers[p], loss(powers[p], r[rating], line)) if l == "grid_export_loss_kw" else loss(powers[p], r[rating], line)
    out["grid_export_delivered_kw"] = out.grid_export_bus_kw - out.grid_export_loss_kw
    return out

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, default=root)
    parser.add_argument("--ensemble", type=Path, default=root/"artifacts/experiment/signal_ensemble_2026-08-02")
    parser.add_argument("--scalar-exact", type=Path, default=root/"artifacts/experiment/loss_audit_2026-08-02/annual_F4_exact_replay.csv")
    parser.add_argument("--output", type=Path, default=root/"artifacts/annual_closure_audit")
    args = parser.parse_args()
    old = args.historical
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters(root)
    ensemble = args.ensemble
    table = pd.read_csv(ensemble / "ensemble_results.csv").sort_values("emissions_penalty_fraction")
    median_id = table.iloc[32].id
    pareto = old / "artifacts/experiment/pareto_2026-08-02"
    reference = pd.read_csv(pareto / "F4_dispatch.csv")
    original_exact = pd.read_csv(args.scalar_exact)
    new_exact = exact_replay(reference, parameters)
    compare_cols = ["grid_import_kw","grid_export_delivered_kw","main_flow_kw","battery_charge_bus_kw","hvac_edge_kw"]
    difference = float(np.max(np.abs(new_exact[compare_cols].to_numpy()-original_exact[compare_cols].to_numpy())))
    assert difference < 1e-8, difference
    dynamic = pd.read_csv(old / "artifacts/experiment/adversarial_signals_2026-08-02/daily_metrics.csv")
    dynamic = dynamic[dynamic.method=="DYNAMIC"].sort_values("day")
    reference_daily = ((reference.grid_import_kw-reference.grid_export_delivered_kw)*reference.carbon_kg_per_kwh).to_numpy().reshape(365,24).sum(axis=1)
    assert np.allclose(reference_daily, dynamic.emissions_kgco2e, rtol=0, atol=1e-8)
    rows, trace, hashes = [], [], []
    sources = [("F0",pareto/"F0_dispatch.csv"), ("F4",pareto/"F4_dispatch.csv")]
    sources += [(sid,ensemble/sid/"checkpoint.pkl") for sid in table.id]
    for sid, path in sources:
        hashes.append({"id":sid,"file":path.relative_to(old).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        if path.suffix==".pkl":
            saved=pickle.loads(path.read_bytes())
            assert saved["next_day"]==365 and len(saved["solver"])==365 and all(r["status"]==0 for r in saved["solver"])
            frame=pd.concat(saved["parts"],ignore_index=True)
        else:
            frame=pd.read_csv(path)
        assert len(frame)==8760
        if sid not in ("F0","F4"):
            expected = json.loads((ensemble/sid/"summary.json").read_text())["metrics"]["annual_emissions_kgco2e"]
            assert abs(evaluate_dispatch(frame)["annual_emissions_kgco2e"]-expected)<1e-7
        exact=exact_replay(frame,parameters)
        volts=node_voltages_pu(frame,parameters)
        violation=(volts<.95)|(volts>1.05)
        where=np.argwhere(violation.to_numpy())
        tm, em = evaluate_dispatch(frame), evaluate_dispatch(exact)
        row={"id":sid,"hours":len(frame),"minimum_voltage_pu":float(volts.min().min()),
             "maximum_voltage_pu":float(volts.max().max()),"violating_node_hours":int(violation.to_numpy().sum()),
             "first_violation_node":str(volts.columns[where[0,1]]) if len(where) else "",
             "first_violation_timestamp":str(frame.timestamp.iloc[where[0,0]]) if len(where) else "",
             "critical_common_resistance_multiplier":critical_multiplier(volts),
             **{"tangent_"+k:v for k,v in tm.items()}, **{"exact_"+k:v for k,v in em.items()}}
        ratios = [exact[col].max()/parameters["network"]["ratings_kw"][rating] for col,rating in
                  (("grid_import_kw","pcc"),("grid_export_bus_kw","pcc"),("pv_used_kw","pv"),
                   ("battery_discharge_kw","bess"),("battery_charge_bus_kw","bess"),("main_flow_kw","main"),
                   ("hvac_edge_kw","hvac_line"),("fixed_edge_kw","fixed_converter"))]
        row["exact_maximum_input_rating_ratio"] = float(max(ratios))
        row["exact_rating_violation"] = bool(max(ratios)>1+1e-8)
        rows.append(row)
        controls=_controls(frame)
        compact=pd.DataFrame({"hour":np.arange(8760),"timestamp":frame.timestamp,
                              **controls, **{node:volts[node] for node in volts},
                              **{name:frame[name] for name in ("grid_import_kw","grid_export_bus_kw","pv_used_kw","battery_charge_bus_kw","main_flow_kw","hvac_edge_kw","fixed_edge_kw")},
                              "tangent_net_grid_kw":frame.grid_import_kw-frame.grid_export_delivered_kw,
                              "exact_net_grid_kw":exact.grid_import_kw-exact.grid_export_delivered_kw})
        compact.to_csv(out/f"controls_voltage_{sid}.csv.gz",index=False,compression="gzip")
        if sid==median_id:
            daily=((frame.grid_import_kw-frame.grid_export_delivered_kw)*frame.carbon_kg_per_kwh).to_numpy().reshape(365,24).sum(axis=1)
            vector=daily-reference_daily
            pd.DataFrame({"day":np.arange(365),"aligned_kg":reference_daily,"surrogate_kg":daily,"difference_kg":vector}).to_csv(out/"median_paired_daily.csv",index=False)
            trace={"run_id":sid,"rank":33,"reference_emissions_kg":float(reference_daily.sum()),
                   "surrogate_emissions_kg":float(daily.sum()),"annual_difference_kg":float(vector.sum()),
                   "mean_daily_difference_kg":float(vector.mean()),"penalty_fraction":float(vector.sum()/reference_daily.sum()),
                   "intervals":{str(block):_block_interval(vector,block) for block in (1,7,14,28)}}
        print(sid,"voltage",row["violating_node_hours"],"exact complete",flush=True)
    metrics=pd.DataFrame(rows)
    f4=metrics.set_index("id").loc["F4"]
    metrics["tangent_penalty_fraction"]=metrics.tangent_annual_emissions_kgco2e/f4.tangent_annual_emissions_kgco2e-1
    metrics["exact_penalty_fraction"]=metrics.exact_annual_emissions_kgco2e/f4.exact_annual_emissions_kgco2e-1
    metrics.to_csv(out/"annual_audit_metrics.csv",index=False)
    surrogate=metrics[~metrics.id.isin(["F0","F4"])]
    summary={"trajectory_count":len(metrics),"schedule_hours":int(metrics.hours.sum()),"node_hours":int(7*metrics.hours.sum()),
             "minimum_voltage_pu":float(metrics.minimum_voltage_pu.min()),"maximum_voltage_pu":float(metrics.maximum_voltage_pu.max()),
             "violating_node_hours":int(metrics.violating_node_hours.sum()),
             "critical_common_resistance_multiplier":float(metrics.critical_common_resistance_multiplier.min()),
             "exact_completed_surrogates":len(surrogate),"exact_positive_surrogates":int((surrogate.exact_penalty_fraction>0).sum()),
             "exact_trajectories_with_rating_violation":int(metrics.exact_rating_violation.sum()),
             "exact_minimum_penalty_fraction":float(surrogate.exact_penalty_fraction.min()),
             "exact_median_penalty_fraction":float(surrogate.exact_penalty_fraction.median()),
             "worst_energy_closure_relative_error":float(metrics[["tangent_energy_closure_relative_error","exact_energy_closure_relative_error"]].max().max()),
             "worst_carbon_closure_relative_error":float(metrics[["tangent_carbon_closure_relative_error","exact_carbon_closure_relative_error"]].max().max()),
             "vectorized_vs_existing_scalar_exact_maximum_kw_difference":difference,"median_run":trace,
             "aligned_F4_vs_DYNAMIC_maximum_daily_emissions_difference_kg":float(np.max(np.abs(reference_daily-dynamic.emissions_kgco2e.to_numpy())))}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    (out/"source_hashes.json").write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
