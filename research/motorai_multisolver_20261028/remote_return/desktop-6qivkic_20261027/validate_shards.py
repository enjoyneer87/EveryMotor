"""Re-read published NPZs and independently integrate their saved phase fields.

Three-point Gaussian quadrature is independent of the closed-form P1 mass rule
in field_reader.py. This validates compression/indexing, not solver convergence.
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
from field_reader import read_json,sha,write_json
from load_shard import load_copper_graph


def enrich(out,config):
    """Add verified circuit voltages and explicit metadata to this run's outputs."""
    w=Path(config["workspace"]);local=Path(config["local_root"])
    paths={}
    for f in sorted((out/"shards").glob("*_audit.json")):
        audit=read_json(f);pid=audit["point_id"];cid=audit["case_id"]
        roots=[w/"cases"/cid,w/"outputs/jeet_fields_16000rpm_20261018/work/restored_qa"/cid,
               w/"outputs/jeet_fields_representative_20261018/work/restored_qa"/cid,local/"restored"/cid]
        raw=next(d/"raw" for d in roots if (d/"raw/mesh/native.result").exists())
        for rel in ["INPUT_CONTRACT.json","circuits.dat","circuits.dat.names"]:
            expected=next(i["sha256"] for i in audit["inputs"] if i["logical_source_key"].endswith("/raw/"+rel))
            assert sha(raw/rel)==expected
        contract=read_json(raw/"INPUT_CONTRACT.json")
        components=sorted([r for r in contract["components"] if r["kind"]=="copper"],key=lambda c:c["physical"])
        names={m[2].strip():int(m[1])-1 for m in re.finditer(r"^\s*(\d+):\s*(.*)$",(raw/"circuits.dat.names").read_text(),re.M)}
        circuits=np.loadtxt(raw/"circuits.dat").reshape(-1,len(names))
        p=out/"shards"/(pid+"_full_history_integrals.npz")
        with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        volt=circuits[:,[names[f"res: v_component({c['component']})"] for c in components]]
        assert np.max(np.abs(d["time_s"]-circuits[:,names["res: time"]]))==0
        if "component_voltage_V" not in d:
            np.savez_compressed(p,**d,component_voltage_V=volt)
        else:assert np.array_equal(d["component_voltage_V"],volt)
        for p in [out/"shards"/(pid+"_field_snapshots.npz"),local/"full_history"/(pid+"_copper_A_all512.npz")]:
            with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
            before={k:v.copy() for k,v in d.items()}
            if "node_phase_deg" in d:d["component_phase_deg"]=d.pop("node_phase_deg")
            if "body_ids" not in d:
                d["body_ids"]=np.array([contract["body_id_by_physical"][str(int(p))] for p in d["physical"]])
                d["material_id"]=np.ones(len(d["physical"]),dtype=np.int32);d["material_names"]=np.array(["copper"])
                np.savez_compressed(p,**d)
            with np.load(p,allow_pickle=False) as z:
                for k,v in before.items():assert np.array_equal(z["component_phase_deg" if k=="node_phase_deg" else k],v)
        for item in audit["inputs"]:
            paths[item["logical_source_key"]]=str(raw/item["logical_source_key"].split("/raw/",1)[1])
        paths["local/full_history/"+pid]=str(local/"full_history"/(pid+"_copper_A_all512.npz"))
    pm=read_json(local/"local_path_map.json") if (local/"local_path_map.json").exists() else {}
    pm.setdefault("logical_source_paths",{}).update(paths);write_json(local/"local_path_map.json",pm)


def validate(out):
    rows=[]
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    for p in sorted((out/"shards").glob("*_field_snapshots.npz")):
        d=load_copper_graph(p);pid=p.name.replace("_field_snapshots.npz","")
        with np.load(out/"shards"/(pid+"_full_history_integrals.npz"),allow_pickle=False) as z:wave={k:z[k] for k in z.files}
        area=d["element_area_m2"];bar=d["physical"]-2301;scale=float(d["symmetry_sectors"]*d["active_length_m"]/d["sigma_S_m"])
        tri=d["connectivity_local"];xy=d["node_xy_m"][tri];e=xy[:,1:]-xy[:,:1]
        residuals=[];currenterrs=[];berrs=[]
        for k,step in enumerate(d["steps"]):
            j=d["J_total_element_node_Am2"][k]
            # Independent quadrature at barycentric degree-two integration points.
            jg=j@bary.T;joule=scale*np.sum(area*(jg*jg).mean(axis=1))
            cur=np.bincount(bar,weights=area*j.mean(axis=1),minlength=36)
            imposed=wave["current_prescribed_A"][step-1]
            dc=scale*np.sum(imposed**2/d["bar_area_m2"])
            residuals.append(np.array([joule,dc,joule-dc])-wave["scalar"][step-1,:3])
            currenterrs.append(float(np.max(np.abs(cur-wave["current_integrated_A"][step-1]))))
            av=d["A_node_Wb_per_m"][k][tri]
            grad=np.linalg.solve(e,(av[:,1:]-av[:,:1])[...,None])[...,0]
            b=np.column_stack([grad[:,1],-grad[:,0]])
            berrs.append(float(np.max(np.abs(b-d["B_element_T"][k]))))
        maxerr=float(np.max(np.abs(residuals)));be=max(berrs);ce=max(currenterrs)
        assert maxerr<=1e-6 and ce<1e-8 and be<1e-10
        # Jensen's inequality: E[B^2] >= E[B]^2, each component and conductor.
        gap=wave["B_area_second_moment_T2"][:,:,:2]-wave["B_area_mean_T"]**2
        assert gap.min()>-1e-12
        rows.append(dict(point_id=pid,status="PASS",phase_count=len(d["steps"]),FE_node_count=len(d["node_ids"]),
            FE_element_count=len(tri),directed_FE_edges=d["edge_index"].shape[1],max_quadrature_loss_delta_W=maxerr,
            max_current_delta_A=ce,max_A_curl_B_delta_T=be,min_B_variance_T2=float(gap.min()),
            compression="float64 NPZ lossless; roundtrip and independent quadrature verified",
            temporal_snapshot_note="8 phases do not preserve P2 loss to a specified tolerance; full 256-sample P2 integrals retained"))
    write_json(out/"shard_validation.json",dict(status="PASS",cases=len(rows),scope="NPZ/ID/FE connectivity/phase quadrature validation only",rows=rows))
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--path-map",type=Path);a=p.parse_args();out=Path(__file__).resolve().parent
    if a.path_map:enrich(out,read_json(a.path_map))
    validate(out)
