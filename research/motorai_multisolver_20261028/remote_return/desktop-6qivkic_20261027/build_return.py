"""Assemble a source-aware return package from existing data. No fitting/solves.

Absolute paths are consumed only from a local path map and written only beside
that map. All published associations use logical source keys and hashes.
"""
from pathlib import Path
import argparse
import collections
import csv
import datetime
import hashlib
import json
import subprocess
import numpy as np
from field_reader import read_json, write_json, sha

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PIN = "0034d7c24b1cc0cf41499d028733792b14815bc7"
HANDOFF = "3b14d149cbcba7c0aff5bf77f90f7c2043814d58"
RESERVED = {"R1_0199", "R1_0076"}


def csv_write(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader();w.writerows(rows)


def stats(values):
    a=np.array(values,dtype=float)
    return dict(count=len(a), max_abs=float(np.abs(a).max()) if len(a) else None,
                mean_abs=float(np.abs(a).mean()) if len(a) else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--path-map",type=Path,required=True);ap.add_argument("--allow-base-rebuild",action="store_true");args=ap.parse_args()
    if (Path(__file__).resolve().parent/'iso_recovery/MS_validation.json').exists() and not args.allow_base_rebuild:
        ap.error('This builds the pre-iso base snapshot. Use --allow-base-rebuild only when followed by iso_recovery/merge_recovery.py; otherwise recovered status would regress.')
    config=read_json(args.path_map);w=Path(config["workspace"]);repo=Path(config["source_repo"]);nas=Path(config["nas_root"])
    sources=[]; pathmap={}; bykey={}
    def source(key,p=None,expected=None,expected_bytes=None,commit=None,kind="source",hash_now=True):
        if key in bykey:return bykey[key]
        p=Path(p) if p is not None else None;exists=p is not None and p.is_file()
        actual=sha(p) if exists and hash_now else None
        size=p.stat().st_size if exists else None
        status="PASS" if actual is not None and expected is not None and actual==expected and (expected_bytes is None or size==expected_bytes) else "FAIL" if actual is not None and expected is not None else "UNKNOWN"
        reason="hash_verified_this_run" if status=="PASS" else "hash_mismatch_preserved" if status=="FAIL" else "missing_here" if not exists else "existence_size_only" if not hash_now else "no_independent_expected_hash"
        r=dict(logical_source_key=key,kind=kind,source_commit=commit,expected_sha256=expected,expected_bytes=expected_bytes,
               original_sha256=actual,original_bytes=size,present_here=exists,validation_status=status,validation_reason=reason,
               extraction="read_only; original unchanged")
        sources.append(r);bykey[key]=r
        if p is not None:pathmap[key]=str(p)
        return r
    def pinned(rel):
        blob=subprocess.check_output(["git","-C",str(repo),"show",f"{PIN}:{rel}"])
        return source("git/"+rel,repo/rel,hashlib.sha256(blob).hexdigest(),len(blob),PIN,"pinned_source")
    manifest=read_json(ROOT/"baseline_audit/source_manifest.json")
    for r in manifest:
        source("baseline/"+r["git_path"],ROOT/"baseline_audit"/r["packaged_relative_path"],r["sha256"],r["bytes"],r["commit"],"frozen_baseline")
    fullsrc=pinned("data/elmer_revision_20261018/JEET_RESULTS_240.json")
    lowplansrc=pinned("handoff/20261019_lowcurrent_pairs/FULL_JOBS_48.json")
    source("contract/scientific_gates",w/"SCIENTIFIC_GATES.json",kind="historical_gate")
    latest=read_json(ROOT/"baseline_audit/sources/handoff/20261019_AF_loaded96/AF_POINTS_192.json")
    half=read_json(ROOT/"baseline_audit/sources/handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json")
    full=read_json(repo/"data/elmer_revision_20261018/JEET_RESULTS_240.json")["rows"]
    low=read_json(repo/"handoff/20261019_lowcurrent_pairs/FULL_JOBS_48.json")
    idx={r["source_key"]:r for r in read_json(ROOT/"baseline_audit/sources/handoff/20261019_rea_hybrid_remote_results/INPUT_INDEX_60.json")["rows"]}
    latest_by={r["case_id"]:r for r in latest};masks=read_json(ROOT/"elmer_adapter/masks.json")
    masks_by={r["case_id"]:r for r in masks}
    catalog=[json.loads(l)["index"] for l in (ROOT/"federated_catalog/output/catalog.jsonl").read_text(encoding="utf-8").splitlines()]
    group_by={r["original_geometry_id"]:r["global_component_id"] for r in catalog}
    (OUT/"waveforms").mkdir(exist_ok=True)
    raw_audits={r["case_id"]:r for f in sorted((OUT/"shards").glob("*_audit.json")) for r in [read_json(f)]}
    conditions=[];observations=[];wave_summary=[];wave_arrays={};canonical_delta=[];ms=[];candidates=[]
    def observation(condition,quantity,value,evaluator,sourcekey,sourcehash,kind,quality,average="P2 post-step arithmetic mean"):
        geometry="jeet_elmer:"+condition["model"]
        m=masks_by.get(condition["case_id"],{})
        observations.append(dict(observation_id=condition["case_id"]+":"+evaluator+":"+quantity,
            template="jeet_e10_ideal_radial_same_template",geometry=geometry,geometry_group=group_by.get(geometry,geometry),
            condition=condition["case_id"],source="Elmer" if kind=="solver_target" else "canonical_JEET_evaluator",
            evaluator=evaluator,quantity=quantity,scope="whole_motor_active_stack_2D_no_end_winding",unit="W",value=value,
            speed_rpm=condition["speed_rpm"],current_Arms=condition["current_Arms"],beta_deg=condition["beta_deg"],
            temperature_C="",temperature_evidence="80C historical contract; per-case direct temperature readback not available",
            sigma_S_m=47092391.429245 if condition["model"]!="HalfSC" else "",active_length_m=0.15 if condition["model"]!="HalfSC" else "",
            waveform="prescribed_sinusoidal_conductor_current; Irms phase RMS",average_window=average,quality=quality,
            source_key=sourcekey,source_hash=sourcehash,observation_role=kind,new_independent_truth=False,
            own_AF_calibration=m.get("own_AF_calibration",False),transferred_coordinate=m.get("transferred_probe_coordinate",False),
            HalfSC_predeclared32=m.get("HalfSC_predeclared32",False),HalfSC_exploratory4=m.get("HalfSC_exploratory4",False),
            reserved_all_source=False,final_test_eligible=False))
    for row in full:
        cid=row["case_id"];case=w/"cases"/cid;ap=case/"derived/CURRENT_LOSS_AUDIT.json"
        src=source(f"full/{cid}/field_derived_integral_history",ap,row["source"]["audit_sha256"],commit=PIN,kind="field_derived_history")
        archive=read_json(case/"ARCHIVE_SHA256.json")
        source(f"full/{cid}/archive_member_manifest",case/"ARCHIVE_SHA256.json",row["source"]["raw_proof_sha256"],commit=PIN,kind="archive_member_hash_reference")
        archive_by={r["path"]:r for r in archive["files"]}
        location=read_json(case/"RAW_LOCATION.json") if (case/"RAW_LOCATION.json").exists() else None
        archive_path=nas/"receipts"/(cid+".tar.gz")
        if location:
            source(f"full/{cid}/archive",archive_path,location["archive_sha256"],kind="raw_archive",hash_now=False)
        data=read_json(ap); rr=data["rows"];n=data["steps_per_period"]
        assert n==256 and len(rr)==512
        wave=np.array([[r["time_s"],r["total_active_W"],r["DC_transport_W"],r["AC_excess_W"],r["max_field_current_error_per_phasepeak"]*100] for r in rr])
        assert np.isfinite(wave).all();dt=row["effective"]["dt_s"]
        time_error=float(np.max(np.abs(np.diff(np.r_[0,wave[:,0]])/dt-1)))
        assert time_error<1e-9
        p2=wave[256:];means=p2[:,1:4].mean(axis=0)
        targets=np.array([row["P2_Cu_total_W"],row["P2_DC_transport_W"],row["P2_AC_excess_W"]])
        errors=means-targets;p2max=float(p2[:,4].max());wholemax=float(wave[:,4].max())
        band=latest_by.get(cid)
        if band:
            assert row["model"]==band["model"] and row["speed_rpm"]==band["speed_rpm"] and row["beta_deg"]==band["beta_deg"]
            assert abs(row["current_Arms"]-band["current_Arms"])<1e-9
            canonical_delta.append(dict(case_id=cid,delta_AC_W=float(means[2]-band["P2_AC_excess_W"]),
                                        delta_DC_W=float(means[1]-band["P2_DC_transport_W"])))
        sums=np.array([[sum(r[k]) for k in ["bar_total_W","bar_DC_W","bar_AC_W"]] for r in rr])
        sumerr=float(np.max(np.abs(sums-wave[:,1:4])))
        record=dict(case_id=cid,point_id=row["point_id"],model=row["model"],latest_loaded192=bool(band),
            source_hash_status=src["validation_status"],steps=512,periods=2,steps_per_period=256,
            P2_start_step=257,P2_end_step=512,P2_start_s=float(p2[0,0]),P2_end_s=float(p2[-1,0]),
            average="uniform post-step arithmetic mean, not trapz",P2_Cu_W=float(means[0]),P2_DC_W=float(means[1]),P2_AC_W=float(means[2]),
            max_scalar_difference_W=float(np.abs(errors).max()),AC_relative_difference_percent=float(100*errors[2]/targets[2]),
            sum_bar_max_difference_W=sumerr,loss_arithmetic_status="PASS" if np.abs(errors).max()<=1e-6 else "FAIL",
            P2_current_max_percent=p2max,P2_current_status="PASS" if p2max<=0.001 else "FAIL",
            whole_current_max_percent=wholemax,whole_current_status="PASS" if wholemax<=0.001 else "FAIL",
            historical_P2_current_max_percent=row["P2_max_current_error_percent"],
            period_convergence_status="UNKNOWN",mesh_convergence_status="UNKNOWN",
            native_field_reintegrated_here=cid in raw_audits,solver_version=raw_audits.get(cid,{}).get("solver_version","UNKNOWN"),
            max_dt_relative_difference=time_error)
        wave_summary.append(record)
        wave_arrays[row["point_id"]]=wave
        base=dict(case_id=cid,model=row["model"],point_id=row["point_id"],speed_rpm=row["speed_rpm"],current_Arms=row["current_Arms"],beta_deg=row["beta_deg"])
        condition=dict(base,cohort="latest_loaded192" if band else "original240_other48",geometry_id="jeet_elmer:"+row["model"],
                       field_derived_history_present=True,field_derived_hash_verified=src["validation_status"]=="PASS",raw_reintegrated_here=cid in raw_audits,
                       archive_present=archive_path.exists(),P2_current_status=record["P2_current_status"],whole_current_status=record["whole_current_status"],
                       period_convergence_status="UNKNOWN",mesh_convergence_status="UNKNOWN",
                       dt_s=dt,steps_per_period=256,calculated_periods=2,selected_period=2,
                       average_start_s=float(p2[0,0]),average_end_s=float(p2[-1,0]),
                       geometry_MODEL_sha256=archive_by["MODEL.json"]["sha256"],input_contract_sha256=archive_by["INPUT_CONTRACT.json"]["sha256"],
                       PARAMETERS_sha256=archive_by["PARAMETERS.json"]["sha256"],solver_input_sif_sha256=archive_by["case.sif"]["sha256"],
                       mesh_nodes_sha256=archive_by["mesh/mesh.nodes"]["sha256"],mesh_elements_sha256=archive_by["mesh/mesh.elements"]["sha256"],
                       mesh_boundary_sha256=archive_by["mesh/mesh.boundary"]["sha256"],native_field_sha256=archive_by["mesh/native.result"]["sha256"],
                       solver_log_sha256=archive_by["solver.log"]["sha256"],circuit_history_sha256=archive_by["circuits.dat"]["sha256"],
                       lamination_BH_sha256=archive_by["lamination_hb.dat"]["sha256"],shaft_BH_sha256=archive_by["shaft_hb.dat"]["sha256"],
                       original_integrator_sha256=data["auditor_sha256"],original_time_contract_source_sha256=data["time_integration_contract"]["source_diff_sha256"],
                       copper_area_min_mm2=min(data["bar_area_mm2"]),copper_area_max_mm2=max(data["bar_area_mm2"]),
                       solver_version=raw_audits.get(cid,{}).get("solver_version","UNKNOWN"))
        conditions.append(condition)
        qstatus="raw_P1_reintegration" if cid in raw_audits else "hash_verified_field_derived_history_reaggregation"
        qstatus+=";P2current="+record["P2_current_status"]+";period_mesh=UNKNOWN"
        for quantity,value in zip(["joule_total_active","transport_DC_prescribed","AC_excess"],means):
            observation(base,quantity,float(value),"Elmer_P1_Joule_minus_prescribed_transport_DC",src["logical_source_key"],src["original_sha256"],"solver_target",qstatus)
        if band:
            for quantity,field in [("AC_excess","hybrid_W"),("proximity_only_evaluator","hybrid_proximity_W"),("skin_excess_evaluator","hybrid_skin_excess_W")]:
                observation(base,quantity,band[field],"canonical_mean_square_field_Hybrid","baseline/handoff/20261019_AF_loaded96/AF_POINTS_192.json",
                            bykey["baseline/handoff/20261019_AF_loaded96/AF_POINTS_192.json"]["original_sha256"],"low_fidelity_prediction","frozen_saved_prediction;raw_MS_missing")
            observation(base,"AC_excess",band["prediction_W"],"frozen_exponent_separable_AF_times_canonical_Hybrid",
                        "baseline/handoff/20261019_AF_loaded96/AF_POINTS_192.json",bykey["baseline/handoff/20261019_AF_loaded96/AF_POINTS_192.json"]["original_sha256"],
                        "frozen_teacher_prediction","already_fitted;not_new_truth")
    np.savez_compressed(OUT/"waveforms/full240_field_derived_histories.npz",**wave_arrays)
    csv_write(OUT/"waveform_reaggregation.csv",wave_summary)
    csv_write(OUT/"latest192_source_link.csv",canonical_delta)
    for key in sorted({"Ref_"+r["ref_point_id"] for r in latest}):
        r=idx[key];man=source(f"ms/{key}/manifest",repo/r["portable_manifest"],r["manifest_sha256"],commit=PIN,kind="MS_acquisition_manifest")
        cache=bykey["baseline/handoff/20261019_rea_hybrid_remote_results/samples/"+key+".npz"]
        receipt_rel=f"handoff/20261019_rea_hybrid_remote_results/evidence/{key}/RECEIPT.json"
        post=pinned(receipt_rel)
        # Original path candidates were explicitly checked; do not treat another host's D: as local data.
        raw=source(f"ms/{key}/ALL_A_HISTORY.npz",Path(r["field_path"]),r["field_sha256"],r["field_bytes"],PIN,"raw_MS_A")
        solver=source(f"ms/{key}/solver_receipt",Path(r["solver_receipt_path"]),r["solver_receipt_sha256"],commit=PIN,kind="MS_solver_receipt")
        with np.load(ROOT/"baseline_audit/sources/handoff/20261019_rea_hybrid_remote_results/samples"/(key+".npz"),allow_pickle=False) as npz:
            assert npz["B_magnitude_T"].shape==(128,600) and np.isfinite(npz["B_magnitude_T"]).all()
        m=read_json(repo/r["portable_manifest"])
        selected=[b for b in latest if "Ref_"+b["ref_point_id"]==key]
        assert all(b["field_sha256"]==r["field_sha256"] and b["manifest_sha256"]==r["manifest_sha256"] for b in selected)
        assert m["diffusion_all_regions_off"] and not m["circuit_ODE_present"] and m["total_steps"]==128
        ms.append(dict(source_key=key,case_id=r["case_id"],conditions_using_source=len(selected),manifest_hash=man["validation_status"],
                       cache_hash=cache["validation_status"],postprocess_receipt_hash=post["validation_status"],raw_A_status=raw["validation_status"],
                       raw_A_present=raw["present_here"],solver_receipt_present=solver["present_here"],raw_A_to_B_replayed=False,
                       source_current_Arms=m["current_rms_A"],source_speed_rpm=m["rpm"],source_beta_deg=m["beta_deg"],
                       mesh_nodes_expected_sha256=r["mesh_nodes_sha256"],solver_log_expected_sha256=r["solver_log_sha256"],
                       field_expected_sha256=r["field_sha256"],manifest_expected_sha256=r["manifest_sha256"],
                       copper_interfaces="removed; uniform homogenized slot source, six turns",scalar_sample_shape="128x600"))
        sanitized={k:m[k] for k in ["case_id","profile","model","radial_factor","current_rms_A","beta_deg","rpm","frequency_Hz","samples_per_period","total_steps","dt_s","omega_m_rad_s","logical_body_to_physical","rotating_physical","winding_source_rows","diffusion_all_regions_off","circuit_ODE_present","individual_Cu_interfaces_removed","source_mesh_grade","source_original_contract_sha256"]}
        (OUT/"ms_metadata").mkdir(exist_ok=True);write_json(OUT/"ms_metadata"/(key+".json"),dict(original_sha256=man["original_sha256"],selected_fields=sanitized))
    csv_write(OUT/"MS_SOURCE_LINKS.csv",ms)
    for row in latest:
        source_key="Ref_"+row["ref_point_id"]
        candidates.append(dict(left_source="Elmer_homogenized_MS/"+source_key,right_source="Elmer_Full/"+row["case_id"],
            geometry_match="nominal_only",OP_match="exact" if row["model"]=="Ref" and row["speed_rpm"]==16000 and row["exact_field_current_match"] else "near_match",
            canonical_current_mapping="exact" if row["exact_field_current_match"] else "bounded_match",
            overall_match="nominal_only",evidence="frozen AF source hash and source-current mapping; homogenized slot and similarity contract",
            differences="MS has no individual Cu interfaces/diffusion; original MS mesh missing; speed-scaled evaluation differs from source solve",
            current_delta_ref_A=row["current_delta_ref_A"],source_speed_rpm=idx[source_key].get("speed_rpm",16000),
            target_speed_rpm=row["speed_rpm"],tolerance="recorded same-beta unique Ref current within 0.2 Arms; no interpolation",
            same_case_pair_label=False))
    halfhash=bykey["baseline/handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json"]["original_sha256"]
    for row in half:
        conditions.append(dict(case_id=row["case_id"],model="HalfSC",cohort=row["role"],geometry_id="jeet_elmer:HalfSC",
            speed_rpm=row["speed_rpm"],current_Arms=row["current_Arms"],beta_deg=row["beta_deg"],
            field_derived_history_present=False,field_derived_hash_verified=False,raw_reintegrated_here=False,archive_present=False,
            P2_current_status="UNKNOWN",historical_P2_current_pass=row["P2_current_pass"],historical_whole_current_pass=row["original_whole_history_current_pass"],
            period_convergence_status="UNKNOWN",mesh_convergence_status="UNKNOWN",period_convergence_qualified=False,mesh_convergence_qualified=False))
        for name in ["RESULT_SUMMARY.json","RESULT.json","PRESERVATION_RECEIPT.json","PRESERVATION_MANIFEST.json"]:
            source(f"halfsc/{row['case_id']}/{name}",None,commit=PIN,kind="missing_HalfSC_raw_link")
        for evaluator,quantity,value,kind in [("saved_Elmer_P2_scalar","AC_excess",row["P2_AC_excess_W"],"solver_target"),
             ("canonical_mean_square_field_Hybrid","AC_excess",row["hybrid_W"],"low_fidelity_prediction"),
             ("frozen_SC_AF_similarity_transfer","AC_excess",row["predictions"]["normalized_current_similarity_speed"]["prediction_W"],"frozen_teacher_prediction")]:
            observation(row,quantity,value,evaluator,"baseline/handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json",halfhash,kind,
                        "saved_scalar_only;raw_missing;historical_P2current_only;period_mesh_unqualified")
    for r in low:
        cid=r["id"];ackpath=nas/"LOWCURRENT_20261019"/(cid+".ACK.json");ack=read_json(ackpath) if ackpath.exists() else {}
        source(f"low48/{cid}/NAS_receipt",ackpath,kind="historical_transport_receipt")
        src=source(f"low48/{cid}/archive",nas/"LOWCURRENT_20261019"/(cid+".tar.gz"),ack.get("sha256"),ack.get("bytes"),kind="raw_archive",hash_now=False)
        conditions.append(dict(case_id=cid,model=r["model"],cohort="low_current_supplement48",geometry_id="jeet_elmer:"+r["model"],
            speed_rpm=r["speed_rpm"],current_Arms=r["current_Arms"],beta_deg=r["beta_deg"],field_derived_history_present=False,
            field_derived_hash_verified=False,raw_reintegrated_here=False,archive_present=src["present_here"],
            P2_current_status="UNKNOWN",whole_current_status="UNKNOWN",period_convergence_status="UNKNOWN",mesh_convergence_status="UNKNOWN"))
    with (ROOT/"geometry_bridge_audit/cases.csv").open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            candidates.append(dict(left_source="MotorCAD20/"+r["design"],right_source="GetDP_R1/"+r["design"],geometry_match="nominal_only",
                 OP_match="no_match",overall_match="nominal_only",evidence="existing geometry_bridge_audit; six parameters match; current CSV is slot-depth corrected",
                 differences="round vs chamfer; missing design readback/insulation/stack; 230.05/460.1 A at 4/8/16krpm beta36 vs 250A/15krpm gamma46.7",
                 tolerance="no authoritative exact-match tolerance; nominal geometry only",slot_depth_difference_mm=r["slot_depth_max_abs_difference_mm"]))
    for model in ["Ref","SC","HalfSC"]:
        for other in ["MotorCAD20","GetDP_R1"]:
            candidates.append(dict(left_source="Elmer/"+model,right_source=other,geometry_match="no_match",OP_match="no_match",overall_match="no_match",
                 evidence="no audited per-design/circuit/material/waveform bridge",differences="different geometry/OP families; R1 ID not sufficient",tolerance="none defined"))
    csv_write(OUT/"geometry_match_candidates.csv",candidates);csv_write(OUT/"observations.csv",observations);csv_write(OUT/"case_inventory.csv",conditions)
    maskfields=["case_id","machine","own_AF_calibration","transferred_probe_coordinate","paper_own_calibration_excluded",
                "paper_own_and_transfer_coordinates_excluded","HalfSC_predeclared32","HalfSC_exploratory4"]
    csv_write(OUT/"baseline_masks.csv",[{k:m[k] for k in maskfields} for m in masks])
    condition_tuple=lambda r:(r["model"],float(r["speed_rpm"]),round(float(r["current_Arms"]),9),float(r["beta_deg"]))
    setfull={condition_tuple(r) for r in full};setlatest={condition_tuple(r) for r in latest};setlow={condition_tuple(r) for r in low}
    groups={k:v for k,v in group_by.items() if k.startswith("jeet_elmer:")}
    split=dict(status="PASS",scope="return membership, identity duplicates and pre-existing global group propagation; no model fit or scoring",
        reserved_ids=sorted(RESERVED),reserved_observation_count=0,reserved_raw_opened_this_run=0,globally_never_opened_claim=False,
        historical_reserved_QC_known=True,geometry_groups=groups,template_group="jeet_e10_ideal_radial_same_template",
        global_catalog_sha256=sha(ROOT/"federated_catalog/output/catalog.jsonl"),global_component_count=277,
        conditions=len(conditions),unique_nominal_condition_count=len({condition_tuple(r) for r in conditions}),
        duplicate_nominal_condition_groups=[list(x) for x in sorted(setfull&setlow)],
        overlap_semantics="same model/current/speed/beta; distinct acquisition revisions retained; no automatic independent-label increment",
        unique_geometry_count=3,independent_reference_template_count=1,original240_latest192_overlap=len(setfull&setlatest),
        original240_low48_overlap=len(setfull&setlow),latest192_low48_overlap=len(setlatest&setlow),
        prohibition="No random per-observation split; all source/fidelity/derived labels of a geometry and conservative alias group stay together",
        exact_cross_source_bridge_count=0,raw_MS_to_Full_pairs_validated_here=0,
        logical_AF_source_to_target_rows=192,source_current_exact_rows=sum(r["exact_field_current_match"] for r in latest),
        source_current_bounded_rows=sum(not r["exact_field_current_match"] for r in latest),
        max_abs_current_delta_ref_A=max(abs(r["current_delta_ref_A"]) for r in latest),
        HalfSC_predeclared32_protected=True,HalfSC_exploratory4_separate=True,new_final_test_count=0,
        masks={model:{k:sum(bool(m[k]) for m in masks if m["machine"]==model) for k in maskfields[2:]} for model in ["Ref","SC","HalfSC"]})
    assert not any(r["geometry"] in RESERVED or any(x in r["condition"] for x in RESERVED) for r in observations)
    assert len(setfull&setlatest)==192 and len(setlatest&setlow)==0
    assert split["masks"]["Ref"]["own_AF_calibration"]==36 and split["masks"]["SC"]["own_AF_calibration"]==27
    assert split["masks"]["SC"]["transferred_probe_coordinate"]==12 and split["masks"]["HalfSC"]["HalfSC_predeclared32"]==32
    write_json(OUT/"split_audit.json",split)
    for cid, audit in raw_audits.items():
        for item in audit["inputs"]:
            sources.append(dict(logical_source_key=item["logical_source_key"],kind="directly_reintegrated_raw_input",source_commit=PIN,
                expected_sha256=item["expected_sha256"],original_sha256=item["sha256"],original_bytes=item["bytes"],present_here=True,
                validation_status=item["status"],validation_reason="hash_checked_by_field_reader_this_run",extraction="read_only; copper ID and topology audit"))
    write_json(OUT/"source_manifest.json",dict(schema="elmer-lineup-return-sources-v1",local_id=config["host_id"],research_handoff_commit=HANDOFF,
        source_object_commit=PIN,source_commit_semantics="pinned Git bytes for packaged files; commit of the referencing metadata for raw acquisition files, not the acquisition code revision",
        paths="logical only; local_path_map.json stays outside Git",sources=sources,
        counts=dict(total=len(sources),by_status=dict(collections.Counter(r["validation_status"] for r in sources))),
        raw_archive_hash_policy="Only selected archives/members read and hashed this run; other NAS files are presence/size only, historical hashes preserved"))
    write_json(Path(config["local_root"])/"local_path_map.json",dict(config=config,logical_source_paths=pathmap))
    latestwave=[r for r in wave_summary if r["latest_loaded192"]]
    quality=dict(schema="elmer-lineup-return-quality-v1",new_solver_runs=0,new_training_runs=0,canonical_AF_refit=False,
        authorization="New direct user follow-up after 2026-10-27 09:00 KST; old automation remains paused",
        scope="MGN/scalar/multisource method development data handoff, not manuscript or reviewer response",
        counts=dict(requested_latest_Full_conditions=192,found_latest_field_derived_histories=len(latestwave),verified_latest_history_hashes=sum(r["source_hash_status"]=="PASS" for r in latestwave),
          original240_conditions=240,full240_histories_reaggregated=len(wave_summary),raw_Full_conditions_reintegrated=len(raw_audits),raw_latest_Full_reintegrated=sum(cid in latest_by for cid in raw_audits),
          raw_low_current_control_reintegrated=sum(cid not in latest_by for cid in raw_audits),
          MS_requested_raw_histories=24,MS_raw_histories_found=sum(r["raw_A_present"] for r in ms),MS_manifests_verified=sum(r["manifest_hash"]=="PASS" for r in ms),
          HalfSC_scalar_conditions=36,HalfSC_raw_found=0,HalfSC_predeclared=32,HalfSC_exploratory=4,
          low48_archive_files_present=sum(r["archive_present"] for r in conditions if r["cohort"]=="low_current_supplement48"),
          source_unique_geometries=3,independent_reference_templates=1,exact_cross_source_bridges=0,
          observations=len(observations),new_independent_Full_FEA_labels=0),
        A_provenance=dict(status="UNKNOWN",latest192_integral_source_hash="PASS" if all(r["source_hash_status"]=="PASS" for r in latestwave) else "FAIL",
            frozen46_hashes="PASS" if all(r["validation_status"]=="PASS" for r in sources if r["kind"]=="frozen_baseline") else "FAIL",
            missing="24 raw MS A + mesh/log/acquisition receipts; 36 HalfSC raw/preservation records; nonselected archive hashes not reread"),
        B_quantities=dict(status="PASS",scope="selected Elmer P1/transport decomposition and fixed canonical Hybrid contract only; other solvers not homogenized",
            conductivity_S_m=47092391.429245,active_length_m=0.15,reference_bare_copper_mm=[3.711,1.686],
            geometry_rule="each raw case uses its own model/area/radial factor; reference dimensions are not assigned to unrelated R1 geometries",
            temperature_direct_readback="UNKNOWN",skin_proximity_decomposition_of_Full="not uniquely defined; no fabricated split"),
        C_reintegration=dict(status="UNKNOWN",field_derived_reaggregation_status="PASS" if all(r["loss_arithmetic_status"]=="PASS" for r in wave_summary) else "FAIL",
            field_derived_loss_error_W=stats([r["max_scalar_difference_W"] for r in wave_summary]),
            latest192_AC_error_W=stats([r["delta_AC_W"] for r in canonical_delta]),
            raw_reintegration_status="PASS" if raw_audits and all(r["raw_loss_reintegration"]=="PASS" for r in raw_audits.values()) else "UNKNOWN",
            raw_AC_error_W=stats([r["delta_W"][2] for r in raw_audits.values()]),
            raw_current_P2_counts=dict(collections.Counter(r["P2_current_status"] for r in raw_audits.values())),
            full240_current_P2_counts=dict(collections.Counter(r["P2_current_status"] for r in wave_summary)),
            full240_current_whole_counts=dict(collections.Counter(r["whole_current_status"] for r in wave_summary)),
            latest192_current_P2_counts=dict(collections.Counter(r["P2_current_status"] for r in latestwave)),
            latest192_current_whole_counts=dict(collections.Counter(r["whole_current_status"] for r in latestwave)),
            raw_MS_A_B_Hybrid_status="UNKNOWN",period_convergence="UNKNOWN",mesh_convergence="UNKNOWN",
            numerical_replay_tolerance_W=1e-6,current_gate_percent=0.001,
            current_gate_source="SCIENTIFIC_GATES.json physical_current and original CURRENT_LOSS_AUDIT physical_gate_pct",
            no_overall_physical_PASS=True),
        D_masks=dict(status="PASS",masks=split["masks"],pseudo_label_new_truth=0),
        E_split=dict(status="PASS",reserved_return_records=0,template_generalization="not demonstrated",global_group_count_preserved=277))
    write_json(OUT/"quality_summary.json",quality)
    print(json.dumps(quality["counts"],indent=2))


if __name__=="__main__":main()
