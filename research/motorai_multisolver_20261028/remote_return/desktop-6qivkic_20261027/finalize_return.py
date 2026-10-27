"""Finalize portable schemas, comparisons, gap queue and artifact hashes."""
from pathlib import Path
import argparse
import collections
import csv
import json
import sys
import numpy as np
from field_reader import sha,read_json,write_json
from build_return import csv_write

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

ARRAYS={
 "node_ids":("1","node","original Elmer node IDs"),
 "node_xy_m":("m","node","stationary copper, original global Cartesian x,y"),
 "element_ids":("1","element","original Elmer triangle IDs"),
 "connectivity_node_ids":("1","element-node","three original node IDs, local corner order"),
 "connectivity_local":("1","element-node","zero-based index into node_ids, not inferred coordinates"),
 "physical":("1","element","2301..2336 copper physical region"),
 "body_ids":("1","element","Elmer logical body ID from INPUT_CONTRACT"),
 "material_id":("1","element","1=copper, no non-copper domains in shard"),
 "material_names":("text","material","material_id 1 maps to first name"),
 "bar_ids":("1","conductor","1..36 sector-local conductor order"),
 "component_ids":("1","conductor","original circuit component ID"),
 "circuit_ids":("1","conductor","original circuit ID"),
 "slot_ids":("1","element","sector-local slot 1..6, six successive conductors per slot"),
 "conductor_in_slot":("1","element","radial-layer order 1..6 from physical-region sequence"),
 "component_phase_deg":("degree","conductor","prescribed sinusoidal source phase, original input convention"),
 "element_area_m2":("m2","element","absolute signed-triangle determinant/2"),
 "bar_area_m2":("m2","conductor","sum of triangles in original copper physical region"),
 "sector_element_volume_m3":("m3","element","area times actual case active length; sector volume"),
 "symmetry_sectors":("1","case","8; multiply sector power by this, not conductor current"),
 "active_length_m":("m","case","actual model axial_length_m"),
 "sigma_S_m":("S/m","case","actual PARAMETERS sigma_cu"),
 "steps":("1","time","original completed step, 1-based"),
 "time_s":("s","time","circuits.dat exact time; uniform dt"),
 "electrical_phase_deg":("degree","time","post-step phase modulo 360, no invented initial phase sample"),
 "A_node_Wb_per_m":("Wb/m","time,node","Elmer magnetic vector potential Az; copper native-result nodal values"),
 "initial_A_node_Wb_per_m":("Wb/m","node","zero initial state before step 1"),
 "B_element_T":("T","time,element,xy","exact P1 curl: (dAz/dy,-dAz/dx), constant per triangle"),
 "J_total_element_node_Am2":("A/m2","time,element-node","sigma*(V_component/L - dAz/dt); total axial conductor current density"),
 "J_residual_element_node_Am2":("A/m2","time,element-node","J_total - I_prescribed/area; analyst residual, NOT an assumed Elmer Je variable"),
 "scalar":("mixed","time,quantity","columns: total_active_W, DC_transport_W, AC_excess_W, residual_J2_W, cross_W, max_current_error_percent"),
 "current_integrated_A":("A","time,conductor","signed integral of total J over conductor cross section; no symmetry factor"),
 "current_prescribed_A":("A","time,conductor","component amplitude_A*sin(2*pi*f*t + phase_deg)"),
 "current_circuit_A":("A","time,conductor","i_component from original circuits.dat"),
 "component_voltage_V":("V","time,conductor","original v_component used with model active length"),
 "B_area_mean_T":("T","time,conductor,xy","area weighted mean Bx/By"),
 "B_area_second_moment_T2":("T2","time,conductor,component","area mean of Bx^2, By^2, Bx*By; not square of area mean"),
 "bar_loss_W":("W","time,conductor,quantity","whole-motor contribution: total, prescribed transport DC, AC excess; sector power already multiplied by 8"),
}


def describe(path,location,logical):
    with np.load(path,allow_pickle=False) as z:
        arrays={}
        for k in z.files:
            u,loc,definition=ARRAYS[k] if k in ARRAYS else ("mixed","time,quantity","field-derived columns: time_s,total_active_W,DC_W,AC_W,max_current_error_percent")
            arrays[k]=dict(shape=list(z[k].shape),dtype=str(z[k].dtype),unit=u,array_location=loc,definition=definition)
    return dict(logical_source_key=logical,storage=location,sha256=sha(path),bytes=path.stat().st_size,arrays=arrays)


def main():
    p=argparse.ArgumentParser();p.add_argument("--path-map",type=Path,required=True);p.add_argument("--allow-base-rebuild",action="store_true");args=p.parse_args()
    if (OUT/'iso_recovery/MS_validation.json').exists() and not args.allow_base_rebuild:
        p.error('This finalizes the pre-iso base snapshot. Follow an intentional --allow-base-rebuild with iso_recovery/merge_recovery.py and refresh_artifacts.py.')
    config=read_json(args.path_map)
    local=Path(config["local_root"]);shards=[]
    path_map=read_json(local/"local_path_map.json")
    private_paths=path_map.setdefault("logical_source_paths",{})
    for path in sorted((OUT/"shards").glob("*.npz")):
        shards.append(describe(path,path.relative_to(OUT).as_posix(),"return/"+path.relative_to(OUT).as_posix()))
    for path in sorted((local/"full_history").glob("*.npz")):
        pid=path.name.replace("_copper_A_all512.npz","")
        shards.append(describe(path,"local_only; resolve with separate private path map","local/full_history/"+pid))
        private_paths["local/full_history/"+pid]=str(path)
    for af in (OUT/"shards").glob("*_audit.json"):
        a=read_json(af);cid=a["case_id"];w=Path(config["workspace"])
        roots=[w/"cases"/cid,w/"outputs/jeet_fields_16000rpm_20261018/work/restored_qa"/cid,
               w/"outputs/jeet_fields_representative_20261018/work/restored_qa"/cid,local/"restored"/cid]
        raw=next(d/"raw" for d in roots if (d/"raw/mesh/native.result").exists())
        for item in a["inputs"]:
            private_paths[item["logical_source_key"]]=str(raw/item["logical_source_key"].split("/raw/",1)[1])
    write_json(local/"local_path_map.json",path_map)
    wave=OUT/"waveforms/full240_field_derived_histories.npz"
    shards.append(describe(wave,wave.relative_to(OUT).as_posix(),"return/waveforms/full240_field_derived_histories.npz"))
    write_json(OUT/"sample_shards_manifest.json",dict(schema="elmer-copper-fe-shards-v1",cases=6,records=shards,
        selection="existing Ref/SC loaded, low-current and high-frequency cases; not error-driven candidate selection",
        spatial_scope="actual copper FE submesh only, 4212 nodes/6912 triangles; not full-machine graph or new MGN training",
        boundary_rule="preserve original node and element IDs; element-node J is never merged across material interfaces; no nearest-coordinate remapping",
        time_rule="8 P2 snapshots for spatial readers; all 512 integrals/current/mean-B/second-moment histories retained; complete A512 stays local",
        original_raw_storage="existing local/NAS, unchanged",validation="shard_validation.json"))
    latest=read_json(ROOT/"baseline_audit/sources/handoff/20261019_AF_loaded96/AF_POINTS_192.json")
    half=read_json(ROOT/"baseline_audit/sources/handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json")
    masks={r["case_id"]:r for r in read_json(ROOT/"elmer_adapter/masks.json")}
    split=read_json(OUT/"split_audit.json")
    paired=[]
    for r in latest+half:
        m=masks[r["case_id"]];is_half=r["model"]=="HalfSC"
        paired.append(dict(case_id=r["case_id"],geometry_id="jeet_elmer:"+r["model"],geometry_group=split["geometry_groups"]["jeet_elmer:"+r["model"]],
            speed_rpm=r["speed_rpm"],current_Arms=r["current_Arms"],beta_deg=r["beta_deg"],Full_AC_excess_W=r["P2_AC_excess_W"],
            canonical_Hybrid_W=r["hybrid_W"],frozen_AF_Hybrid_W=r["predictions"]["normalized_current_similarity_speed"]["prediction_W"] if is_half else r["prediction_W"],
            source_field_sha256=r["MS_field_sha256"] if is_half else r["field_sha256"],
            MS_source_key="Ref_"+(r["hybrid_source_ref_point_id"] if is_half else r["ref_point_id"]),
            own_calibration=m["own_AF_calibration"],transferred_coordinate=m["transferred_probe_coordinate"],
            own_excluded_development=m["paper_own_calibration_excluded"],own_transfer_excluded_development=m["paper_own_and_transfer_coordinates_excluded"],
            HalfSC_predeclared32=m["HalfSC_predeclared32"],HalfSC_exploratory4=m["HalfSC_exploratory4"],
            source_current_mapping="historical_exact" if is_half and r["MS_field_current_matches_target_exactly"] else "bounded" if not is_half and not r["exact_field_current_match"] else "exact",
            source_current_delta_ref_A="" if is_half else r["current_delta_ref_A"],
            raw_target_reintegrated_here=(OUT/"shards"/(r.get("point_id","")+"_audit.json")).exists() if not is_half else False,
            source_quality="saved_scalar_only_raw_missing" if is_half else "hash_verified_field_derived_history;P2current_PASS;period_mesh_UNKNOWN",
            use_policy="protected_historical_transfer_evaluation" if m["HalfSC_predeclared32"] else "exploratory_only" if is_half else "development_only_with_fixed_group_masks",
            new_final_test=False,new_independent_truth=False))
    csv_write(OUT/"comparison_ready.csv",paired)
    queue=[
      dict(queue_id="Q1",priority="A",cases="24 Ref loaded MS histories",target_label="A->P1 curl B[128,600]->canonical Hybrid",action="retrieve_existing_raw_by_logical_key_and_SHA",reason="original A/mesh/log/solver receipts absent on this host",expected_runtime="retrieval depends on source; A content about 24x8.3 MB plus mesh/log; no solve",memory="reader streaming, <1 GiB estimate",disk="0.3-1 GiB estimate including metadata",max_new_solves=0,max_retries=0,owned_process="none",gate="expected SHA and 128x600 sample parity; no invented physical accuracy gate",stop="abort on hash/schema mismatch; retain mismatch separately"),
      dict(queue_id="Q2",priority="B",cases="HalfSC predeclared32 + exploratory4",target_label="P2 Joule/DC/current; preservation lineage",action="retrieve_existing_preservation_manifests_and_selected_raw",reason="RESULT/RESULT_SUMMARY/PRESERVATION records not present locally",expected_runtime="UNKNOWN until archive sizes are returned",memory="UNKNOWN until manifest",disk="UNKNOWN until manifest; no bulk transfer scheduled",max_new_solves=0,max_retries=0,owned_process="none",gate="hash link, fixed P2, preserve period/mesh unqualified",stop="missing/mismatched preservation hashes"),
      dict(queue_id="Q3",priority="B",cases="low-current48, including 24 SC nominal repeats",target_label="distinct acquisition comparison, not 48 independent points",action="selectively_read_existing_NAS_archives_after_case_group_review",reason="NAS files/ACKs found; bulk raw not reread in six-case development sample",expected_runtime="about 35.9 GB compressed total; IO-bound, throughput not measured",memory="streamed <1 GiB estimate",disk="selected extraction only; budget after chosen cases",max_new_solves=0,max_retries=0,owned_process="none",gate="archive/member SHA and preserved 0.001% current gate",stop="raw hash mismatch or insufficient disk"),
      dict(queue_id="Q4",priority="C",cases="existing geometry-only bridge candidates from case_selection",target_label="same physical geometry and OP MS/Full pairs across solvers",action="complete_input_geometry_material_circuit_waveform_readback_first",reason="certified exact cross-source bridge count remains zero",expected_runtime="UNKNOWN; source readback required before any solver estimate",memory="UNKNOWN",disk="UNKNOWN",max_new_solves=0,max_retries=0,owned_process="none",gate="all geometry/OP fields and tolerances explicitly reviewed",stop="unresolved corner/insulation/stack/temperature/phase identity"),
      dict(queue_id="Q5",priority="C",cases="only selected numerical gaps after source recovery",target_label="P2/P3 period, time and mesh sensitivity",action="propose_bounded_plan_only_no_execution",reason="two periods and saved loss replay do not establish convergence",expected_runtime="UNKNOWN until mesh/step/case scope chosen; historical per-case wall costs in source metadata",memory="UNKNOWN; measure retained case RSS before proposal",disk="UNKNOWN; historical raw about 1.82 GB/case is context, not a forecast",max_new_solves=0,max_retries=0,owned_process="none",gate="historical period mean0.1%, waveformL2 0.5%, time/mesh mean1%; scope must match exact case",stop="needs explicit separate run plan with owned PID and cancellation; no automatic launch"),
      dict(queue_id="Q6",priority="research",cases="fixed matched Full budget and grouped evaluation cohort",target_label="AF vs scalar residual vs multisource vs MS pretraining/Field transfer",action="freeze_data_and_comparison_contract_before_training",reason="no recovered raw MS pairs or new independent reference templates yet",expected_runtime="UNKNOWN until model/geometry/OP budget fixed",memory="UNKNOWN; CPU pilot first, GPU not assumed necessary",disk="UNKNOWN",max_new_solves=0,max_retries=0,owned_process="none",gate="same target Full budget, canonical denominator, all-source group closure, reserved exclusion",stop="do not tune on HalfSC32 or reserved geometries; no claim of final superiority from development errors")]
    csv_write(OUT/"case_queue.csv",queue)
    write_json(OUT/"training_policy.json",dict(
        purpose="MGN and scalar/multisource method development; no new model trained in this return",
        strict_physical_truth_cohort="none newly qualified by period+mesh convergence",
        development_cohorts={"latest192":"usable as existing P2 numerical targets with frozen masks; 192 reaggregated, only five raw target histories re-integrated here",
            "HalfSC32":"protected historical transfer evaluation; raw unavailable; never tune new models on this cohort",
            "HalfSC4":"exploratory only, exclude from predeclared32 score",
            "original240_other48":"P2 current FAIL; quarantine from quality-gated target learning",
            "low48":"archive inventory only; 24 SC nominal repeats; do not create new scalar truth rows from ACK",
            "six_field_shards":"five loaded development cases; Ref_P001 is current-failed diagnostic only"},
        label_selector=["quantity=AC_excess","observation_role=solver_target","unit=W","scope=whole_motor_active_stack_2D_no_end_winding","explicit quality/cohort selector required"],
        split_group="geometry_group inherited from federated catalog across all solver/fidelity/derived labels",
        reserved_all_sources=["R1_0199","R1_0076"],globally_never_opened_claim=False,
        ms_teacher_0_84_percent_is_Full_error=False,point36_knn_is_MeshGraphNet=False,
        present_graph="4212-node copper FE submesh with original triangles; spatial reader validated, full-machine/transient MGN training not performed",
        exact_MS_Full_raw_pairs=0,exact_cross_solver_bridges=0,
        comparison_requirement="same Full-FEA budget and evaluation cases; compare frozen AF x canonical mean-square Hybrid, scalar residual, multisource, MS-pretrained/Field models separately from raw Hybrid"))
    write_json(OUT/"runtime.json",dict(python=sys.version.split()[0],numpy=np.__version__,dependencies=["Python >=3.10", "NumPy"],
        solver_invocations=0,training_invocations=0,third_party_installs=0))
    # Canonical LF prevents Git's text normalization from changing published hashes.
    for file in OUT.rglob("*"):
        if file.is_file() and file.suffix in {".py",".md",".json",".csv",".txt"}:
            content=file.read_text(encoding="utf-8-sig");file.write_text(content,encoding="utf-8",newline="\n")
    files=[]
    for file in sorted(OUT.rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts and file.name!="artifact_manifest.json":
            files.append(dict(path=file.relative_to(OUT).as_posix(),sha256=sha(file),bytes=file.stat().st_size))
    write_json(OUT/"artifact_manifest.json",dict(schema="return-artifact-hashes-v1",files=files,total_bytes=sum(f["bytes"] for f in files),
        exclude="artifact_manifest.json self-reference and ignored __pycache__",source_originals_modified=False))
    print(json.dumps(dict(files=len(files),bytes=sum(f["bytes"] for f in files),comparison_rows=len(paired),shards=len(shards))))


if __name__=="__main__":main()
