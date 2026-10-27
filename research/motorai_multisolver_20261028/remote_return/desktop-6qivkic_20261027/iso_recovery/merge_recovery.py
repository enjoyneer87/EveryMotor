"""Merge recovered evidence into the published return without altering baseline sources.

Run after downloading source_records.json, half_results.json, half_shards and
HalfSC_metadata.zip into a PRIVATE local recovery directory. Public source paths
remain logical; raw metadata with machine paths never enter this package.
"""
import argparse, collections, copy, csv, json, sys, zipfile
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1];ISO=OUT/'iso_recovery';sys.path.insert(0,str(OUT))
from field_reader import read_json,write_json,sha
from build_return import csv_write
from finalize_return import describe


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);a=p.parse_args();private=a.private
    progress=read_json(private/'progress.json');assert progress['phase']=='completed' and not progress['failures']
    source=read_json(private/'source_records.json');half=read_json(private/'half_results.json');assert len(half)==36
    unit_rows=[]
    with zipfile.ZipFile(private/'HalfSC_metadata.zip') as z:
        for h in half:
            cid=h['case_id'];old=z.read('halfsc/'+cid+'/derived/CURRENT_LOSS_AUDIT.json')
            old=json.loads(old.decode('utf-8-sig'))
            fraction=max(x['max_field_current_error_per_phasepeak'] for x in old['rows'][256:])
            expected_percent=100*fraction
            assert abs(100*max(x['max_field_current_error_per_phasepeak'] for x in old['rows'])-old['max_field_current_error_pct'])<1e-12
            summary='halfsc/'+cid+'/RESULT_SUMMARY.json'
            if summary in z.namelist():
                saved=json.loads(z.read(summary).decode('utf-8-sig'))['P2_current_metrics_percent']['max_field_current_error_per_phasepeak']
                assert abs(saved-expected_percent)<1e-12
            h['scalar_current_error_delta_percent']=h['P2_max_current_error_percent']-expected_percent
            h['saved_P2_current_error_percent']=expected_percent
            h['saved_history_current_error_unit']='dimensionless ratio; multiply100 for percent summary/gate'
            assert abs(h['scalar_current_error_delta_percent'])<1e-10
            unit_rows.append(dict(case_id=cid,saved_history_max_fraction=fraction,saved_P2_max_percent=expected_percent,
                raw_reintegrated_max_percent=h['P2_max_current_error_percent'],delta_percent=h['scalar_current_error_delta_percent']))
    write_json(ISO/'current_metric_unit_audit.json',dict(status='PASS',cases=36,rows=unit_rows,
        correction='expected comparison metric converted ratio to percent; raw integration/current gate unchanged; original audit files unchanged'))
    ms=read_json(ISO/'MS_validation.json');assert ms['raw_MS_count']==24 and ms['A_B_numerical_status']=='PASS' and ms['Hybrid_replay_status']=='PASS'
    byhalf={x['case_id']:x for x in half};pairs=rows(ISO/'MS_Full_pairs.csv');bypair={r['case_id']:r for r in pairs}
    base=OUT.parents[1]/'baseline_audit/sources';frozen_half=read_json(base/'handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json')
    # Check original manifest hashes against the frozen 32-row handoff, not only
    # against a co-located preservation receipt. Four exploratory rows have no such root.
    bysource={x['logical_source_key']:x for x in source};lineage=[]
    for row in frozen_half:
        cid=row['case_id'];want=row.get('source_raw_manifest_sha256');got=bysource['halfsc/'+cid+'/ARCHIVE_SHA256.json']['sha256']
        if want is not None:assert want==got
        lineage.append(dict(case_id=cid,role=row['role'],canonical_archive_manifest_sha256=want,observed_archive_manifest_sha256=got,
            canonical_hash_root_status='PASS' if want==got else 'UNKNOWN',canonical_scalar_delta_W=byhalf[cid]['canonical_scalar_delta_W'],
            preservation_receipt_available=byhalf[cid]['preservation_receipt_available']))
        if want is not None:
            bysource['halfsc/'+cid+'/ARCHIVE_SHA256.json'].update(expected_sha256=want,status='PASS')
    write_json(ISO/'HalfSC_lineage.json',dict(rows=lineage,canonical_hash_root_pass=32,canonical_hash_root_unknown=4,
        unknown_reason='exploratory4 handoff has scalar/case linkage but no frozen archive-manifest hash or standalone preservation receipt'))
    write_json(ISO/'HalfSC_raw_audits.json',half)
    # Keep every original timestep and conductor current; do not derive mean loss
    # from the eight spatial snapshots. Original detailed integrals also stay private.
    wavekeys=['time_s','scalar','current_integrated_A','current_prescribed_A','current_circuit_A']
    waves={k:[] for k in wavekeys};case_ids=[]
    for h in half:
        z=np.load(private/'half_shards'/(h['case_id']+'_full_history_integrals.npz'))
        for k in wavekeys:waves[k].append(z[k])
        case_ids.append(h['case_id'])
        assert np.max(np.abs(z['scalar'][256:,:3].mean(axis=0)-h['P2_saved_W']))<=1e-6
    np.savez_compressed(ISO/'HalfSC36_waveforms.npz',case_ids=np.array(case_ids),**{k:np.array(v) for k,v in waves.items()})
    selected=['HALFSC_FIG14_N16000_I690_B36_N256_P2_A02','HALFVAL_N2000_L1_B00_N256_P2_A01','HALFVAL_N16000_L4_B90_N256_P2_A01']
    import shutil
    sharddir=ISO/'shards';sharddir.mkdir(exist_ok=True)
    for cid in selected:
        for suffix in ['_field_snapshots.npz','_full_history_integrals.npz']:
            src=private/'half_shards'/(cid+suffix);shutil.copyfile(src,sharddir/src.name);assert sha(src)==sha(sharddir/src.name)
    # Independent quadrature/ID checks use the same portable validator as prior six.
    from validate_shards import validate
    validate(ISO)
    allraw=[read_json(x) for x in (OUT/'shards').glob('*_audit.json')]+half
    rawids={r['case_id'] for r in allraw};rawpairids=rawids & set(bypair)
    assert len(allraw)==42 and len(rawpairids)==41
    for row in pairs:
        row['Full_raw_reintegrated']=row['case_id'] in rawpairids
        row['both_raw_numerically_verified']=row['case_id'] in rawpairids
        row['shared_MS_history_group']='source_sha256:'+row['MS_source_sha256']
        row['pair_scope']='canonical evaluator lineage; declared MS homogenization/scaling, not exact physical same-OP pair'
    csv_write(ISO/'MS_Full_pairs.csv',pairs)
    # Replace matching missing records; retain aliases explicitly, not as new acquisitions.
    manifest=read_json(OUT/'source_manifest.json');sources={x['logical_source_key']:x for x in manifest['sources']}
    for item in source+[i for h in half for i in h['inputs']]:
        key=item['logical_source_key'];old=sources.get(key,{})
        if old.get('expected_sha256') is not None:assert old['expected_sha256']==item['sha256']
        sources[key]=dict(logical_source_key=key,kind='recovered_original_file',source_commit=manifest['source_object_commit'],
            expected_sha256=item.get('expected_sha256'),expected_bytes=old.get('expected_bytes'),original_sha256=item['sha256'],original_bytes=item['bytes'],
            present_here=True,validation_status=item['status'],validation_reason='SHA checked on iso; raw originals unchanged',
            extraction='read_only; iso recovery; private path map',storage_host='iso DESKTOP-513NQ4P',
            root_hash_scope='see iso_recovery/HalfSC_lineage.json for exploratory4 root UNKNOWN')
    for key in list(sources):
        alias=key.replace('/solver_receipt','/MS_RECEIPT.json') if key.startswith('ms/') and key.endswith('/solver_receipt') else key.replace('/manifest','/MS_MANIFEST.json') if key.startswith('ms/') and key.endswith('/manifest') else None
        if alias in sources:
            sources[key]=dict(sources[alias],logical_source_key=key,alias_of=alias)
    manifest.update(sources=list(sources.values()),counts=dict(total=len(sources),by_status=dict(collections.Counter(x['validation_status'] for x in sources.values()))),
        recovery_scope='MS24 original A/input/mesh/log plus HalfSC36 consumed native/input/circuit/mesh/VTU/log hashes; not all 512 VTU files rehashed')
    write_json(OUT/'source_manifest.json',manifest)
    links=rows(OUT/'MS_SOURCE_LINKS.csv')
    for r in links:r.update(raw_A_status='PASS',raw_A_present=True,solver_receipt_present=True,raw_A_to_B_replayed=True,recovery_host='iso DESKTOP-513NQ4P',
        conditions_using_source_scope='latest192 only',conditions_using_source_including_HalfSC=sum(x['MS_source_key']==r['source_key'] for x in pairs))
    csv_write(OUT/'MS_SOURCE_LINKS.csv',links)
    compare=rows(OUT/'comparison_ready.csv')
    for r in compare:
        cid=r['case_id'];r.update(raw_MS_A_B_verified=True,canonical_Hybrid_raw_replay=True,raw_target_reintegrated_here=cid in rawpairids,
            both_raw_numerically_verified=cid in rawpairids,source_history_group=bypair[cid]['shared_MS_history_group'])
        if cid in byhalf:r['source_quality']='raw_P1_reintegration;P2current='+byhalf[cid]['P2_current_status']+';period_mesh_UNKNOWN'
    csv_write(OUT/'comparison_ready.csv',compare)
    inventory=rows(OUT/'case_inventory.csv')
    for r in inventory:
        if r['case_id'] not in byhalf:continue
        h=byhalf[r['case_id']];hashes={i['logical_source_key'].split('/raw/')[1]:i['sha256'] for i in h['inputs']}
        r.update(field_derived_history_present=True,field_derived_hash_verified=r['cohort']=='new_validation',raw_reintegrated_here=True,archive_present=True,
            P2_current_status=h['P2_current_status'],whole_current_status=h['whole_current_status'],dt_s=h['dt_s'],steps_per_period=256,calculated_periods=2,
            selected_period=2,average_start_s=h['P2_time_s'][0],average_end_s=h['P2_time_s'][1],solver_version=h['solver_version'])
        for col,rel in [('geometry_MODEL_sha256','MODEL.json'),('input_contract_sha256','INPUT_CONTRACT.json'),('PARAMETERS_sha256','PARAMETERS.json'),
            ('solver_input_sif_sha256','case.sif'),('mesh_nodes_sha256','mesh/mesh.nodes'),('mesh_elements_sha256','mesh/mesh.elements'),
            ('mesh_boundary_sha256','mesh/mesh.boundary'),('native_field_sha256','mesh/native.result'),('solver_log_sha256','solver.log'),('circuit_history_sha256','circuits.dat')]:r[col]=hashes[rel]
        with np.load(private/'half_shards'/(h['case_id']+'_field_snapshots.npz')) as geometry:
            r['copper_area_min_mm2']=float(geometry['bar_area_m2'].min()*1e6)
            r['copper_area_max_mm2']=float(geometry['bar_area_m2'].max()*1e6)
    csv_write(OUT/'case_inventory.csv',inventory)
    obs=rows(OUT/'observations.csv');existing={r['observation_id'] for r in obs};new=[]
    for r in obs:
        if r['condition'] not in byhalf:continue
        h=byhalf[r['condition']];r.update(sigma_S_m=h['sigma_S_m'],active_length_m=h['active_length_m'])
        if r['observation_role']=='solver_target':
            r['quality']='raw_P1_reintegration;P2current='+h['P2_current_status']+';period_mesh_UNKNOWN'
            r['raw_native_source_hash']=next(i['sha256'] for i in h['inputs'] if i['logical_source_key'].endswith('/mesh/native.result'))
            for index,quantity in [(0,'joule_total_active'),(1,'transport_DC')]:
                nr=dict(r,observation_id=h['case_id']+':Elmer_P1_Joule_minus_prescribed_transport_DC:'+quantity,
                    evaluator='Elmer_P1_Joule_minus_prescribed_transport_DC',quantity=quantity,value=h['P2_reintegrated_W'][index],
                    source_key='full/'+h['case_id']+'/raw/mesh/native.result',source_hash=r['raw_native_source_hash'])
                if nr['observation_id'] not in existing:new.append(nr)
        else:r['quality']='raw_MS_denominator_verified;frozen_prediction;not_solver_truth'
    obs.extend(new);csv_write(OUT/'observations.csv',obs)
    match=rows(OUT/'geometry_match_candidates.csv')
    for r in match:r['differences']=r['differences'].replace('original MS mesh missing','original MS mesh recovered; homogenized vs discrete conductor interfaces differ')
    present_targets={x['right_source'] for x in match}
    for r in pairs:
        if r['model']!='HalfSC' or 'Elmer_Full/'+r['case_id'] in present_targets:continue
        match.append(dict(left_source='Elmer_homogenized_MS/'+r['MS_source_key'],right_source='Elmer_Full/'+r['case_id'],
            geometry_match='nominal_only',OP_match='near_match',canonical_current_mapping=r['recorded_mapping'],overall_match='nominal_only',
            evidence='verified raw MS A and HalfSC native integrals; frozen similarity evaluator lineage',
            differences='Ref homogenized-slot MS -> HalfSC1.5 similarity; diffusion/individual copper interfaces differ; source speed16000 vs target speed; exact input/temperature identity not established',
            current_delta_ref_A=r['computed_target_minus_source_ref_A'],source_speed_rpm=r['source_speed_rpm'],target_speed_rpm=r['target_speed_rpm'],
            tolerance='no exact physical-equivalence tolerance approved; preserve recorded mapping',same_case_pair_label=False,slot_depth_difference_mm=''))
    csv_write(OUT/'geometry_match_candidates.csv',match)
    split=read_json(OUT/'split_audit.json');split.update(raw_MS_to_Full_pairs_validated_here=len(rawpairids),logical_AF_source_to_target_rows=228,
        raw_pair_scope='5 loaded192 + HalfSC36; source/target numerical replay, not exact same-geometry-and-OP bridge',
        recovered_case_additions=0,MS_history_reuse='Ref MS24 already shared by Ref/SC/HalfSC; group source hash for inductive pretraining splits',
        shared_MS_sources_allowed_across_target_geometries_only_with_explicit_transductive_protocol=True)
    write_json(OUT/'split_audit.json',split)
    q=read_json(OUT/'quality_summary.json');q['counts'].update(raw_Full_conditions_reintegrated=42,MS_raw_histories_found=24,MS_raw_histories_missing=0,
        HalfSC_raw_found=36,HalfSC_raw_reintegrated=36,HalfSC_raw_missing=0,raw_MS_to_Full_pairs=41,logical_MS_to_Full_pairs=228,observations=len(obs))
    q['A_provenance'].update(status='UNKNOWN',MS24_original_hashes='PASS',HalfSC32_frozen_archive_manifest_hash='PASS',
        HalfSC36_consumed_raw_member_hash='PASS',HalfSC4_independent_archive_hash_root='UNKNOWN',
        missing='No MS24 or HalfSC36 raw case missing. Exploratory4 standalone preservation receipts/frozen archive-root hashes absent; other nonselected Full native files not reread.')
    c=q['C_reintegration'];c.update(raw_MS_A_B_Hybrid_status='PASS',raw_AC_error_W=dict(count=42,max_abs=max(abs(x['delta_W'][2]) for x in allraw),mean_abs=float(np.mean([abs(x['delta_W'][2]) for x in allraw]))),
        raw_current_P2_counts=dict(collections.Counter(x['P2_current_status'] for x in allraw)),
        HalfSC36_current_P2_counts=dict(collections.Counter(x['P2_current_status'] for x in half)),
        HalfSC36_current_whole_counts=dict(collections.Counter(x['whole_current_status'] for x in half)),
        HalfSC36_max_abs_AC_delta_W=max(abs(x['delta_W'][2]) for x in half),MS24_B_cache_bitwise_equal=sum(x['B_cache_bitwise_equal'] for x in ms['sources']),
        raw_MS_Hybrid_max_abs_delta_W=ms['Hybrid_max_abs_delta_W'])
    write_json(OUT/'quality_summary.json',q)
    policy=read_json(OUT/'training_policy.json');policy['development_cohorts'].update(
        latest192='P2 development targets with frozen masks; MS raw24 now verified for all192; target raw5 re-integrated',
        HalfSC32='Raw36 recovered and reintegrated; these32 remain protected historical transfer evaluation, including their fields; never pretrain/tune on them',
        HalfSC4='Exploratory only; raw loss/current verified; high-speed whole-history current FAIL retained',
        MS24='Raw A and stationary homogenized-slot FE graph available; MS teacher only, no Full-FEA J/loss truth')
    policy.update(raw_MS_Full_lineage_pairs=41,exact_MS_Full_raw_pairs=0,
        exact_pair_semantics='exact means same physical geometry and OP; 41 verified evaluator-lineage pairs include homogenization and similarity/speed transfers',
        shared_source_history_policy='Inductive geometry/template splits must close over shared MS source hash; a known-template/transductive study must be labeled separately')
    write_json(OUT/'training_policy.json',policy)
    queue=rows(OUT/'case_queue.csv')
    for r in queue:
        if r['queue_id']=='Q1':r.update(action='COMPLETED_existing_raw_recovered',reason='24/24 hash verified; A->B bitwise parity; canonical denominator228 replayed',expected_runtime='completed; no new solver')
        elif r['queue_id']=='Q2':r.update(action='COMPLETED_native36_reintegrated_with_lineage_limits',reason='32 frozen archive roots match; exploratory4 scalar/member linkage exists but standalone preservation root UNKNOWN',expected_runtime='completed; no new solver')
        elif r['queue_id']=='Q6':r['reason']='MS24 and HalfSC36 recovered; independent reference templates and frozen equal-budget protocol still required'
        r['new_FEA_training_authorized']=True;r['authorization_scope']='user follow-up permits new FEA/training; bounded case/model execution plan required before starting; this recovery ran neither'
    csv_write(OUT/'case_queue.csv',queue)
    shard=read_json(OUT/'sample_shards_manifest.json');kept=[x for x in shard['records'] if not x['storage'].startswith('iso_recovery/')]
    for f in sorted(ISO.rglob('*.npz')):
        desc=describe(f,f.relative_to(OUT).as_posix(),'return/'+f.relative_to(OUT).as_posix())
        if '_MS_slots' in f.name:
            desc['scope']='stationary homogenized MS slots2401..2406;128 phases; no individual copper/circuit/J'
            desc['arrays']['physical'].update(definition='2401..2406 homogenized source slots')
            desc['arrays']['slot_ids'].update(definition='slot1..6 from physical-2400')
            for k in ['source_current_Arms','source_beta_deg','source_radial_factor','source_speed_rpm','edge_index']:
                desc['arrays'][k].update(unit={'source_current_Arms':'Arms','source_beta_deg':'degree','source_radial_factor':'1','source_speed_rpm':'rpm','edge_index':'1'}[k],definition='source manifest value' if k!='edge_index' else 'directed original triangle edges; local node indices')
        if f.name=='MS24_reconstructed_B.npz':
            for k in desc['arrays']:desc['arrays'][k].update(unit='T',array_location='phase,sample',definition='canonical slot1 |B| at600 samples,128 phases; source key is array name')
        if f.name=='HalfSC36_waveforms.npz':
            for k in desc['arrays']:
                desc['arrays'][k]['array_location']='case,'+desc['arrays'][k]['array_location']
            desc['arrays']['case_ids'].update(unit='text',array_location='case',definition='case order for every stacked waveform array')
        kept.append(desc)
    # Retain detailed 512-step B moments/voltages for all36 outside Git.
    kept=[x for x in kept if not x['logical_source_key'].startswith('iso/recovered_halfsc/')]
    for f in sorted((private/'half_shards').glob('*_full_history_integrals.npz')):
        kept.append(describe(f,'iso_and_local_private; resolve with separate private path map','iso/recovered_halfsc/'+f.name))
    shard.update(records=kept,cases=9,MS_graph_cases=2,raw_reintegrated_cases=42,
        original_raw_storage='MS recovered local copy and original iso; Full original local/NAS; HalfSC original iso unchanged',
        additional_validation='iso_recovery/shard_validation.json and MS_graph_validation.json',
        protected_fields='HalfSC predeclared32 field examples retain evaluation-only status')
    write_json(OUT/'sample_shards_manifest.json',shard)
    runtime=read_json(OUT/'runtime.json');execution=read_json(private/'execution_receipt.json')
    runtime['iso_recovery_runtime']={k:execution[k] for k in ['python','numpy','matplotlib']}
    runtime['canonical_MS_dependencies']=['NumPy','matplotlib'];write_json(OUT/'runtime.json',runtime)
    execution['published_code_note']='After execution: parent-import portability and expected-current ratio-to-percent adapter corrected. Raw integration and gates unchanged. merge_recovery revalidates all36 expected metrics against preserved histories/summaries; see current_metric_unit_audit.json. Executed bytes retained privately.'
    write_json(ISO/'execution_receipt.json',execution)
    write_json(ISO/'recovery_summary.json',dict(host='iso DESKTOP-513NQ4P',receiving_local='desktop-6qivkic',MS_requested=24,MS_found=24,HalfSC_requested=36,HalfSC_found=36,
        requested_unique_existing_cases=60,missing_cases=0,duplicate_physical_acquisitions=0,source_geometries=3,
        source_geometries_scope='cumulative return package Ref/SC/HalfSC',requested_raw_geometry_variants=2,reference_templates=1,
        raw_target_reintegrations_new=36,raw_target_reintegrations_total=42,both_raw_lineage_pairs=41,logical_MS_Full_pairs=228,
        canonical_Hybrid_max_abs_delta_W=ms['Hybrid_max_abs_delta_W'],HalfSC_max_abs_AC_delta_W=c['HalfSC36_max_abs_AC_delta_W'],
        HalfSC_P2_current=c['HalfSC36_current_P2_counts'],HalfSC_whole_current=c['HalfSC36_current_whole_counts'],
        period_convergence='UNKNOWN',mesh_convergence='UNKNOWN',exact_cross_solver_bridges=0,solver_calls=0,training_calls=0,AF_refit=False,
        elapsed_read_only_recovery_seconds=progress['elapsed_s'],source_files_modified=False))
    print(json.dumps(dict(observations=len(obs),both_raw_pairs=len(rawpairids),sources=len(sources),new_HalfSC_raw=len(half))))


if __name__=='__main__':main()
