"""Publish small predictions/provenance; preserve large source arrays privately."""
from pathlib import Path
import csv,json,hashlib,argparse
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-data',type=Path,required=True);a=ap.parse_args()
    split=json.loads((HERE/'SPLIT_MANIFEST.json').read_text());returned=HERE.parent/'remote_return/desktop-6qivkic_20261027'
    rows=list(csv.DictReader((returned/'comparison_ready.csv').open()));lookup={r['case_id']:r for r in rows}
    previous_split=json.loads((returned/'split_audit.json').read_text())
    catalog_root=HERE.parent/'federated_catalog/output'
    assert sha(catalog_root/'catalog.jsonl')==previous_split['global_catalog_sha256']
    identity_index=[json.loads(line) for line in (catalog_root/'index.jsonl').read_text().splitlines()]
    reserved_groups={row['global_component_id'] for row in identity_index if row['reserved_all_source']}
    used_groups={row['geometry_group'] for row in rows}
    assert used_groups.isdisjoint(reserved_groups)
    assert all(previous_split['geometry_groups'][row['geometry_id']]==row['geometry_group'] for row in rows)
    ms_hash={r['source_key']:r['field_sha256'] for r in split['MS_sources']}
    records=[];model_sources=[]
    for stage in ['stage1','stage2']:
        folder=HERE/'artifacts'/stage;pred=json.loads((folder/'predictions_frozen.json').read_text())
        for model,values in pred.items():
            key=f'jeet_mgn_20261027:{stage}:{model}';head_hash=sha(folder/(model+'_head.npz'))
            source=dict(logical_source_key=key,head_sha256=head_hash,head_bytes=(folder/(model+'_head.npz')).stat().st_size,Full_budget=63)
            if model!='scalar':
                source['neural_weights_sha256']=sha(folder/(model+'.pt'))
                source['embedding_sha256']=sha(folder/f'embeddings_seed{model.split("seed")[-1]}.npz')
            model_sources.append(source)
            for case,value in values.items():
                row=lookup[case]
                records.append(dict(template='JEET_e10_Ref_family',geometry=row['geometry_id'],geometry_group=row['geometry_group'],condition=case,
                    source=key,evaluator='canonical_mean_square_Hybrid_times_exp_log_residual',quantity='predicted_active_AC_excess',scope='full_motor_active_length_0.15m',
                    unit='W',value=value,temperature='sigma_contract_47092391.429245_S_per_m;per_case_temperature_not_newly_measured',
                    waveform='balanced_sinusoidal_prescribed_RMS_current',average_window='target_P2_uniform_post_step_mean',quality='model_prediction_not_independent_truth',
                    source_hash=head_hash,MS_source_key=row['MS_source_key'],MS_source_hash=ms_hash[row['MS_source_key']],
                    source_current_mapping=row['source_current_mapping'],source_current_delta_ref_A=row['source_current_delta_ref_A'],
                    own_calibration=row['own_calibration'],transferred_coordinate=row['transferred_coordinate'],HalfSC_predeclared32=row['HalfSC_predeclared32'],
                    HalfSC_exploratory4=row['HalfSC_exploratory4'],new_independent_truth=False,new_final_test=False))
    with (HERE/'observations.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with np.load(a.training_data/'MS_graph_data.npz') as z:
        keys=['node_xy_m','node_ids','element_ids','connectivity_local','edge_index','element_area_m2','node_area_weights','node_static','source_features','source_keys','phases_rad']
        np.savez_compressed(HERE/'artifacts/graph_topology.npz',**{k:z[k] for k in keys})
        arrays={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
    fea_shards=[];fea_sources=[]
    for folder in sorted((HERE/'convergence').glob('*')):
        if not (folder/'shard_manifest.json').exists():continue
        fea_shards.append(dict(case=folder.name,manifest=folder.relative_to(HERE).as_posix()+'/shard_manifest.json',
            location=folder.relative_to(HERE).as_posix()+'/copper_graph_8phases.npz',sha256=sha(folder/'copper_graph_8phases.npz'),bytes=(folder/'copper_graph_8phases.npz').stat().st_size,
            full_time_scalar_loss_current_preserved=True,sparse_field_snapshots_are_not_period_loss_labels=True))
        raw_manifest=json.loads((folder/'raw_source_manifest.json').read_text());fea_sources.append(dict(case=folder.name,logical_source_key=raw_manifest['logical_source_key'],
            raw_manifest=folder.relative_to(HERE).as_posix()+'/raw_source_manifest.json',raw_manifest_sha256=sha(folder/'raw_source_manifest.json'),raw_bytes=raw_manifest['total_raw_bytes'],new_independent_training_label=False))
    write(HERE/'sample_shards_manifest.json',dict(FEA_validation_shards=fea_shards,graph_topology=dict(location='artifacts/graph_topology.npz',sha256=sha(HERE/'artifacts/graph_topology.npz'),
        bytes=(HERE/'artifacts/graph_topology.npz').stat().st_size,scope='topology and source OP features only; no full B tensor',arrays={k:v for k,v in arrays.items() if k!='B_node_area_projected_T'}),
        training_tensor=dict(logical_source_key='jeet_mgn_20261027:MS24_projected_B',location='private_source_storage',sha256=sha(a.training_data/'MS_graph_data.npz'),
        bytes=(a.training_data/'MS_graph_data.npz').stat().st_size,arrays=arrays,units='node xy m; area m2; B T; phase rad; features normalized as DATA_CONTRACT.json',
        reader='prepare_data.py and immutable prior return iso_recovery/load_ms_graph.py',original_ID_linkage='original Elmer node/element IDs retained, triangle adjacency; no coordinate matching')))
    write(HERE/'source_manifest.json',dict(source_return_commit='4c6ef9325dd0539b0220fb2817782562b9932fb7',AF_source_commit='0034d7c24b1cc0cf41499d028733792b14815bc7',
        MS_sources=split['MS_sources'],prediction_sources=model_sources,FEA_validation_sources=fea_sources,comparison_source_sha256=sha(returned/'comparison_ready.csv'),
        preserved_execution_sources=[dict(location=p.relative_to(HERE).as_posix(),sha256=sha(p)) for p in sorted((HERE/'execution_sources').glob('*.py'))],
        AF_objects=[dict(logical_source_key=f'loaded96:{name}',sha256=sha(HERE.parent/'baseline_audit/sources/handoff/20261019_AF_loaded96'/name),bytes=(HERE.parent/'baseline_audit/sources/handoff/20261019_AF_loaded96'/name).stat().st_size) for name in ['AF_model_Ref_loaded96.json','AF_model_SC_loaded96.json','AF_POINTS_192.json']],
        training_tensor_sha256=sha(a.training_data/'MS_graph_data.npz'),extraction_provenance='Recovered original24MS source hashes -> original FEcurl -> nodal projection -> fixed MGN -> calibration-only RBF head',
        original_results_modified=False,local_paths='kept in private path mapping, not public catalog'))
    write(HERE/'split_audit.json',dict(status='PASS',scope='condition/calibration exclusions only; not a geometry-generalization test',
        global_alias_index_sha256=sha(catalog_root/'index.jsonl'),global_catalog_sha256=previous_split['global_catalog_sha256'],
        global_reserved_components=sorted(reserved_groups),used_components_disjoint_from_reserved=True,
        calibration_count=63,Ref=36,SC=27,HalfSC_training=0,MS_pretraining_sources=24,MS_pretraining_geometry='Ref only',
        reserved_aliases=['R1_0199','R1_0076'],reserved_observations_in_training_or_evaluation=0,globally_never_opened_claim=False,
        source_groups={key:[r['case_id'] for r in rows if r['MS_source_key']==key] for key in ms_hash},
        geometry_groups={key:sorted(set(r['geometry_id'] for r in rows if r['geometry_group']==key)) for key in sorted(set(r['geometry_group'] for r in rows))},
        groups=split['groups'],duplicates=dict(same_condition_across_model_predictions='intentional separate predictions, never added to independent truth count',
        Full_numerical_variants='same Ref_P117 condition, no extra independent fit labels'),
        model_hyperparameters='calibration-source-group CV only; all final models frozen before evaluation',
        HalfSC='historical predeclared32 protected from this run training/tuning; exploratory4 separate; historical QC acknowledged',
        new_reference_template_generalization='UNKNOWN: no independent template test',exact_cross_solver_bridges=0))
    print(json.dumps(dict(prediction_observations=len(records),models=len(model_sources),MS_sources=24)))


if __name__=='__main__':main()
