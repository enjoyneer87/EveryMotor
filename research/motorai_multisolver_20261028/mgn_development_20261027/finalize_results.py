"""Refresh completion/quality summaries from observed files, never promote UNKNOWN."""
from pathlib import Path
import json,csv

ROOT=Path(__file__).resolve().parent


def write(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    read=lambda name:json.loads((ROOT/name).read_text())
    metrics=read('evaluation/metrics.json');conv=read('CONVERGENCE_RESULTS.json');split=read('split_audit.json')
    completed=conv['completed_numerical_variants'];ready=completed==3
    quality=dict(
        source_provenance=dict(status='PASS',MS_source_count=24,comparison_rows=228,original_results_modified=False,
            frozen_source_return_commit='4c6ef9325dd0539b0220fb2817782562b9932fb7',hashes='source_manifest.json and ARTIFACT_MANIFEST.json'),
        quantity_unit_contract=dict(status='PASS',target='full motor active AC excess W, prescribed DC subtracted',
            canonical_denominator_preserved=True,source_current_bounded_mapping_preserved=True,temperature_not_newly_measured=True),
        old_raw_reader_regression=dict(status=read('GENERALIZED_READER_CHECK.json')['status'],max_abs_error_W=max(abs(x) for x in read('GENERALIZED_READER_CHECK.json')['P2_errors_W'].values())),
        new_numerical_variants=dict(status='PASS' if ready else 'UNKNOWN',requested=3,completed=completed,pending=3-completed,
            status_scope='execution and raw readback completion only',unique_physical_conditions=1,new_independent_training_conditions=0),
        formal_period_mesh_time_convergence=dict(status='UNKNOWN',reason='No separately approved qualification gate; proposed development diagnostics and measured errors are in CONVERGENCE_RESULTS.json'),
        baseline_masks=dict(status='PASS',Ref_calibration=36,Ref_excluded=60,SC_calibration=27,SC_excluded=69,SC_strict_excluded=57,HalfSC_predeclared=32,HalfSC_exploratory=4,Full_unique_budget=63),
        leakage_controls=dict(status='PASS',reserved_count=0,HalfSC_training_count=0,global_alias_closure=split['used_components_disjoint_from_reserved'],
            final_model_frozen_before_evaluation=True,after_evaluation_retraining=False,globally_never_opened_claim=False),
        MGN_execution=dict(status='PASS',trained_seeds=[17,29,43],updates_per_seed=18000,original_FE_edges=True,parameters=74018,
            scope='conditional stationary homogenized-slot vector-B model and pooled-feature RBF residual; not Full J or complete moving-mesh temporal model'),
        checkpoint_replay=dict(status=read('CHECKPOINT_REPLAY.json')['status'],models=3),head_replay=dict(status=read('HEAD_REPLAY.json')['status'],models=10,rows_each=228),
        outperform_latest_AF=dict(status='FAIL',scope='observed fixed-set AC wMAE; not a population-wide theorem',
            values=[r for r in metrics['summary'] if r['method'] in ['latest_frozen_AF','scalar','MS_MGN_primary'] and r['group'] in ['Ref_excluded60','SC_strict57','HalfSC_predeclared32']]),
        end_to_end_efficiency_advantage=dict(status='UNKNOWN',reason='No end-to-end advantage demonstrated; cached CPU head timing excludes MS/Hybrid/encoding, AF historical fit cost not rerun'),
        new_template_generalization=dict(status='UNKNOWN',independent_reference_templates=1,scale_variants=3,exact_cross_solver_bridges=0),
        historical_data_qualification=dict(status='UNKNOWN',reason='Ref one-anchor sensitivity does not qualify whole192loaded/36HalfSC; low-current failures remain quarantined'))
    write('quality_summary.json',quality)
    write('PACKAGE_STATE.json',dict(phase='COMPUTATION_COMPLETE' if ready else 'MODELS_COMPLETE_NUMERICS_IN_PROGRESS',model_comparison_complete=True,
        numerical_variants_complete=completed,numerical_variants_requested=3,new_physical_training_labels=0,automatic_resume=False,
        source_automations_unchanged=True,model_changes_after_evaluation=False,formal_period_mesh_time_qualification='UNKNOWN',
        overall_AF_improvement_goal_achieved=False))
    p=ROOT/'case_queue.csv';rows=list(csv.DictReader(p.open(encoding='utf-8')))
    mapping={'Ref_P117_N256_P3':'N256_P3','Ref_P117_N512_P3':'N512_P3','Ref_P117_fine_N256_P3':'fine_N256_P3'}
    for row in rows:
        if row['case'] in mapping and (ROOT/'convergence'/mapping[row['case']]/'audit.json').exists():row['status']='COMPLETED_RAW_AUDIT;FORMAL_CONVERGENCE_UNKNOWN'
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(completed=completed,requested=3,AF_improvement='FAIL',formal_qualification='UNKNOWN')))


if __name__=='__main__':main()
