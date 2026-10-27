"""Reconstruct canonical MS samples from recovered A and validate 228 denominators.

Requires the pinned baseline_audit/sources directory and NumPy/matplotlib.
No solver, fit, AF coefficient change or model training occurs.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from field_reader import read_json, write_json, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--ms',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False);sys.path.insert(0,str(a.sources))
    from jeet_acloss_rbf.rea_hybrid.postprocess_elmer import extract_samples
    from jeet_acloss_rbf.rea_hybrid.e10_sampling import evaluate_active_samples
    from matplotlib.tri import Triangulation
    contract=read_json(a.sources/'jeet_acloss_rbf/rea_hybrid/examples/e10_ref_input.json')
    latest=read_json(a.sources/'handoff/20261019_AF_loaded96/AF_POINTS_192.json')
    half=read_json(a.sources/'handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json')
    audits=[];cache={};sample_arrays={}
    for d in sorted(a.ms.iterdir()):
        field=d/'ALL_A_HISTORY.npz';m=read_json(d/'MS_MANIFEST.json');b,g=extract_samples(field,m,contract)
        with np.load(field) as f:ids=f['node_ids'];xy=f['node_xy_m'];tri=f['triangles'];ph=f['physical'];A=f['A_node_Wb_per_m']
        mesh=np.loadtxt(d/'raw/mesh/mesh.nodes');elements=np.loadtxt(d/'raw/mesh/mesh.elements',dtype=np.int64)
        lookup={int(n):i for i,n in enumerate(mesh[:,0])}
        expected={(int(e[1]),tuple(sorted(map(int,e[3:])))) for e in elements if e[2]==303}
        actual={(int(t),tuple(sorted(map(int,ids[conn])))) for t,conn in zip(ph,tri)}
        assert expected==actual and len(np.unique(ids))==len(ids)
        chosen=np.flatnonzero(ph==2401);used=np.unique(tri[chosen]);mi=[lookup[int(n)] for n in ids[used]]
        xyerr=float(np.max(np.abs(xy[used]-mesh[mi,2:4])))
        assert xyerr<1e-12
        found=Triangulation(xy[:,0],xy[:,1],tri[chosen]).get_trifinder()(*(g['points_m']*m['radial_factor']).T)
        assert np.all(found>=0);ti=chosen[found];v=xy[tri[ti]];u=v[:,1]-v[:,0];w=v[:,2]-v[:,0]
        det=u[:,0]*w[:,1]-u[:,1]*w[:,0];av=A[:,tri[ti]];dv=av[:,:,1:]-av[:,:,:1]
        gx=(dv[:,:,0]*w[:,1]-dv[:,:,1]*u[:,1])/det
        gy=(u[:,0]*dv[:,:,1]-w[:,0]*dv[:,:,0])/det
        independent=np.sqrt(gx*gx+gy*gy)
        old=np.load(a.sources/'handoff/20261019_rea_hybrid_remote_results/samples'/(d.name+'.npz'))['B_magnitude_T']
        diff=b-old;ind=independent-b
        assert b.shape==(128,600)
        source=m['winding_source_rows']
        source_current_error=max(abs(x['J_peak_Am2']*x['area_m2']-x['series_turns']*np.sqrt(2)*m['current_rms_A']) for x in source)
        item=dict(source_key=d.name,case_id=m['case_id'],sha256=sha(field),manifest_sha256=sha(d/'MS_MANIFEST.json'),
            shape=list(b.shape),source_current_Arms=m['current_rms_A'],source_speed_rpm=m['rpm'],source_beta_deg=m['beta_deg'],
            source_radial_factor=m['radial_factor'],mesh_nodes=len(ids),mesh_triangles=len(tri),topology_ID_set_equal=True,
            stationary_slot_coordinate_error_m=xyerr,B_cache_max_abs_T=float(np.max(np.abs(diff))),B_cache_mean_abs_T=float(np.mean(np.abs(diff))),
            B_cache_relative_L2=float(np.linalg.norm(diff)/np.linalg.norm(old)),B_cache_bitwise_equal=bool(np.array_equal(b,old)),
            independent_curl_max_abs_T=float(np.max(np.abs(ind))),ampere_turn_max_error_A=source_current_error,
            numerical_reproduction_status='PASS' if np.max(np.abs(diff))<=1e-12 and np.max(np.abs(ind))<=1e-12 else 'FAIL',
            numerical_tolerance_T=1e-12,numerical_gate_scope='floating-point implementation parity only; NOT a mesh or physical accuracy qualification',
            physical_accuracy_status='UNKNOWN',phase_window='128 distinct post-step MS phases, one electrical cycle',mesh_convergence_qualified=False)
        audits.append(item);cache[d.name]=(b,g,m);sample_arrays[d.name]=b
    pairs=[]
    for row in latest+half:
        ish=row['model']=='HalfSC';key='Ref_'+row['hybrid_source_ref_point_id' if ish else 'ref_point_id'];b,g,m=cache[key]
        kr={'Ref':1.,'SC':2.,'HalfSC':1.5}[row['model']]
        c=dict(contract,Ref_current_Arms=row['current_Arms']/kr)
        replay=evaluate_active_samples(b,g,c,row['current_Arms'],row['speed_rpm'],kr)
        want=row['MS_field_sha256'] if ish else row['field_sha256'];assert want==next(x['sha256'] for x in audits if x['source_key']==key)
        delta_current=row['current_Arms']/kr-m['current_rms_A']
        pairs.append(dict(case_id=row['case_id'],model=row['model'],MS_source_key=key,MS_source_sha256=want,
            source_current_Arms=m['current_rms_A'],target_current_Arms=row['current_Arms'],normalized_target_current_Arms=row['current_Arms']/kr,
            computed_target_minus_source_ref_A=delta_current,recorded_source_current_delta_ref_A='' if ish else row['current_delta_ref_A'],
            recorded_mapping=('exact' if row['MS_field_current_matches_target_exactly'] else 'bounded') if ish else ('exact' if row['exact_field_current_match'] else 'bounded'),
            source_speed_rpm=m['rpm'],target_speed_rpm=row['speed_rpm'],target_radial_factor=kr,
            geometry_relation='same_Ref_template_homogenized_MS_vs_discrete_Cu' if kr==1 else 'declared_similarity_transfer_from_Ref_MS',
            exact_same_geometry_and_OP=False,canonical_evaluator='mean-square-field Hybrid',
            replay_Hybrid_W=replay['hybrid_W'],saved_Hybrid_W=row['hybrid_W'],delta_Hybrid_W=replay['hybrid_W']-row['hybrid_W'],
            relative_delta_percent=100*(replay['hybrid_W']/row['hybrid_W']-1),Full_AC_excess_W=row['P2_AC_excess_W'],
            role=row.get('role','latest_loaded96'),new_independent_truth=False))
    with (a.out/'MS_Full_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    np.savez_compressed(a.out/'MS24_reconstructed_B.npz',**sample_arrays)
    result=dict(schema='raw-MS-A-to-canonical-denominator-v1',sources=audits,pair_count=len(pairs),latest192=192,HalfSC36=36,
        raw_MS_count=len(audits),A_B_numerical_status='PASS' if all(x['numerical_reproduction_status']=='PASS' for x in audits) else 'FAIL',
        Hybrid_replay_status='PASS' if max(abs(x['delta_Hybrid_W']) for x in pairs)<=1e-6 else 'FAIL',
        Hybrid_max_abs_delta_W=max(abs(x['delta_Hybrid_W']) for x in pairs),Hybrid_mean_abs_delta_W=float(np.mean([abs(x['delta_Hybrid_W']) for x in pairs])),
        Hybrid_max_abs_relative_percent=max(abs(x['relative_delta_percent']) for x in pairs),Hybrid_numerical_gate_W=1e-6,
        gate_source='same stored-denominator numerical replay gate used by baseline_audit/replay_denominator.py; not physical-accuracy gate',
        sigma_S_m=contract['sigma_S_m'],active_length_m=contract['active_length_m'],contract_sha256=sha(a.sources/'jeet_acloss_rbf/rea_hybrid/examples/e10_ref_input.json'),
        source_current_mapping_preserved=True,exact_cross_solver_bridges=0,new_solver_calls=0,model_training_calls=0,AF_refit=False)
    write_json(a.out/'MS_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)


if __name__=='__main__':main()
