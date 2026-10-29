"""Synthetic-only native geometry and mask-admission tests; no remote/data/model I/O."""
import copy
import unittest
import numpy as np
from audit_train328_geometry import inspect_training_case, validate_request, TRAIN_STEPS, SELECTION_SHA
from bind_train328_masks import validate_graph, phase_files, hash_array


class Names:
    def asstr(self):
        return np.array(['a1','a2','stator'])


class Metadata(dict):
    def __init__(self, values):
        super().__init__(values); self.read_keys=[]
    def __getitem__(self,key):
        if not key.startswith(('mesh/','regions/')):
            raise AssertionError('field access forbidden')
        self.read_keys.append(key)
        return super().__getitem__(key)


def fixture():
    c=np.sqrt(.5)
    xy=np.array([[100.,0.],[110.,0.],[100*c,100*c],[110*c,110*c],
                 [90.,0.],[90.,10.],[95.,0.],[80.,0.],[80.,5.],[85.,0.]])
    ids=np.arange(10,110,10,dtype=np.int64)
    tri=np.array([[10,20,30],[20,40,30],[50,60,70],[80,90,100]])
    return Metadata({'mesh/node_id':ids,'mesh/tri_index':np.array([100,200,300,900]),
        **{f'mesh/node_{k+1}':tri[:,k] for k in range(3)},'mesh/reg_code':np.array([1,1,3,2]),
        'regions/reg_code':np.array([1,2,3]),'regions/name':Names(),
        'mesh/moving_reg_codes':np.array([2]),'mesh/moving_node_id':ids[7:],
        'mesh/moving_node_indices':np.array([7,8,9]),'mesh/node_x_mm':xy[:,0],'mesh/node_y_mm':xy[:,1],
        'mesh/node_x_mm_by_step_moving':np.tile(xy[7:,0],(45,1)),
        'mesh/node_y_mm_by_step_moving':np.tile(xy[7:,1],(45,1))})


REC={'case_index':40,'geometry_group':'synthetic40','source_sha256':'synthetic','stack_length_m':.15}


def operator_and_graph():
    captured=[]
    result=inspect_training_case(fixture(),REC,lambda:None,lambda op,extra:captured.append((op,extra)))
    op,extra=captured[0]
    operator={key:getattr(op,key) for key in ('admitted_element_ids','a1_element_ids','a1_triangle_node_ids')}
    operator.update({key:getattr(op.band,key) for key in ('region_codes','area_m2','radius_m','cos_theta','sin_theta')})
    operator.update(extra)
    # Graph element rows are deliberately permuted, unlike the sorted operator.
    permutation=np.array([2,0,1])
    ids=op.admitted_element_ids[permutation]
    centroids=np.zeros((3,2))
    centroids[:2]=np.column_stack([op.band.radius_m*op.band.cos_theta,op.band.radius_m*op.band.sin_theta])/.1
    arrays={'native_element_id':ids,'native_tri_node_ids':extra['admitted_triangle_node_ids'][permutation],
            'native_reg_code':extra['admitted_region_codes'][permutation],
            'weights':extra['admitted_area_m2_by_train_step'][0,permutation].astype(np.float32),
            'node':centroids[permutation].astype(np.float32),'step':np.array(0,dtype=np.int64)}
    return operator,arrays,result


class TrainingGeometryTests(unittest.TestCase):
    def test_float32_hdf_coordinates_follow_strict_float64_geometry(self):
        h=fixture()
        for key in ('mesh/node_x_mm','mesh/node_y_mm','mesh/node_x_mm_by_step_moving','mesh/node_y_mm_by_step_moving'):
            h[key]=h[key].astype(np.float32)
        captured=[]
        inspect_training_case(h,REC,lambda:None,lambda op,extra:captured.append((op,extra)))
        op,extra=captured[0]
        self.assertTrue(np.array_equal(extra['admitted_area_m2_by_train_step'][:,:2],np.broadcast_to(op.band.area_m2,(9,2))))
        self.assertEqual(extra['admitted_area_m2_by_train_step'].dtype,np.dtype('float64'))

    def test_geometry_all45_proof_and_phase9_full_area(self):
        h=fixture(); captured=[]
        row=inspect_training_case(h,REC,lambda:None,lambda op,extra:captured.append((op,extra)))
        self.assertEqual(row['steps_checked'],45)
        self.assertEqual(row['a1_stationarity_steps_checked'],45)
        self.assertEqual(row['train_steps_checked'],TRAIN_STEPS)
        self.assertEqual(row['all_admitted_area_steps_checked'],9)
        self.assertEqual(row['field_array_reads'],0)
        self.assertEqual(captured[0][1]['admitted_area_m2_by_train_step'].shape,(9,3))
        self.assertTrue(all(key.startswith(('mesh/','regions/')) for key in h.read_keys))

    def test_a1_candidate_motion_at_unused_phase_still_rejected(self):
        h=fixture(); positions=np.array([1,7,8,9])
        h['mesh/moving_node_indices']=positions
        h['mesh/moving_node_id']=h['mesh/node_id'][positions]
        h['mesh/node_x_mm_by_step_moving']=np.tile(h['mesh/node_x_mm'][positions],(45,1))
        h['mesh/node_y_mm_by_step_moving']=np.tile(h['mesh/node_y_mm'][positions],(45,1))
        row=inspect_training_case(h,REC,lambda:None)
        self.assertEqual(row['a1_candidate_moving_nodes'],1)
        h['mesh/node_x_mm_by_step_moving'][44,0]+=1e-9
        with self.assertRaises(ValueError): inspect_training_case(h,REC,lambda:None)

    def test_degenerate_admitted_geometry_rejected(self):
        h=fixture();h['mesh/node_3'][2]=h['mesh/node_2'][2]
        with self.assertRaises(ValueError): inspect_training_case(h,REC,lambda:None)

    def test_stop_callback(self):
        def stop(): raise InterruptedError('test STOP')
        with self.assertRaises(InterruptedError): inspect_training_case(fixture(),REC,stop)

    def test_permuted_native_mask_and_q_match_graph_weights(self):
        op,arrays,_=operator_and_graph();row=validate_graph(arrays,op,0)
        mask=np.array([False,True,True],dtype=np.bool_)
        self.assertEqual(row['a1_mask_sha256'],hash_array(mask))
        self.assertEqual(row['native_element_id_sha256'],hash_array(arrays['native_element_id']))
        self.assertEqual(row['a1_count'],2)
        values=arrays['weights'].astype(float)
        self.assertEqual(row['q'],values[mask].sum()/values.sum())
        self.assertEqual(row['effective_a1_allocation_alpha05'],.5+.5*row['q'])
        self.assertEqual(row['native_area_validation_scope'],'ALL_ADMITTED')

    def test_nona1_area_permutation_or_corruption_rejected(self):
        op,arrays,_=operator_and_graph();arrays['weights'][0]*=1.001
        with self.assertRaisesRegex(ValueError,'all admitted native area'):validate_graph(arrays,op,0)

    def test_native_id_incidence_region_corruption_rejected(self):
        op,original,_=operator_and_graph()
        for change in ('id','incidence','region'):
            arrays=copy.deepcopy(original)
            if change=='id': arrays['native_element_id'][0]=arrays['native_element_id'][1]
            elif change=='incidence': arrays['native_tri_node_ids'][0,0]+=1
            else: arrays['native_reg_code'][0]=1
            with self.subTest(change=change),self.assertRaises(ValueError):validate_graph(arrays,op,0)

    def test_nonfinite_or_bad_weight_rejected(self):
        op,original,_=operator_and_graph()
        for bad in (np.nan,np.inf,0.,-1.):
            arrays=copy.deepcopy(original);arrays['weights'][0]=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_graph(arrays,op,0)

    def test_dev_reuse_scope_explicit_and_a1_still_checked(self):
        op,arrays,_=operator_and_graph()
        row=validate_graph(arrays,op,0,False)
        self.assertEqual(row['native_area_validation_scope'],'A1_REUSED_DEV32_ADMISSION')
        arrays['weights'][1]*=1.001
        with self.assertRaises(ValueError):validate_graph(arrays,op,0,False)

    def test_exact_phase_coverage_and_native_step(self):
        record={'case_index':40,'split':'train','files':[{'name':f'case0040_step{s:02d}.npz'} for s in range(45)]}
        self.assertEqual([row['step'] for row in phase_files(record)],TRAIN_STEPS)
        record['files'].pop()
        with self.assertRaises(ValueError):phase_files(record)
        record={'case_index':40,'split':'development','files':[{'name':f'case0040_step{s:02d}.npz'} for s in TRAIN_STEPS]}
        self.assertEqual(len(phase_files(record)),9)
        record['files'].append(dict(record['files'][0]))
        with self.assertRaises(ValueError):phase_files(record)
        op,arrays,_=operator_and_graph();arrays['step']=np.array(5)
        with self.assertRaises(ValueError):validate_graph(arrays,op,0)

    def test_request_bound_to_all328_frozen_train_records(self):
        records=[dict(case_index=i,geometry_group=f'g{i}',source_key=f'case{i}.h5',source_sha256=f'hash{i}',
                      split='train',original_split='train',native_input={'Stator_Lam_Length':150}) for i in range(328)]
        request={'selection_sha256':SELECTION_SHA,'steps':TRAIN_STEPS,'cpu_wall_seconds_max':1800,
                 'records':[{**{k:v for k,v in row.items() if k!='native_input'},'stack_length_m':.15} for row in records]}
        self.assertEqual(len(validate_request(request,{'records':records})),328)
        for field,value in [('source_sha256','wrong'),('split','development'),('stack_length_m',.3)]:
            changed=copy.deepcopy(request);changed['records'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_request(changed,{'records':records})
        changed=copy.deepcopy(request);changed['records'][-1]=changed['records'][0]
        with self.assertRaises(ValueError):validate_request(changed,{'records':records})


if __name__=='__main__':unittest.main()
