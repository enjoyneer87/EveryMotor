"""Scientific contract checks; run on the isolated training venv."""
from pathlib import Path
import argparse, json, unittest
import numpy as np
import torch
from model import MeshGraphNet, edge_features
from residual import feature_matrix, fit, predict, SCALE

DATA = None


class Contracts(unittest.TestCase):
    def test_graph_permutation_equivariance(self):
        torch.manual_seed(137)
        n=9
        static=torch.randn(n,4)
        edges=torch.tensor([[0,1,1,2,3,4,4,5,6,7,7,8],[1,0,2,1,4,3,5,4,7,6,8,7]])
        net=MeshGraphNet().eval()
        condition=torch.tensor([[.25,.4],[1.,.6]])
        phase=torch.tensor([.1,.2])
        p=torch.randperm(n);inverse=torch.argsort(p)
        with torch.no_grad():
            y,h=net(static,edges,edge_features(static,edges),condition,phase)
            ep=inverse[edges]
            yp,hp=net(static[p],ep,edge_features(static[p],ep),condition,phase)
        torch.testing.assert_close(y[:,p],yp,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(h[:,p],hp,rtol=2e-5,atol=2e-5)

    def test_original_triangle_adjacency_and_projection(self):
        with np.load(DATA/'MS_graph_data.npz') as d:
            tri=d['connectivity_local'];edges=d['edge_index']
            expected=set()
            for a,b,c in tri:
                expected.update([(a,b),(b,a),(a,c),(c,a),(b,c),(c,b)])
            self.assertEqual(expected,set(map(tuple,edges.T)))
            self.assertEqual(edges.shape,(2,7124))
            self.assertEqual(len(d['node_ids']),1377)
            self.assertEqual(len(set(d['node_ids'])),1377)
            self.assertEqual(d['B_node_area_projected_T'].shape,(24,128,1377,2))
            self.assertTrue(np.isfinite(d['B_node_area_projected_T']).all())
            self.assertTrue((d['element_area_m2']>0).all())
            self.assertAlmostEqual(float(d['node_area_weights'].sum()),1)

    def test_budget_and_protected_sources(self):
        train=json.loads((DATA/'calibration63.json').read_text())
        infer=json.loads((DATA/'inference228.json').read_text())
        split=json.loads((DATA/'SPLIT_MANIFEST.json').read_text())
        ids=set(r['case_id'] for r in train)
        self.assertEqual(len(ids),63)
        self.assertEqual(sum(r['model']=='Ref' for r in train),36)
        self.assertEqual(sum(r['model']=='SC' for r in train),27)
        self.assertEqual(ids,set(split['calibration_case_ids']))
        for name,count in [('Ref_excluded60',60),('SC_strict57',57),('SC_excluded69',69),('HalfSC_predeclared32',32),('HalfSC_exploratory4',4)]:
            values=split['groups'][name]
            self.assertEqual(len(values),count)
            self.assertFalse(ids.intersection(values))
        self.assertFalse(set(split['groups']['HalfSC_predeclared32']).intersection(split['groups']['HalfSC_exploratory4']))
        self.assertTrue(all('truth' not in r for r in infer))
        self.assertTrue(all(not any(x in r['case_id'] for x in ['R1_0199','R1_0076']) for r in infer))
        with np.load(DATA/'MS_graph_data.npz') as d:
            self.assertTrue(all(str(s).startswith('Ref_Ref_') for s in d['source_keys']))
        for group in set(r['source_key'] for r in train):
            inner={r['source_key'] for r in train if r['source_key']!=group}
            self.assertNotIn(group,inner)

    def test_fit_preprocessing_and_saved_head(self):
        rng=np.random.default_rng(19)
        x=rng.normal(size=(12,4));target=rng.normal(size=12)
        model=fit(x,target,1.,.001)
        np.testing.assert_allclose(model['mean'],x.mean(0))
        a=predict(model,x)
        # Query outliers must not alter preprocessing or predictions of earlier queries.
        b=predict(model,np.vstack([x,np.ones((1,4))*1000]))
        np.testing.assert_allclose(a,b[:len(x)],atol=1e-12)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(SCALE,{'Ref':1.,'SC':2.,'HalfSC':1.5})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();DATA=a.data
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    a.report.write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
