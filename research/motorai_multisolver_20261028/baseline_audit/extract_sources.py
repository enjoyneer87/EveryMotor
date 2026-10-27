"""Read pinned local Git objects; writes only this audit directory."""
import subprocess, pathlib, hashlib, json
ROOT=pathlib.Path(__file__).resolve().parent
REPO=pathlib.Path('D:/KangDH/JEET-repro-appendix-review-22add80')
REV='0034d7c'
paths=[
'handoff/20261019_AF_loaded96/AF_POINTS_192.json',
'handoff/20261019_AF_loaded96/AF_model_Ref_loaded96.json',
'handoff/20261019_AF_loaded96/AF_model_SC_loaded96.json',
'handoff/20261019_AF_loaded96/MODEL_REPORT.json',
'handoff/20261019_AF_loaded96/SAMPLING_PLAN.json',
'handoff/20261019_AF_loaded96/REGRESSION_INPUTS.json',
'handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.csv',
'handoff/20261021_appendix_halfsc/halfsc/HALFSC_POINTS_36.json',
'handoff/20261021_appendix_halfsc/halfsc/FINAL_AUDIT.json',
'handoff/20261020_halfsc_direct_af/PLAN.json',
'handoff/20261020_halfsc_direct_af/COMPARISON.json',
'scripts/figure_interview_20261019/adopt_loaded96.py',
'scripts/halfsc_validation_20261021/build_validation.py',
'jeet_acloss_rbf/SeparableRbfModel.py',
'jeet_acloss_rbf/RbfModelBuilder.py',
'jeet_acloss_rbf/RbfModel3D.py',
'jeet_acloss_rbf/rea_hybrid/hybrid_kernel.py',
'jeet_acloss_rbf/rea_hybrid/e10_sampling.py',
'jeet_acloss_rbf/rea_hybrid/postprocess_elmer.py',
'jeet_acloss_rbf/rea_hybrid/examples/e10_ref_input.json',
'handoff/20261019_rea_hybrid_remote_results/INPUT_INDEX_60.json',
'handoff/20261019_rea_hybrid_remote_results/evidence/Ref_Ref_P097/RECEIPT.json',
]
base_rows=json.loads(subprocess.check_output(['git','show',f'{REV}:handoff/20261019_AF_loaded96/AF_POINTS_192.json'],cwd=REPO))
for point in sorted({r['ref_point_id'] for r in base_rows}):
 paths.append('handoff/20261019_rea_hybrid_remote_results/samples/Ref_'+point+'.npz')
manifest=[]
commit=subprocess.check_output(['git','rev-parse',REV],cwd=REPO,text=True).strip()
for path in paths:
 b=subprocess.check_output(['git','show',f'{REV}:{path}'],cwd=REPO)
 out=ROOT/'sources'/path
 out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
 manifest.append(dict(repo=str(REPO),commit=commit,git_path=path,local_path=str(out),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(commit=commit,files=len(manifest))))
