"""Package audit bindings; excludes this manifest's own hash."""
from pathlib import Path
import subprocess,hashlib,json,datetime
R=Path(__file__).resolve().parent
man=Path('C:/Users/user/.codex/worktrees/jeet-r7-final-integration/JEET-2024_rev1/manuscript/JEET_KDH_10p_EN_revision_20261017.tex')
repo=man.parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((R/'source_manifest.json').read_text())
for a in source:assert sha(Path(a['local_path']))==a['sha256']
for p in ['replay_baseline.py','replay_denominator.py']:
 subprocess.run(['python',str(R/p)],check=True,stdout=subprocess.DEVNULL)
out=dict(schema='jeet-calibrated-baseline-audit-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 canonical_manuscript=dict(path=str(man),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),sha256=sha(man)),
 reproduction_code_repository='enjoyneer87/JEET-repro',source_object_commit=source[0]['commit'],
 sources=source,evidence_levels=dict(independent_frozen_coefficient_replay=True,independent_scalar_metric_replay=True,
 canonical_denominator_on_saved_scalar_B_replay=True,raw_MS_A_to_B_recomputed=False,raw_Full_FEA_loss_reintegrated=False,
 new_FEA_runs=0,canonical_AF_refit=False,manuscript_modified=False),
 remaining_inputs=['24 Ref raw MS A histories and matching acquisition manifests/receipts for curl re-extraction',
 '192 Ref/SC saved Full-FEA Joule/DC/current integration sources and raw-audit receipts',
 '32 HalfSC preserved raw/result/receipt/manifest sources for fresh readback verification'],
 artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.iterdir()) if p.is_file() and p.name!='baseline_manifest.json'])
(R/'baseline_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(source_files=len(source),artifacts=len(out['artifacts']),manuscript_sha256=sha(man),status='PASS'),indent=2))
