"""Pure geometry generation/QA; never invokes meshing, solvers or training.

Each process loads a private pinned ipmfea source. Mutable generator constants
are changed only in this process and restored; source files are never edited.
"""
from pathlib import Path
import argparse,csv,json,sys,hashlib,subprocess,math
ROOT=Path(__file__).resolve().parent
def area(poly):return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()
 assert commit=='a46e5d8bc0e5621c455915e627279356c1a1269e'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=a.source,text=True).strip()
 sys.path.insert(0,str(a.source/'src'));from ipmfea.geometry import generator as g
 a.out.mkdir(parents=True,exist_ok=False)
 defaults=(g.N_COND,g.COND_H,g.LADDER_PITCH)
 results=[]
 for r in csv.DictReader((ROOT/'configuration_queue.csv').open(encoding='utf-8')):
  n=int(r['conductors_per_slot']);h=float(r['bare_height_mm']);g.N_COND=n;g.COND_H=h;g.LADDER_PITCH=h+.35
  p={k:float(r[k]) for k in ['RotorDia','Airgap','Bs0','Bs1','L1_PoleVAngle','L2_PoleVAngle']};p.update(VAnchor='magnet-tip',MBSplit=1)
  out=a.out/r['configuration_id'];out.mkdir()
  try:
   v=g.build(p,str(out));polys=v['conductors'];assert len(polys)==6*n
   ar=[area(poly) for poly in polys];expected=float(r['bare_width_mm'])*h-2*.197**2
   assert max(abs(x-expected) for x in ar)<1e-8
   rb=p['RotorDia']/2+p['Airgap'];margin=min(rb+g.SLOT_DEPTH-math.hypot(*pt) for poly in polys for pt in poly)
   assert margin>0 and h>2*g.COND_CH and float(r['bare_width_mm'])>2*g.COND_CH
   assert abs((n-1)*g.LADDER_PITCH+h-float(r['stack_envelope_mm']))<1e-10
   baseline='NOT_APPLICABLE'
   if n==6:
    g.N_COND,g.COND_H,g.LADDER_PITCH=defaults
    legacy=a.out/(r['configuration_id']+'_baseline_parity');legacy.mkdir();g.build(p,str(legacy))
    assert all(f.read_bytes()==(legacy/f.name).read_bytes() for f in out.glob('*.geo'))
    baseline='PASS_BYTE_IDENTICAL_GEO'
   files=[dict(path=f'{r["configuration_id"]}/{x.name}',bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in out.glob('*.geo')]
   results.append(dict(configuration_id=r['configuration_id'],status='PASS',scope='analytic conductor count/area/envelope plus generator; not mesh or physical validation',
      sector_conductors=len(polys),sector_copper_area_mm2=sum(ar),min_outer_slot_clearance_mm=margin,six_conductor_parity=baseline,generator_warnings=v.get('warn',[]),files=files))
  except Exception as e:results.append(dict(configuration_id=r['configuration_id'],status='FAIL',reason=repr(e)))
 g.N_COND,g.COND_H,g.LADDER_PITCH=defaults
 report=dict(requested=816,generated_pass=sum(x['status']=='PASS' for x in results),failed=sum(x['status']=='FAIL' for x in results),
  source_commit=commit,mesh_quality='UNKNOWN',manufacturing_qualification='UNKNOWN',transient_cases_solved=0,MGN_models_trained=0,
  packing_policy='fixed stack envelope; copper area changes with insulation gaps',results=results)
 (ROOT/'GEOMETRY_QA.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='results'}))
if __name__=='__main__':main()
