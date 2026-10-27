"""Compare saved coarse/fine tagged domains without coordinate ID reconstruction."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np


def describe(mesh):
    nodes=np.loadtxt(mesh/'mesh.nodes');e=np.loadtxt(mesh/'mesh.elements',dtype=np.int64)
    assert np.all(e[:,2]==303)
    lookup={int(n):i for i,n in enumerate(nodes[:,0])};tri=np.array([[lookup[int(n)] for n in r] for r in e[:,3:]])
    xy=nodes[:,2:4];v=xy[tri];u=v[:,1]-v[:,0];w=v[:,2]-v[:,0];area=abs(u[:,0]*w[:,1]-u[:,1]*w[:,0])/2
    rows={}
    for tag in np.unique(e[:,1]):
        m=e[:,1]==tag;points=v[m].reshape(-1,2);a=area[m]
        rows[int(tag)]=dict(elements=int(m.sum()),area_m2=float(a.sum()),centroid_m=(v[m].mean(1)*a[:,None]).sum(0)/a.sum(),bbox_m=np.r_[points.min(0),points.max(0)])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in mesh.iterdir() if p.name in ['mesh.nodes','mesh.elements','mesh.boundary','mesh.header']}
    return rows,hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--coarse',type=Path,required=True);p.add_argument('--fine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    coarse,ch=describe(a.coarse);fine,fh=describe(a.fine);assert set(coarse)==set(fine)
    rows=[]
    for tag in coarse:
        c=coarse[tag];f=fine[tag]
        rows.append(dict(physical=tag,coarse_elements=c['elements'],fine_elements=f['elements'],coarse_area_m2=c['area_m2'],fine_area_m2=f['area_m2'],
            area_difference_percent=100*(f['area_m2']-c['area_m2'])/c['area_m2'],centroid_distance_m=float(np.linalg.norm(f['centroid_m']-c['centroid_m'])),
            max_bbox_difference_m=float(abs(f['bbox_m']-c['bbox_m']).max())))
    copper=[r for r in rows if 2301<=r['physical']<=2336]
    result=dict(same_physical_tags=True,coarse_hashes=ch,fine_hashes=fh,domains=rows,
        copper_max_area_difference_percent=max(abs(r['area_difference_percent']) for r in copper),
        copper_max_centroid_distance_m=max(r['centroid_distance_m'] for r in copper),copper_max_bbox_difference_m=max(r['max_bbox_difference_m'] for r in copper),
        status='QUANTIFIED_NOT_EXACT_CONTINUUM_GEOMETRY_PROOF',
        interpretation='Pre-existing RefFine mesh retains physical domains and copied material/circuit/model inputs. Discrete geometry differences are reported, not concealed. No coordinate nearest-neighbor ID reconstruction.')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['domains','coarse_hashes','fine_hashes']}))


if __name__=='__main__':main()
