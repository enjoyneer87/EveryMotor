"""Repo-local CLI for the generalized read-only native integrator."""
from pathlib import Path
import sys,argparse,json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'remote_return/desktop-6qivkic_20261027'))
from audit_convergence import audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    result=audit(a.raw,a.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ['area_m2_by_bar','input_sha256']},indent=2))
