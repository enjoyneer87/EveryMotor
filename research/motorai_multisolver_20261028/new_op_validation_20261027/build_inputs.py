"""Derive only current/speed/time controls from preserved Ref/SC native inputs."""
from pathlib import Path
import argparse,json,math,re,hashlib,shutil
ROOT=Path(__file__).resolve().parent
FILES=['case.sif','PARAMETERS.json','MODEL.json','INPUT_CONTRACT.json','circuits.definitions','CircuitsBDF2Consistent.so','lamination_hb.dat','shaft_hb.dat','mesh/mesh.nodes','mesh/mesh.elements','mesh/mesh.boundary','mesh/mesh.header']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    frozen=json.loads((ROOT/'PREDICTIONS_FROZEN.json').read_text());records=[]
    for row in frozen['conditions']:
        source=a.base/f"ISO64_ACQ_{row['model']}_P117_N256_P2_Q01/raw";dest=a.out/row['case_id'];dest.mkdir()
        originals={f:sha(source/f) for f in FILES}
        for f in FILES:
            target=dest/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/f,target)
        pars=json.loads((dest/'PARAMETERS.json').read_text());contract=json.loads((dest/'INPUT_CONTRACT.json').read_text());model=json.loads((dest/'MODEL.json').read_text());sif=(dest/'case.sif').read_text()
        assert pars['beta_deg']==row['beta']==36 and pars['sigma_cu']==47092391.429245 and model['axial_length_m']==.15
        frequency=row['speed']*4/60;omega=-row['speed']*2*math.pi/60;ratio=row['current']/pars['Irms'];old_sif=sif
        patterns=[(r'(?m)^\$ freq = [^\r\n]+',f'$ freq = {frequency:.17g}'),(r'(?m)^\$ omega_m = [^\r\n]+',f'$ omega_m = {omega:.17g}'),(r'Timestep Sizes = [\d.eE+-]+',f'Timestep Sizes = {1/frequency/256:.17g}'),(r'Timestep Intervals = \d+','Timestep Intervals = 768')]
        for pattern,value in patterns:sif,n=re.subn(pattern,value,sif);assert n==1
        count=[0]
        def amp(match):
            old=float(match[1])
            if old!=0:
                assert abs(old-math.sqrt(2)*pars['Irms'])<1e-8;count[0]+=1
            return 'Real MATC "'+format(old*ratio,'.17g')+'*sin(2*pi*freq*tx+'
        sif=re.sub(r'Real MATC "([\d.eE+-]+)\*sin\(2\*pi\*freq\*tx\+',amp,sif);assert count[0]==36
        for component in contract['components']:
            if component['kind']=='copper':component['amplitude_A']*=ratio;assert abs(component['amplitude_A']-math.sqrt(2)*row['current'])<1e-8
        pars.update(Irms=row['current'],rpm=row['speed'],NbSteps=256,NbPeriods=3)
        model.update(anchor_current_Arms=row['current'],anchor_speed_rpm=row['speed'])
        contract.update(parent_point_id=contract['point_id'],point_id=row['case_id'],derivation='Current/speed and3period controls only; original physical geometry/material/circuit topology retained',parent_input_contract_sha256=originals['INPUT_CONTRACT.json'])
        (dest/'case.sif').write_text(sif)
        for name,d in [('PARAMETERS.json',pars),('MODEL.json',model),('INPUT_CONTRACT.json',contract)]: (dest/name).write_text(json.dumps(d,indent=2)+'\n')
        assert all(sha(source/f)==h for f,h in originals.items())
        expected_change={'case.sif','PARAMETERS.json','MODEL.json','INPUT_CONTRACT.json'};new={f:sha(dest/f) for f in FILES};assert {f for f in FILES if new[f]!=originals[f]}<=expected_change
        records.append(dict(case_id=row['case_id'],parent=f"{row['model']}_P117",parent_hashes=originals,input_hashes=new,source_unchanged=True,current_sources_changed=36,speed_ratio=row['speed']/16000,omega_over_frequency=omega/frequency,dt_s=1/frequency/256,
            condition={k:row[k] for k in ['model','current','beta','speed']},physical_geometry_material_files_unchanged=True))
    (ROOT/'INPUT_MANIFEST.json').write_text(json.dumps(dict(predictions_sha256=sha(ROOT/'PREDICTIONS_FROZEN.json'),cases=records,files_per_case=FILES,physical_deltas='none; speed/current are intended OP changes',local_paths='private mapping only'),indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),all_parent_files_unchanged=True)))
if __name__=='__main__':main()
