"""Native Motor-CAD private-copy readback. No FEA is executed."""
from pathlib import Path
import argparse,json,hashlib,datetime,shutil,os
import psutil
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
 source=r/'source.mot';expected=json.loads((r/'input.json').read_text())['sha256']
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-10-27T22:55:00+00:00'):raise SystemExit('Deadline')
 import ansys.motorcad.core as api
 mc=None;result={'scope':'native geometry/settings readback only; no magnetic calculation','solver_runs':0,'cases':[]}
 try:
  mc=api.MotorCAD(open_new_instance=True,enable_success_variable=False)
  pid=mc.connection.pid;proc=psutil.Process(pid)
  write(r/'MC_OWNER.json',dict(pid=pid,create_time=proc.create_time(),name=proc.name(),source='new-instance API connection PID'))
  mc.set_variable('MessageDisplayState',2);mc.set_visible(False)
  for n in [6,4,8]:
   mot=r/f'probe_slot{n}.mot';shutil.copyfile(source,mot);mc.load_from_file(str(mot))
   h=(11.866-(n-1)*.35)/n
   for name,value in [('ConductorsPerSlot',n),('Copper_Height',h),('TorquePointsPerCycle',128),('TorqueNumberCycles',1),('RMSCurrent',250*6/n)]:mc.set_variable(name,value)
   mc.save_to_file(str(mot));mc.load_from_file(str(mot))
   row=dict(conductors_per_slot_requested=n,requested_height_mm=h,readback={},unavailable={})
   for key in ['ConductorsPerSlot','ConductorsPerSlot_Total','NoMushConductorsDrawn','Copper_Height','Copper_Width','Copper_CornerRadius','Insulation_Thickness','ConductorSeparation','Liner_Thickness','Slot_Depth','TorquePointsPerCycle','TorqueNumberCycles','RMSCurrent','WindingTemperature','WindingTemp','ProximityLossModel']:
    try:row['readback'][key]=mc.get_variable(key)
    except Exception as e:row['unavailable'][key]=type(e).__name__
   v=row['readback'];row['requested_turns_drawn_status']='PASS' if v.get('NoMushConductorsDrawn')==n and v.get('ConductorsPerSlot_Total')==n else 'FAIL'
   row['step_setting_status']='PASS' if v.get('TorquePointsPerCycle')==128 and v.get('TorqueNumberCycles')==1 else 'FAIL'
   row['actual_solver_step_count']='UNKNOWN_no_solve';row['automatic_geometry_fit']=False
   row['saved_model_sha256']=hashlib.sha256(mot.read_bytes()).hexdigest();result['cases'].append(row)
  result['status']='COMPLETED';result['source_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==expected
 except Exception as e:result.update(status='FAILED',error=repr(e));raise
 finally:
  if mc is not None:
   try:mc.quit()
   except Exception as e:result['quit_error']=repr(e)
  write(r/'PROBE_RESULT.json',result)
if __name__=='__main__':main()
