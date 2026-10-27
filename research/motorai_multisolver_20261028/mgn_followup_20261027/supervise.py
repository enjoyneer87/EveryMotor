"""Owned-process deadline guardian, suitable for a one-shot Windows task."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,time
import psutil
DEADLINE=datetime.datetime.fromisoformat('2026-10-27T23:00:00+00:00').timestamp()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--state',type=Path,required=True);a=ap.parse_args()
    a.state.mkdir(exist_ok=False);start=time.monotonic();reason=None
    if time.time()+3600>=DEADLINE:raise RuntimeError('Insufficient deadline margin')
    command=[__import__('sys').executable,str(Path(__file__).with_name('train_curl.py')),'--data',str(a.data),'--out',str(a.out)]
    with (a.state/'stdout.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);identity=psutil.Process(p.pid).create_time()
        (a.state/'OWNER.json').write_text(json.dumps(dict(supervisor_PID=os.getpid(),child_PID=p.pid,child_create_time=identity,deadline_UTC='2026-10-27T23:00:00Z',retry=0),indent=2))
        while p.poll() is None:
            if time.time()>=DEADLINE:reason='08:00 KST cutoff'
            elif time.monotonic()-start>11000:reason='aggregate supervisor bound'
            elif (a.state/'STOP').exists():reason='owned STOP'
            if reason:
                if a.out.exists():(a.out/'STOP').write_text(reason)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    owned=psutil.Process(p.pid)
                    if owned.create_time()==identity:
                        for child in owned.children(recursive=True):child.terminate()
                        owned.terminate()
                    p.wait(timeout=30)
                break
            time.sleep(5)
    result=dict(returncode=p.returncode,stop_reason=reason,elapsed_seconds=time.monotonic()-start,deadline_UTC='2026-10-27T23:00:00Z',completed=(a.out/'TRAINING_COMPLETE.json').exists(),retry=0)
    (a.state/'EXECUTION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
