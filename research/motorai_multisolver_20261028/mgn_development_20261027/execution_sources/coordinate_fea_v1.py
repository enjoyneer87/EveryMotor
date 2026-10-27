"""Owned case budget and duplicate-prevention monitor, private path config only."""
from pathlib import Path
import argparse,json,os,time


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def live(pid,ticks):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[19]==str(ticks)
    except FileNotFoundError:return False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();c=read(a.config)
    root=Path(c['coordinator']);root.mkdir(exist_ok=False);roots=[Path(x) for x in c['roots']]
    first=roots[0];owner=read(first/'N256_P3/OWNER.json');start=time.monotonic();finished_first=False
    write(root/'OWNER.json',dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()[19]))
    while True:
        if not finished_first and not live(owner['pid'],owner['start_ticks']):
            (first/'STOP').write_text('N256 native process exited; prevent duplicate planned variants. Other two variants have separate owned roots.\n')
            finished_first=True
        completed=0;elapsed=0;states=[]
        for r in roots:
            p=r/'progress.json'
            if not p.exists():states.append('not_started');continue
            try:s=read(p)
            except json.JSONDecodeError:continue
            if s.get('phase')=='completed':completed+=1;elapsed+=s['total_wall_seconds'];states.append('completed')
            else:
                elapsed+=s['elapsed_seconds']+max(0,time.time()-p.stat().st_mtime);states.append(s['case'])
        reason=None
        if (root/'STOP').exists():reason='coordinator_STOP'
        elif elapsed>16200:reason='aggregate_case_elapsed_limit'
        elif time.monotonic()-start>10000:reason='coordinator_wall_limit'
        if reason:
            for r in roots:
                if r.exists():(r/'STOP').write_text(reason+'\n')
        write(root/'status.json',dict(states=states,completed=completed,aggregate_case_elapsed_seconds=elapsed,stop_reason=reason,
            duplicate_prevention_applied=finished_first,wall_seconds=time.monotonic()-start))
        if completed==3:break
        time.sleep(1)
    print(json.dumps(read(root/'status.json')),flush=True)


if __name__=='__main__':main()
