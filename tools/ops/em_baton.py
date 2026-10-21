"""em_baton.py -- read or set the GPU baton (results/logs/hb_control.json).

session_coordination.md: only the baton owner launches GPU work on this track.
    python tools/ops/em_baton.py                       # print (used by em_status.ps1)
    python tools/ops/em_baton.py set --owner "..." --status running --experiment "..." [--next "..."]
"""
import argparse
import datetime
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
PATH = "results/logs/hb_control.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "set"])
    ap.add_argument("--owner")
    ap.add_argument("--status")
    ap.add_argument("--experiment")
    ap.add_argument("--next")
    a = ap.parse_args()
    hb = json.load(open(PATH, encoding="utf-8"))
    if a.cmd == "set":
        for key, val in (("owner", a.owner), ("status_hint", a.status),
                         ("experiment", a.experiment), ("next_in_queue", a.next)):
            if val is not None:
                hb[key] = val
        hb["updated"] = datetime.datetime.now().astimezone().isoformat(timespec="minutes")
        with open(PATH, "w", encoding="utf-8") as fh:
            json.dump(hb, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print(f"baton  {hb.get('status_hint')} | owner: {str(hb.get('owner'))[:60]} | updated {hb.get('updated')}")
    print(f"       exp:  {str(hb.get('experiment'))[:140]}")
    if a.cmd == "show" and hb.get("next_in_queue"):
        print(f"       next: {str(hb.get('next_in_queue'))[:140]}")


if __name__ == "__main__":
    main()
