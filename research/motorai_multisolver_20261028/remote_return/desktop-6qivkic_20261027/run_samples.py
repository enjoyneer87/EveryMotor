"""Six preselected existing cases, sequential read-only reintegration."""
from pathlib import Path
import argparse
from field_reader import read_json, write_json, audit

SELECTION = {
    "Ref_P007": "loaded baseline; low speed 2000 rpm; 115.075 Arms; beta 0",
    "Ref_P117": "Ref high current/high frequency; 16000 rpm; 460 Arms; beta 36",
    "SC_P117": "SC scaled high-current counterpart; 16000 rpm; 920 Arms; beta 36",
    "SC_P120": "SC high frequency and beta 90 endpoint; 920 Arms",
    "Ref_P075": "Ref intermediate speed/current; 8000 rpm; beta 36",
    "Ref_P001": "0.1 Arms low-current sensitivity control; 2000 rpm; outside latest loaded192",
}


def main():
    p=argparse.ArgumentParser();p.add_argument("--path-map",type=Path,required=True);p.add_argument("--point-ids",nargs="*")
    a=p.parse_args();c=read_json(a.path_map);w=Path(c["workspace"]);repo=Path(c["source_repo"]);local=Path(c["local_root"])
    out=Path(__file__).resolve().parent
    rows=read_json(repo/"data/elmer_revision_20261018/JEET_RESULTS_240.json")["rows"]
    selected=a.point_ids or list(SELECTION)
    for pid in selected:
        assert pid in SELECTION
        row=next(r for r in rows if r["point_id"]==pid);cid=row["case_id"]
        candidates=[w/"cases"/cid, w/"outputs/jeet_fields_16000rpm_20261018/work/restored_qa"/cid,
                    w/"outputs/jeet_fields_representative_20261018/work/restored_qa"/cid,local/"restored"/cid]
        available=[d for d in candidates if (d/"raw/mesh/native.result").exists()]
        assert available, f"Raw history missing for {pid}"
        source=available[0]
        expected={k:row[k] for k in ["point_id","case_id","P2_Cu_total_W","P2_DC_transport_W","P2_AC_excess_W","P2_max_current_error_percent"]}
        expected["archive_files"]=read_json(w/"cases"/cid/"ARCHIVE_SHA256.json")["files"]
        expected["selection_reason"]=SELECTION[pid]
        (out/"sample_expected").mkdir(exist_ok=True);write_json(out/"sample_expected"/(pid+".json"),expected)
        audit(source/"raw",expected,out/"shards",local/"full_history")


if __name__=="__main__":main()
