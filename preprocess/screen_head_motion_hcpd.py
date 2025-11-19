import argparse
import csv
import re
import sys
from pathlib import Path

# example:
# python screen_head_motion_hcpd.py --fmriprep-dir /ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/datasets/HCPD/fmriprep --out /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/table/rest_fd_summary.csv

def find_subject_id(p: Path) -> str:
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return ""


def summarize_fd(tsv_path: Path) -> tuple[int, str, int, str]:
    frame_count = 0
    total = 0.0
    valid = 0
    low = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "framewise_displacement" not in reader.fieldnames:
            return 0, "NA", 0, "NA"
        for row in reader:
            frame_count += 1
            v = row.get("framewise_displacement")
            if v is None:
                continue
            s = v.strip()
            if not s or s.lower() == "n/a":
                continue
            try:
                x = float(s)
            except ValueError:
                continue
            total += x
            valid += 1
            if x < 0.2:
                low += 1
    if valid == 0:
        return frame_count, "NA", 0, "NA"
    return frame_count, f"{total / valid:.6f}", valid, f"{low / valid:.6f}"


def infer_run_idx(name: str) -> int | None:
    m = re.search(r"run-([12])", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"task-REST([12])", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def collect_subject_runs(fmriprep_dir: Path) -> dict[str, dict[int, Path]]:
    d: dict[str, dict[int, Path]] = {}
    for tsv in fmriprep_dir.rglob("*desc-confounds_timeseries.tsv"):
        if "func" not in tsv.parts:
            continue
        if not re.search(r"task-REST", tsv.name, flags=re.IGNORECASE):
            continue
        sid = find_subject_id(tsv)
        idx = infer_run_idx(tsv.name)
        if idx is None or idx not in (1, 2):
            continue
        m = d.setdefault(sid, {})
        m[idx] = tsv
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-dir", "--fmriprep_dir", required=True, dest="fmriprep_dir")
    parser.add_argument("--out", "--out_csv", default="rest_fd_summary.csv", dest="out")
    args = parser.parse_args()
    fdir = Path(args.fmriprep_dir)
    if not fdir.exists():
        print(f"Input directory not found: {fdir}", file=sys.stderr)
        return
    runs_map = collect_subject_runs(fdir)
    rows = []
    excluded = 0
    eligible = 0
    for sid, run_files in runs_map.items():
        has_both = (1 in run_files) and (2 in run_files)
        if not has_both:
            excluded += 1
            continue
        r1_fc, r1_fd, r1_valid_cnt, r1_low = summarize_fd(run_files[1])
        r2_fc, r2_fd, r2_valid_cnt, r2_low = summarize_fd(run_files[2])
        r1_valid = "1" if (r1_fd != "NA" and r1_low != "NA" and r1_fc == 180 and float(r1_fd) <= 0.5 and float(r1_low) > 0.4) else "0"
        r2_valid = "1" if (r2_fd != "NA" and r2_low != "NA" and r2_fc == 180 and float(r2_fd) <= 0.5 and float(r2_low) > 0.4) else "0"
        valid_num = (1 if r1_valid == "1" else 0) + (1 if r2_valid == "1" else 0)
        valid_subject = "1" if valid_num >= 2 else "0"
        bad_motion = False
        for fd, low in ((r1_fd, r1_low), (r2_fd, r2_low)):
            if fd == "NA" or low == "NA":
                bad_motion = True
                break
            if float(fd) > 0.5 or float(low) < 0.4:
                bad_motion = True
                break
        if bad_motion:
            excluded += 1
        else:
            eligible += 1
        rows.append(
            {
                "subid": sid,
                "rest1_frame": str(r1_fc),
                "rest1_fd": r1_fd,
                "rest1_valid": r1_valid,
                "rest2_frame": str(r2_fc),
                "rest2_fd": r2_fd,
                "rest2_valid": r2_valid,
                "valid_num": str(valid_num),
                "valid_subject": valid_subject,
            }
        )
    rows.sort(key=lambda r: r["subid"])
    headers = [
        "subid",
        "rest1_frame", "rest1_fd", "rest1_valid",
        "rest2_frame", "rest2_fd", "rest2_valid",
        "valid_num", "valid_subject",
    ]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"excluded={excluded}")
    print(f"eligible={eligible}")


if __name__ == "__main__":
    main()