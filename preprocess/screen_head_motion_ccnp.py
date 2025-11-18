import argparse
import csv
import re
import sys
from pathlib import Path

# example:
# python screen_head_motion_ccnp.py --fmriprep-dir /ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/datasets/CCNP/fmriprep --out /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/CCNP/table/rest_fd_summary.csv

def find_subject_id(p: Path) -> str:
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return ""


def summarize_fd(tsv_path: Path) -> tuple[str, str]:
    total = 0.0
    valid = 0
    low = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "framewise_displacement" not in reader.fieldnames:
            return "NA", "NA"
        for row in reader:
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
        return "NA", "NA"
    mean = total / valid
    ratio = low / valid
    return f"{mean:.6f}", f"{ratio:.6f}"


def load_age_map(age_csv: Path) -> dict[str, float]:
    m = {}
    with age_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        subj_field = None
        age_field = None
        for col in fields:
            lc = col.lower()
            if subj_field is None and ("subject" in lc or "participant" in lc or lc in ("subid", "subject_id", "participant_id")):
                subj_field = col
            if age_field is None and "age" in lc:
                age_field = col
        if subj_field is None or age_field is None:
            return m
        for row in reader:
            sid = str(row.get(subj_field, "")).strip()
            a = row.get(age_field)
            if not sid or a is None:
                continue
            try:
                age = float(str(a).strip())
            except ValueError:
                continue
            m[sid] = age
    return m


def collect_subject_runs(fmriprep_dir: Path, task: str) -> dict[str, list[Path]]:
    d = {}
    for tsv in fmriprep_dir.rglob(f"*task-{task}*desc-confounds_timeseries.tsv"):
        if "func" not in tsv.parts:
            continue
        sid = find_subject_id(tsv)
        lst = d.setdefault(sid, [])
        lst.append(tsv)
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-dir", "--fmriprep_dir", required=True, dest="fmriprep_dir")
    parser.add_argument("--age-csv", "--age_csv", dest="age_csv")
    parser.add_argument("--min-age", "--min_age", type=float, default=6.0, dest="min_age")
    parser.add_argument("--task", default="rest")
    args = parser.parse_args()
    fdir = Path(args.fmriprep_dir)
    if not fdir.exists():
        print(f"Input directory not found: {fdir}", file=sys.stderr)
        return
    age_map = {}
    if args.age_csv:
        ap = Path(args.age_csv)
        if ap.exists():
            age_map = load_age_map(ap)
    runs_map = collect_subject_runs(fdir, args.task)
    excluded = 0
    eligible = 0
    for sid, files in runs_map.items():
        if len(files) == 0:
            excluded += 1
            continue
        if len(files) == 1:
            excluded += 1
            continue
        age = age_map.get(sid)
        if age is not None and age < args.min_age:
            excluded += 1
            continue
        bad = False
        for tsv in files:
            mean_fd, low_ratio = summarize_fd(tsv)
            if mean_fd == "NA" or low_ratio == "NA":
                bad = True
                break
            try:
                if float(mean_fd) > 0.5 or float(low_ratio) < 0.4:
                    bad = True
                    break
            except Exception:
                bad = True
                break
        if bad:
            excluded += 1
        else:
            eligible += 1
    print(f"excluded={excluded}")
    print(f"eligible={eligible}")


if __name__ == "__main__":
    main()