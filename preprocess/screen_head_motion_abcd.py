import argparse
import csv
import re
import sys
from pathlib import Path

# command:
# python screen_head_motion_abcd.py --fmriprep-dir /ibmgpfs/cuizaixu_lab/congjing/WM_prediction/ABCD/data/bids --out /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/ABCD/table/rest_fd_summary.csv --debug --log /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/log/preprocess/screen_head_motion_abcd.log

def find_subject_id(p: Path) -> str:
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return ""


LOG_FH = None

def log(msg: str) -> None:
    global LOG_FH
    if LOG_FH is None:
        print(msg)
    else:
        LOG_FH.write(msg + "\n")
        LOG_FH.flush()


def summarize_fd(tsv_path: Path, debug: bool = False) -> tuple[int, str, int, str]:
    frame_count = 0
    total = 0.0
    valid = 0
    low = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        first = f.readline()
        delim = "\t" if "\t" in first else ("," if "," in first else None)
        sample_vals: list[float] = []
        if delim is not None:
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delim)
            fields = reader.fieldnames or []
            fd_col = None
            if debug:
                log(f"[DEBUG] reading file: {tsv_path}")
                log(f"[DEBUG] header columns: {fields}")
            if "framewise_displacement" in fields:
                fd_col = "framewise_displacement"
            if debug:
                log(f"[DEBUG] selected fd_col: {fd_col}")
            for row in reader:
                frame_count += 1
                if fd_col is None:
                    continue
                v = row.get(fd_col)
                if v is None:
                    continue
                s = str(v).strip()
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
                if len(sample_vals) < 5:
                    sample_vals.append(x)
        else:
            header_fields = re.split(r"\s+", first.strip()) if first.strip() else []
            if debug:
                log(f"[DEBUG] reading file: {tsv_path}")
                log(f"[DEBUG] header columns: {header_fields}")
            fd_idx = None
            for i, col in enumerate(header_fields):
                if col == "framewise_displacement":
                    fd_idx = i
                    break
            if debug:
                log(f"[DEBUG] selected fd_col index: {fd_idx}")
            for line in f:
                if not line.strip():
                    continue
                parts = re.split(r"\s+", line.strip())
                frame_count += 1
                if fd_idx is None or fd_idx >= len(parts):
                    continue
                s = parts[fd_idx].strip()
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
                if len(sample_vals) < 5:
                    sample_vals.append(x)
    if valid == 0:
        if debug:
            log(f"[DEBUG] frames={frame_count}, valid={valid}, low={low}, mean=NA, na_or_invalid={(frame_count - valid)}")
        return frame_count, "NA", 0, "NA"
    mean = total / valid
    if debug:
        log(f"[DEBUG] frames={frame_count}, valid={valid}, low={low}, mean={mean:.6f}, na_or_invalid={(frame_count - valid)}, samples={sample_vals}")
    return frame_count, f"{mean:.6f}", valid, f"{low / valid:.6f}"


def infer_run_idx(name: str) -> int | None:
    m = re.search(r"run-0*([0-9]+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def parse_ses(name: str) -> str:
    m = re.search(r"ses-([A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    return ""


def has_t1w_anat(fmriprep_dir: Path, sid: str) -> bool:
    anat_dir = fmriprep_dir / sid / "anat"
    if not anat_dir.exists():
        return False
    for f in anat_dir.glob("*preproc_T1w.nii.gz"):
        return True
    return False

def collect_subject_runs(fmriprep_dir: Path) -> dict[tuple[str, str], dict[int, Path]]:
    d: dict[tuple[str, str], dict[int, Path]] = {}
    for subj_dir in fmriprep_dir.glob("sub-*"):
        func_dir = subj_dir / "func"
        if not func_dir.exists():
            continue
        sid = subj_dir.name
        for tsv in func_dir.glob(f"{sid}_ses-*_task-rest_run-*_desc-confounds_timeseries.tsv"):
            ses = parse_ses(tsv.name)
            idx = infer_run_idx(tsv.name)
            if not ses or idx is None:
                continue
            key = (sid, ses)
            m = d.setdefault(key, {})
            m[idx] = tsv
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-dir", "--fmriprep_dir", required=True, dest="fmriprep_dir")
    parser.add_argument("--out", "--out_csv", default="rest_fd_summary.csv", dest="out")
    parser.add_argument("--debug", action="store_true", dest="debug")
    parser.add_argument("--log", dest="log")
    args = parser.parse_args()
    fdir = Path(args.fmriprep_dir)
    if not fdir.exists():
        print(f"Input directory not found: {fdir}", file=sys.stderr)
        return
    global LOG_FH
    if args.log:
        lp = Path(args.log)
        lp.parent.mkdir(parents=True, exist_ok=True)
        LOG_FH = lp.open("w", encoding="utf-8")
    runs_map = collect_subject_runs(fdir)
    rows = []
    excluded = 0
    eligible = 0
    for (sid, ses), run_files in runs_map.items():
        r1_fc = r1_fd = r1_low = None
        r2_fc = r2_fd = r2_low = None
        t1w_valid = "1" if has_t1w_anat(fdir, sid) else "0"
        if args.debug:
            log(f"[DEBUG] T1w_valid={t1w_valid} for {sid}")
        if 1 in run_files:
            a, b, c, d = summarize_fd(run_files[1], args.debug)
            r1_fc, r1_fd, r1_valid_cnt, r1_low = a, b, c, d
        if 2 in run_files:
            a, b, c, d = summarize_fd(run_files[2], args.debug)
            r2_fc, r2_fd, r2_valid_cnt, r2_low = a, b, c, d
        frame_issue = ((1 in run_files and r1_fc is not None and r1_fc < 100) or (2 in run_files and r2_fc is not None and r2_fc < 100))
        if args.debug and frame_issue:
            log(f"[DEBUG] frame<100: subid={sid}, ses={ses}, r1_frame={r1_fc}, r2_frame={r2_fc}")
        r1_valid = "1" if (r1_fd not in (None, "NA") and r1_low not in (None, "NA") and r1_fc is not None and r1_fc >= 100 and float(r1_fd) <= 0.5 and float(r1_low) > 0.4) else "0"
        r2_valid = "1" if (r2_fd not in (None, "NA") and r2_low not in (None, "NA") and r2_fc is not None and r2_fc >= 100 and float(r2_fd) <= 0.5 and float(r2_low) > 0.4) else "0"
        valid_num = (1 if r1_valid == "1" else 0) + (1 if r2_valid == "1" else 0)
        valid_subject = "1" if (valid_num >= 2 and t1w_valid == "1" and not frame_issue) else "0"
        invalid_reason = ""
        if frame_issue:
            reasons = []
            if r1_fc is not None and r1_fc < 100:
                reasons.append("rest1_frame_lt_100")
            if r2_fc is not None and r2_fc < 100:
                reasons.append("rest2_frame_lt_100")
            invalid_reason = ";".join(reasons) if reasons else "frame_lt_100"
        if valid_subject == "1":
            eligible += 1
        else:
            excluded += 1
        rows.append(
            {
                "subid": sid,
                "ses": ses,
                "rest1_frame": str(r1_fc) if r1_fc is not None else "NA",
                "rest1_fd": r1_fd if r1_fd is not None else "NA",
                "rest1_low_ratio": r1_low if r1_low is not None else "NA",
                "rest1_valid": r1_valid,
                "rest2_frame": str(r2_fc) if r2_fc is not None else "NA",
                "rest2_fd": r2_fd if r2_fd is not None else "NA",
                "rest2_low_ratio": r2_low if r2_low is not None else "NA",
                "rest2_valid": r2_valid,
                "valid_num": str(valid_num),
                "T1w_valid": t1w_valid,
                "valid_subject": valid_subject,
                "invalid_reason": invalid_reason,
            }
        )
    rows.sort(key=lambda r: (r["subid"], r["ses"]))
    headers = [
        "subid", "ses",
        "rest1_frame", "rest1_fd", "rest1_low_ratio", "rest1_valid",
        "rest2_frame", "rest2_fd", "rest2_low_ratio", "rest2_valid",
        "valid_num", "T1w_valid", "valid_subject", "invalid_reason",
    ]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"excluded={excluded}")
    log(f"eligible={eligible}")
    if LOG_FH is not None:
        LOG_FH.close()


if __name__ == "__main__":
    main()
