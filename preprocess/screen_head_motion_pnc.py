import argparse
import csv
import re
import sys
from pathlib import Path


LOG_FH = None

def log(msg: str) -> None:
    global LOG_FH
    if LOG_FH is None:
        print(msg)
    else:
        LOG_FH.write(msg + "\n")
        LOG_FH.flush()


def find_subject_id(p: Path) -> str:
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return ""


def parse_ses(name: str) -> str:
    m = re.search(r"ses-([A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    return ""


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


def collect_subject_sessions(fmriprep_dir: Path, debug: bool = False) -> dict[tuple[str, str], Path]:
    d: dict[tuple[str, str], Path] = {}
    for subj_dir in fmriprep_dir.glob("sub-*"):
        sid = subj_dir.name
        if debug:
            log(f"[DEBUG] scanning subject: {sid}")
        for ses_dir in subj_dir.glob("ses-*"):
            func_dir = ses_dir / "func"
            if not func_dir.exists():
                continue
            ses = ses_dir.name.split("ses-")[-1]
            if debug:
                log(f"[DEBUG] func directory: {func_dir}")
            for tsv in func_dir.glob(f"{sid}_ses-{ses}_task-rest*_desc-confounds_timeseries.tsv"):
                if debug:
                    log(f"[DEBUG] found confounds file: {tsv}")
                d[(sid, ses)] = tsv
                break
    if debug:
        log(f"[DEBUG] subject-session pairs collected: {len(d)}")
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
    sessions = collect_subject_sessions(fdir, args.debug)
    rows = []
    excluded = 0
    eligible = 0
    for (sid, ses), path in sessions.items():
        fc, fd, vcnt, low = summarize_fd(path, args.debug)
        if fc < 60:
            excluded += 1
            if args.debug:
                log(f"[DEBUG] excluded due to frame<60: subid={sid}, ses={ses}, frame={fc}")
            continue
        valid = "1" if (fd != "NA" and low != "NA" and fc >= 60 and float(fd) <= 0.5 and float(low) > 0.4) else "0"
        valid_subject = valid
        if valid_subject == "1":
            eligible += 1
        else:
            excluded += 1
        rows.append(
            {
                "subid": sid,
                "ses": ses,
                "rest_frame": str(fc),
                "rest_fd": fd,
                "rest_low_ratio": low,
                "rest_valid": valid,
                "valid_subject": valid_subject,
            }
        )
    rows.sort(key=lambda r: (r["subid"], r["ses"]))
    headers = [
        "subid", "ses",
        "rest_frame", "rest_fd", "rest_low_ratio", "rest_valid",
        "valid_subject",
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