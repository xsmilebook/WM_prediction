import argparse
import csv
import re
import sys
from pathlib import Path

# example:
# python screen_head_motion_hcpd.py --fmriprep-dir /ibmgpfs/cuizaixu_lab/zhaoshaoling/MSC_data/HCPD/code_xcpd0.7.1rc5_hcpMiniPrepData/final2025/data/xcpd0.7.1rc5/step_2nd_24PcsfGlobal --out /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/table/rest_fd_summary.csv

def find_subject_id(p: Path) -> str:
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return ""


def summarize_fd(tsv_path: Path) -> tuple[int, float, int, int]:
    frame_count = 0
    total = 0.0
    valid = 0
    low = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "framewise_displacement" not in reader.fieldnames:
            return 0, 0.0, 0, 0
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
    return frame_count, total, valid, low


def summarize_fd_list(paths: list[Path]) -> tuple[int, str, int, str]:
    frame = 0
    total = 0.0
    valid = 0
    low = 0
    for p in paths:
        f, t, v, l = summarize_fd(p)
        frame += f
        total += t
        valid += v
        low += l
    if valid == 0:
        return frame, "NA", 0, "NA"
    return frame, f"{total / valid:.6f}", valid, f"{low / valid:.6f}"


def infer_run_idx(name: str) -> int | None:
    m = re.search(r"run-([12])", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"task-REST([12])", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def parse_acq(name: str) -> str | None:
    m = re.search(r"acq-([A-Za-z]+)", name)
    if m:
        v = m.group(1).upper()
        if v in ("AP", "PA"):
            return v
    return None


def collect_subject_runs(fmriprep_dir: Path) -> dict[str, dict[str, Path]]:
    d: dict[str, dict[str, Path]] = {}
    for tsv in fmriprep_dir.rglob("*desc-filtered_motion.tsv"):
        if "func" not in tsv.parts:
            continue
        if not re.search(r"task-REST", tsv.name, flags=re.IGNORECASE):
            continue
        sid = find_subject_id(tsv)
        idx = infer_run_idx(tsv.name)
        acq = parse_acq(tsv.name)
        if idx is None or idx not in (1, 2) or acq is None:
            continue
        m = d.setdefault(sid, {})
        key = f"{idx}_{acq}"
        m[key] = tsv
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
        needed = ["1_AP", "1_PA", "2_AP", "2_PA"]
        if any(k not in run_files for k in needed):
            excluded += 1
            continue
        def compute(path: Path) -> tuple[int, str, str]:
            fc, tot, v, low = summarize_fd(path)
            if v == 0:
                return fc, "NA", "NA"
            return fc, f"{tot / v:.6f}", f"{low / v:.6f}"
        r1_ap_fc, r1_ap_fd, r1_ap_low = compute(run_files["1_AP"])
        r1_pa_fc, r1_pa_fd, r1_pa_low = compute(run_files["1_PA"])
        r2_ap_fc, r2_ap_fd, r2_ap_low = compute(run_files["2_AP"])
        r2_pa_fc, r2_pa_fd, r2_pa_low = compute(run_files["2_PA"])
        r1_ap_valid = "1" if (r1_ap_fd != "NA" and r1_ap_low != "NA" and float(r1_ap_fd) <= 0.5 and float(r1_ap_low) > 0.4) else "0"
        r1_pa_valid = "1" if (r1_pa_fd != "NA" and r1_pa_low != "NA" and float(r1_pa_fd) <= 0.5 and float(r1_pa_low) > 0.4) else "0"
        r2_ap_valid = "1" if (r2_ap_fd != "NA" and r2_ap_low != "NA" and float(r2_ap_fd) <= 0.5 and float(r2_ap_low) > 0.4) else "0"
        r2_pa_valid = "1" if (r2_pa_fd != "NA" and r2_pa_low != "NA" and float(r2_pa_fd) <= 0.5 and float(r2_pa_low) > 0.4) else "0"
        valid_num = sum(1 for s in (r1_ap_valid, r1_pa_valid, r2_ap_valid, r2_pa_valid) if s == "1")
        valid_subject = "1" if valid_num == 4 else "0"
        if valid_subject == "1":
            eligible += 1
        else:
            excluded += 1
        rows.append(
            {
                "subid": sid,
                "rest1_AP_frame": str(r1_ap_fc),
                "rest1_AP_fd": r1_ap_fd,
                "rest1_AP_low_ratio": r1_ap_low,
                "rest1_AP_valid": r1_ap_valid,
                "rest1_PA_frame": str(r1_pa_fc),
                "rest1_PA_fd": r1_pa_fd,
                "rest1_PA_low_ratio": r1_pa_low,
                "rest1_PA_valid": r1_pa_valid,
                "rest2_AP_frame": str(r2_ap_fc),
                "rest2_AP_fd": r2_ap_fd,
                "rest2_AP_low_ratio": r2_ap_low,
                "rest2_AP_valid": r2_ap_valid,
                "rest2_PA_frame": str(r2_pa_fc),
                "rest2_PA_fd": r2_pa_fd,
                "rest2_PA_low_ratio": r2_pa_low,
                "rest2_PA_valid": r2_pa_valid,
                "valid_num": str(valid_num),
                "valid_subject": valid_subject,
            }
        )
    rows.sort(key=lambda r: r["subid"])
    headers = [
        "subid",
        "rest1_AP_frame", "rest1_AP_fd", "rest1_AP_low_ratio", "rest1_AP_valid",
        "rest1_PA_frame", "rest1_PA_fd", "rest1_PA_low_ratio", "rest1_PA_valid",
        "rest2_AP_frame", "rest2_AP_fd", "rest2_AP_low_ratio", "rest2_AP_valid",
        "rest2_PA_frame", "rest2_PA_fd", "rest2_PA_low_ratio", "rest2_PA_valid",
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