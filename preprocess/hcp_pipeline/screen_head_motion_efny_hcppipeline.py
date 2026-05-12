import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


MOTION_RADIUS_MM = 50.0


def find_subject_id(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return ""


def parse_run_index(run_name: str) -> int | None:
    match = re.fullmatch(r"rfMRI_REST([1-4])_[A-Z]+", run_name)
    if not match:
        return None
    return int(match.group(1))


def read_motion_regressors(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 12:
        raise ValueError(f"Motion regressors must have at least 12 columns, got {data.shape[1]}")
    return data


def compute_framewise_displacement(motion_data: np.ndarray) -> np.ndarray:
    deriv = motion_data[:, 6:12].copy()
    deriv[:, 3:6] = np.deg2rad(deriv[:, 3:6])
    trans_deriv = np.abs(deriv[:, :3])
    rot_deriv = np.abs(deriv[:, 3:6]) * MOTION_RADIUS_MM
    return trans_deriv.sum(axis=1) + rot_deriv.sum(axis=1)


def summarize_fd(motion_file: Path) -> tuple[int, str, int, str]:
    motion_data = read_motion_regressors(motion_file)
    fd = compute_framewise_displacement(motion_data)

    frame_count = int(fd.shape[0])
    valid = int(np.sum(~np.isnan(fd)))
    if valid == 0:
        return frame_count, "NA", 0, "NA"

    valid_fd = fd[~np.isnan(fd)]
    mean_fd = float(np.mean(valid_fd))
    low_ratio = float(np.mean(valid_fd < 0.2))
    return frame_count, f"{mean_fd:.6f}", valid, f"{low_ratio:.6f}"


def collect_rows(hcp_studyfolder: Path) -> list[dict]:
    subjects = {}
    pattern = "sub-*/MNINonLinear/Results/rfMRI_REST*_*"
    for run_dir in hcp_studyfolder.glob(pattern):
        if not run_dir.is_dir():
            continue
        subject_id = find_subject_id(run_dir)
        run_idx = parse_run_index(run_dir.name)
        if not subject_id or run_idx is None:
            continue

        motion_file = run_dir / "Movement_Regressors.txt"
        if not motion_file.exists():
            continue

        frame_num, mean_fd, valid_count, low_ratio = summarize_fd(motion_file)
        is_valid = False
        if mean_fd != "NA" and low_ratio != "NA":
            try:
                is_valid = (
                    frame_num == 180
                    and float(mean_fd) <= 0.5
                    and float(low_ratio) > 0.4
                )
            except Exception:
                is_valid = False

        runs = subjects.setdefault(subject_id, {})
        runs[run_idx] = {
            "frame": str(frame_num),
            "fd": mean_fd,
            "low_ratio": low_ratio,
            "valid": "1" if is_valid else "0",
        }

    rows = []
    for subject_id in sorted(subjects.keys()):
        runs = subjects[subject_id]
        row = {"subid": subject_id}
        valid_num = 0
        for i in range(1, 5):
            run_info = runs.get(i)
            if run_info:
                row[f"rest{i}_frame"] = run_info["frame"]
                row[f"rest{i}_fd"] = run_info["fd"]
                row[f"rest{i}_valid"] = run_info["valid"]
                row[f"rest{i}_low_ratio"] = run_info["low_ratio"]
                if run_info["valid"] == "1":
                    valid_num += 1
            else:
                row[f"rest{i}_frame"] = "NA"
                row[f"rest{i}_fd"] = "NA"
                row[f"rest{i}_valid"] = "0"
                row[f"rest{i}_low_ratio"] = "NA"
        row["valid_num"] = str(valid_num)
        row["valid_subject"] = "1" if valid_num >= 2 else "0"
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "subid",
        "rest1_frame", "rest1_fd", "rest1_low_ratio", "rest1_valid",
        "rest2_frame", "rest2_fd", "rest2_low_ratio", "rest2_valid",
        "rest3_frame", "rest3_fd", "rest3_low_ratio", "rest3_valid",
        "rest4_frame", "rest4_fd", "rest4_low_ratio", "rest4_valid",
        "valid_num", "valid_subject",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcp-studyfolder", required=True)
    parser.add_argument("--out", default="rest_fd_summary_hcppipeline.csv")
    args = parser.parse_args()

    hcp_studyfolder = Path(args.hcp_studyfolder)
    if not hcp_studyfolder.exists():
        print(f"Input directory not found: {hcp_studyfolder}", file=sys.stderr)
        return

    rows = collect_rows(hcp_studyfolder)
    write_csv(rows, Path(args.out))
    eligible = sum(1 for row in rows if row.get("valid_subject") == "1")
    excluded = len(rows) - eligible
    print(f"excluded={excluded}")
    print(f"eligible={eligible}")


if __name__ == "__main__":
    main()
