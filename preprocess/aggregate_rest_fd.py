import argparse
import csv
import re
from pathlib import Path


def find_subject_id(p: Path) -> str:
    for part in p.parts:
        if part.startswith("sub-"):
            return part
    return ""


def parse_task_run(name: str) -> str:
    task_match = re.search(r"task-([A-Za-z0-9]+)", name)
    run_match = re.search(r"run-([0-9]+)", name)
    task = task_match.group(1) if task_match else ""
    run = run_match.group(1) if run_match else ""
    if task and run:
        return f"{task}_{run}"
    return task


def summarize_fd(tsv_path: Path) -> tuple[int, str]:
    frame_count = 0
    total = 0.0
    valid = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "framewise_displacement" not in reader.fieldnames:
            return 0, "NA"
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
    if valid == 0:
        return frame_count, "NA"
    return frame_count, f"{total / valid:.6f}"


def collect_rows(fmriprep_dir: Path) -> list[dict]:
    rows = []
    for tsv in fmriprep_dir.rglob("*task-rest*desc-confounds_timeseries.tsv"):
        if "func" not in tsv.parts:
            continue
        subject_id = find_subject_id(tsv)
        label = parse_task_run(tsv.name)
        frame_num, mean_fd = summarize_fd(tsv)
        rows.append(
            {
                "subject_id": subject_id,
                "task_run": label or "rest",
                "frame_num": str(frame_num),
                "mean_fd": mean_fd,
            }
        )
    rows.sort(key=lambda r: (r["subject_id"], r["task_run"]))
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "task_run", "frame_num", "mean_fd"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep-dir", required=True)
    parser.add_argument("--out", default="rest_fd_summary.csv")
    args = parser.parse_args()
    fdir = Path(args.fmriprep_dir)
    rows = collect_rows(fdir)
    write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()