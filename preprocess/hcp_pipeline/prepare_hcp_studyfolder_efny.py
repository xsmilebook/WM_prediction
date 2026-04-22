#!/usr/bin/env python3
"""Stage EFNY BIDS inputs into an HCP-style StudyFolder layout using symlinks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BIDS_ROOT = Path("/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/BIDS")
DEFAULT_STUDY_FOLDER = Path("/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder")


@dataclass(frozen=True)
class RunSpec:
    run_index: int
    hcp_name: str
    bold_path: Path


def read_subjects(subject: str | None, subject_list: str | None) -> list[str]:
    if subject:
        return [subject]
    if subject_list:
        items = []
        with open(subject_list, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    items.append(line)
        return items
    raise ValueError("Either --subject or --subject-list must be provided.")


def find_single(path_iter: Iterable[Path], description: str) -> Path:
    paths = sorted(path_iter)
    if not paths:
        raise FileNotFoundError(f"Missing required {description}")
    return paths[0]


def symlink_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def sidecar_json_path(src: Path) -> Path:
    if src.name.endswith(".nii.gz"):
        return src.with_suffix("").with_suffix(".json")
    return src.with_suffix(".json")


def link_with_json(src: Path, dst: Path) -> None:
    symlink_or_replace(src, dst)
    json_src = sidecar_json_path(src)
    if json_src.exists():
        symlink_or_replace(json_src, dst.with_suffix(".json"))


def find_rest_runs(subject_dir: Path) -> list[RunSpec]:
    run_paths = sorted(subject_dir.glob("func/*task-rest_run-*_bold.nii*"))
    specs: list[RunSpec] = []
    for path in run_paths:
        if path.suffix == ".json":
            continue
        match = re.search(r"_run-(\d+)_bold", path.name)
        if not match:
            continue
        run_index = int(match.group(1))
        meta_path = sidecar_json_path(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing JSON sidecar for {path}")
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        pe_dir = metadata.get("PhaseEncodingDirection")
        suffix_map = {"j": "PA", "j+": "PA", "j-": "AP", "i": "RL", "i+": "RL", "i-": "LR"}
        if pe_dir not in suffix_map:
            raise ValueError(f"Unsupported PhaseEncodingDirection {pe_dir!r} in {meta_path}")
        hcp_name = f"rfMRI_REST{run_index}_{suffix_map[pe_dir]}"
        specs.append(RunSpec(run_index=run_index, hcp_name=hcp_name, bold_path=path))
    if not specs:
        raise FileNotFoundError(f"No rest runs found under {subject_dir / 'func'}")
    run_indices = sorted(item.run_index for item in specs)
    if len(specs) > 4:
        raise ValueError(
            f"Expected 1 to 4 rest runs for {subject_dir.name}, found {len(specs)}: {run_indices}"
        )
    if any(index < 1 or index > 4 for index in run_indices):
        raise ValueError(
            f"EFNY rest run index must be within 1 to 4 for {subject_dir.name}, found: {run_indices}"
        )
    return sorted(specs, key=lambda item: item.run_index)


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "kind",
        "hcp_name",
        "source_path",
        "staged_path",
        "notes",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def stage_subject(subject: str, bids_root: Path, study_folder: Path) -> list[dict[str, str]]:
    subject_dir = bids_root / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    t1_path = find_single(subject_dir.glob("anat/*run-1*T1w.nii*"), "run-1 T1w")
    t2_path = find_single(subject_dir.glob("anat/*T2w.nii*"), "T2w")
    fmap_ap = find_single(subject_dir.glob("fmap/*dir-AP*acq-rest_epi.nii*"), "rest AP spin-echo fmap")
    fmap_pa = find_single(subject_dir.glob("fmap/*dir-PA*acq-rest_epi.nii*"), "rest PA spin-echo fmap")
    runs = find_rest_runs(subject_dir)

    subject_root = study_folder / subject / "unprocessed" / "3T"
    t1_dir = subject_root / "T1w_MPR1"
    t2_dir = subject_root / "T2w_SPC1"

    manifest_rows: list[dict[str, str]] = []

    t1_staged = t1_dir / f"{subject}_3T_T1w_MPR1{t1_path.suffix}"
    t2_staged = t2_dir / f"{subject}_3T_T2w_SPC1{t2_path.suffix}"
    fmap_ap_staged = t1_dir / f"{subject}_3T_SpinEchoFieldMap_AP{fmap_ap.suffix}"
    fmap_pa_staged = t1_dir / f"{subject}_3T_SpinEchoFieldMap_PA{fmap_pa.suffix}"

    link_with_json(t1_path, t1_staged)
    link_with_json(t2_path, t2_staged)
    link_with_json(fmap_ap, fmap_ap_staged)
    link_with_json(fmap_pa, fmap_pa_staged)

    manifest_rows.extend(
        [
            {"subject": subject, "kind": "anat", "hcp_name": "T1w_MPR1", "source_path": str(t1_path), "staged_path": str(t1_staged), "notes": "primary T1w input"},
            {"subject": subject, "kind": "anat", "hcp_name": "T2w_SPC1", "source_path": str(t2_path), "staged_path": str(t2_staged), "notes": "primary T2w input"},
            {"subject": subject, "kind": "fmap", "hcp_name": "SpinEchoFieldMap_AP", "source_path": str(fmap_ap), "staged_path": str(fmap_ap_staged), "notes": "structural and rest TOPUP negative polarity"},
            {"subject": subject, "kind": "fmap", "hcp_name": "SpinEchoFieldMap_PA", "source_path": str(fmap_pa), "staged_path": str(fmap_pa_staged), "notes": "structural and rest TOPUP positive polarity"},
        ]
    )

    for run in runs:
        run_dir = subject_root / run.hcp_name
        run_staged = run_dir / f"{subject}_3T_{run.hcp_name}{run.bold_path.suffix}"
        run_fmap_ap = run_dir / f"{subject}_3T_SpinEchoFieldMap_AP{fmap_ap.suffix}"
        run_fmap_pa = run_dir / f"{subject}_3T_SpinEchoFieldMap_PA{fmap_pa.suffix}"

        link_with_json(run.bold_path, run_staged)
        link_with_json(fmap_ap, run_fmap_ap)
        link_with_json(fmap_pa, run_fmap_pa)

        manifest_rows.extend(
            [
                {"subject": subject, "kind": "rest", "hcp_name": run.hcp_name, "source_path": str(run.bold_path), "staged_path": str(run_staged), "notes": f"rest run {run.run_index}"},
                {"subject": subject, "kind": "rest_fmap", "hcp_name": f"{run.hcp_name}:AP", "source_path": str(fmap_ap), "staged_path": str(run_fmap_ap), "notes": "per-run TOPUP negative polarity"},
                {"subject": subject, "kind": "rest_fmap", "hcp_name": f"{run.hcp_name}:PA", "source_path": str(fmap_pa), "staged_path": str(run_fmap_pa), "notes": "per-run TOPUP positive polarity"},
            ]
        )

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an HCP-style StudyFolder for EFNY from BIDS inputs.")
    parser.add_argument("--bids-root", default=str(DEFAULT_BIDS_ROOT))
    parser.add_argument("--study-folder", default=str(DEFAULT_STUDY_FOLDER))
    parser.add_argument("--subject")
    parser.add_argument("--subject-list")
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    study_folder = Path(args.study_folder)
    study_folder.mkdir(parents=True, exist_ok=True)
    (study_folder / "logs").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for subject in read_subjects(args.subject, args.subject_list):
        rows.extend(stage_subject(subject, bids_root, study_folder))

    write_manifest(rows, study_folder / "manifests" / "hcp_efny_manifest.tsv")


if __name__ == "__main__":
    main()
