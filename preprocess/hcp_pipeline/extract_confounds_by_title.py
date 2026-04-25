#!/usr/bin/env python3
"""Generate bridge confounds and custom confounds for EFNY HCP->XCP-D."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


CSF_LABELS = {
    4, 5, 14, 15, 24, 31, 43, 44, 63,
}

MOTION_COLUMNS = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
)


def load_image(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return nib.load(str(path))


def compute_mean_signal(data_4d: np.ndarray, mask_3d: np.ndarray, name: str) -> np.ndarray:
    if mask_3d.shape != data_4d.shape[:3]:
        raise ValueError(
            f"{name} mask shape {mask_3d.shape} does not match BOLD shape {data_4d.shape[:3]}"
        )
    voxels = data_4d[mask_3d]
    if voxels.size == 0:
        raise ValueError(f"{name} mask is empty")
    return voxels.mean(axis=0)


def derivative(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] > 1:
        out[1:] = np.diff(values)
    return out


def read_motion_regressors(path: Path, n_frames: int) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] != n_frames:
        raise ValueError(
            f"Motion regressors have {data.shape[0]} frames, but BOLD has {n_frames} frames"
        )
    if data.shape[1] < 12:
        raise ValueError(
            f"Motion regressors must have at least 12 columns, got {data.shape[1]}"
        )

    confounds: dict[str, np.ndarray] = {}
    base = data[:, :6]
    deriv = data[:, 6:12]
    # HCP stores rotations in degrees, while fMRIPrep-style confounds use radians.
    base[:, 3:6] = np.deg2rad(base[:, 3:6])
    deriv[:, 3:6] = np.deg2rad(deriv[:, 3:6])
    for idx, name in enumerate(MOTION_COLUMNS):
        confounds[name] = base[:, idx]
        confounds[f"{name}_derivative1"] = deriv[:, idx]
        confounds[f"{name}_power2"] = np.square(base[:, idx])
        confounds[f"{name}_derivative1_power2"] = np.square(deriv[:, idx])

    radius = 50.0
    trans_deriv = np.abs(deriv[:, :3])
    rot_deriv = np.abs(deriv[:, 3:6]) * radius
    confounds["framewise_displacement"] = trans_deriv.sum(axis=1) + rot_deriv.sum(axis=1)
    return confounds


def write_tsv(path: Path, columns: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(columns.keys())
    n_rows = len(next(iter(columns.values())))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for row_idx in range(n_rows):
            row = {name: f"{columns[name][row_idx]:.10f}" for name in headers}
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bold-file", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--seg-file", required=True)
    parser.add_argument("--brain-mask-file", required=True)
    parser.add_argument("--base-confounds-out", required=True)
    parser.add_argument("--custom-confounds-out", required=True)
    parser.add_argument("--bold-json-out", required=False)
    parser.add_argument("--task-name", default="rest")
    args = parser.parse_args()

    bold_img = load_image(Path(args.bold_file))
    seg_img = load_image(Path(args.seg_file))
    brain_mask_img = load_image(Path(args.brain_mask_file))

    bold_data = np.asanyarray(bold_img.dataobj, dtype=np.float32)
    if bold_data.ndim != 4:
        raise ValueError(f"BOLD image must be 4D, got shape {bold_data.shape}")

    seg_data = np.asanyarray(seg_img.dataobj)
    brain_mask_data = np.asanyarray(brain_mask_img.dataobj)
    if seg_data.ndim > 3:
        seg_data = np.squeeze(seg_data)
    if brain_mask_data.ndim > 3:
        brain_mask_data = np.squeeze(brain_mask_data)

    brain_mask = brain_mask_data > 0
    seg_int = np.rint(seg_data).astype(np.int32)
    csf_mask = np.isin(seg_int, list(CSF_LABELS))

    global_signal = compute_mean_signal(bold_data, brain_mask, "brain")
    csf_signal = compute_mean_signal(bold_data, csf_mask, "CSF")

    n_frames = bold_data.shape[-1]
    base_confounds = read_motion_regressors(Path(args.motion_file), n_frames)
    custom_confounds = {
        "csf": csf_signal,
        "global_signal": global_signal,
    }
    for key in ("csf", "global_signal"):
        d1 = derivative(custom_confounds[key])
        custom_confounds[f"{key}_derivative1"] = d1
        custom_confounds[f"{key}_power2"] = np.square(custom_confounds[key])
        custom_confounds[f"{key}_derivative1_power2"] = np.square(d1)

    write_tsv(Path(args.base_confounds_out), base_confounds)
    write_tsv(Path(args.custom_confounds_out), custom_confounds)

    if args.bold_json_out:
        tr = float(bold_img.header.get_zooms()[-1])
        metadata = {
            "RepetitionTime": tr,
            "TaskName": args.task_name,
        }
        out_path = Path(args.bold_json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
