#!/usr/bin/env python3
"""Build a minimal 3-class tissue dseg from HCP ribbon and wmparc outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np


RIBBON_GM_LABELS = {3, 42}
RIBBON_WM_LABELS = {2, 41}
WMPARC_WM_LABELS = {7, 46}
CSF_LABELS = {4, 5, 14, 15, 24, 31, 43, 44, 63}


def load_image(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return nib.load(str(path))


def resample_ribbon_to_wmparc(
    ribbon_img: nib.spatialimages.SpatialImage,
    wmparc_img: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    """Resample ribbon to the wmparc grid so the output dseg stays 2 mm like the BOLD."""
    if ribbon_img.shape == wmparc_img.shape and np.allclose(ribbon_img.affine, wmparc_img.affine):
        ribbon_data = np.asanyarray(ribbon_img.dataobj)
    else:
        ribbon_resampled = nibabel.processing.resample_from_to(ribbon_img, wmparc_img, order=0)
        ribbon_data = np.asanyarray(ribbon_resampled.dataobj)
    return np.rint(ribbon_data).astype(np.int16)


def build_tissue_dseg(
    ribbon_path: Path,
    wmparc_path: Path,
    output_path: Path,
) -> Path:
    ribbon_img = load_image(ribbon_path)
    wmparc_img = load_image(wmparc_path)

    ribbon_data = resample_ribbon_to_wmparc(ribbon_img, wmparc_img)
    wmparc_data = np.rint(np.asanyarray(wmparc_img.dataobj)).astype(np.int16)

    tissue_dseg = np.zeros(wmparc_data.shape, dtype=np.int16)
    tissue_dseg[np.isin(ribbon_data, list(RIBBON_GM_LABELS))] = 1
    tissue_dseg[np.isin(ribbon_data, list(RIBBON_WM_LABELS))] = 2

    # wmparc supplements cerebellar white matter that is outside the cortical ribbon.
    cerebellar_wm_mask = np.isin(wmparc_data, list(WMPARC_WM_LABELS))
    tissue_dseg[cerebellar_wm_mask] = 2

    csf_mask = np.isin(wmparc_data, list(CSF_LABELS)) & (tissue_dseg == 0)
    tissue_dseg[csf_mask] = 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    header = wmparc_img.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(tissue_dseg, wmparc_img.affine, header), str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 3-class tissue dseg from HCP ribbon and wmparc.")
    parser.add_argument("--ribbon-file", required=True)
    parser.add_argument("--wmparc-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_tissue_dseg(
        ribbon_path=Path(args.ribbon_file),
        wmparc_path=Path(args.wmparc_file),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
