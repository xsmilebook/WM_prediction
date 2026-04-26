#!/usr/bin/env python3
"""Run EFNY HCP-pipeline FC generation by reusing the existing unified FC logic."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Entry script lives one level below conn_matrix/, so add the parent directory
# in order to import the existing unified pipeline without changing its logic.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process_dataset_unified import DatasetProcessor  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


RIBBON_GM_LABELS = {3, 42}
RIBBON_WM_LABELS = {2, 41}
CSF_LABELS = {4, 5, 14, 15, 24, 31, 43, 44, 63}


def build_compatibility_dseg(subject_id: str, hcp_subject_dir: Path, output_root: Path) -> Path:
    """Create a minimal 3-class tissue dseg compatible with the existing EFNY FC pipeline."""
    ribbon_path = hcp_subject_dir / "MNINonLinear" / "ribbon.nii.gz"
    aseg_path = hcp_subject_dir / "MNINonLinear" / "aparc+aseg.nii.gz"

    ribbon_img = nib.load(str(ribbon_path))
    ribbon_data = np.asanyarray(ribbon_img.dataobj).astype(np.int16)
    aseg_data = np.asanyarray(nib.load(str(aseg_path)).dataobj).astype(np.int16)

    tissue_dseg = np.zeros(ribbon_data.shape, dtype=np.int16)
    tissue_dseg[np.isin(ribbon_data, list(RIBBON_GM_LABELS))] = 1
    tissue_dseg[np.isin(ribbon_data, list(RIBBON_WM_LABELS))] = 2
    csf_mask = np.isin(aseg_data, list(CSF_LABELS)) & (tissue_dseg == 0)
    tissue_dseg[csf_mask] = 3

    anat_dir = output_root / "compat_fmriprep" / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    out_path = anat_dir / f"{subject_id}_space-MNI152NLin6Asym_dseg.nii.gz"
    nib.save(nib.Nifti1Image(tissue_dseg, ribbon_img.affine, ribbon_img.header), str(out_path))

    gm_voxels = int((tissue_dseg == 1).sum())
    wm_voxels = int((tissue_dseg == 2).sum())
    csf_voxels = int((tissue_dseg == 3).sum())
    logger.info(
        "Saved compatibility dseg to %s (GM=%d, WM=%d, CSF=%d)",
        out_path,
        gm_voxels,
        wm_voxels,
        csf_voxels,
    )
    return out_path


def run_subject(subject_id: str, project_root: Path, output_root: Path) -> None:
    hcp_subject_dir = project_root / "data" / "EFNY" / "hcp_studyfolder" / subject_id
    xcpd_root = project_root / "data" / "EFNY" / "xcpd_hcp" / "step_2nd_24PcsfGlobal"
    atlas_root = project_root / "data" / "atlas" / "resliced_efny"

    build_compatibility_dseg(subject_id, hcp_subject_dir, output_root)

    processor = DatasetProcessor(
        dataset_name="EFNY",
        subject_id=subject_id,
        dataset_path=str(output_root),
        mask_output_dir=str(output_root / "wm_postproc"),
        fc_output_dir=str(output_root / "fc_matrix" / "individual"),
        z_output_dir=str(output_root / "fc_matrix" / "individual_z"),
    )
    processor.fmriprep_dirs = [output_root / "compat_fmriprep"]
    processor.xcpd_dirs = [xcpd_root]
    processor.table_dir = output_root / "table"

    gm_atlas = atlas_root / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz"
    wm_atlas = atlas_root / "rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz"

    success = processor.process_subject(str(gm_atlas), str(wm_atlas))
    if not success:
        raise RuntimeError(f"FC generation failed for {subject_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EFNY HCP-pipeline FC matrices for one subject.")
    parser.add_argument("--subject_id", required=True, help="Subject ID, e.g. sub-THU20231118133GYC")
    parser.add_argument(
        "--project_root",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction",
        help="Project root path",
    )
    parser.add_argument(
        "--output_root",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcppipeline_fc",
        help="Independent output root for HCP-pipeline FC results",
    )
    args = parser.parse_args()

    run_subject(args.subject_id, Path(args.project_root), Path(args.output_root))


if __name__ == "__main__":
    main()
