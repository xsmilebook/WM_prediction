#!/usr/bin/env python3
"""Run EFNY HCP-pipeline FC generation by reusing the existing unified FC logic."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib

SCRIPT_PATH = Path(__file__).resolve()
# Entry script lives below src/, so expose both src/ and conn_matrix/ for imports
# without changing the existing unified pipeline layout.
sys.path.insert(0, str(SCRIPT_PATH.parents[2]))
sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
from preprocess.hcp_pipeline.build_hcp_tissue_dseg import build_tissue_dseg  # noqa: E402
from process_dataset_unified import DatasetProcessor  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_compatibility_dseg(subject_id: str, hcp_subject_dir: Path, output_root: Path) -> Path:
    """Create a minimal 3-class tissue dseg compatible with the existing EFNY FC pipeline."""
    ribbon_path = hcp_subject_dir / "MNINonLinear" / "ribbon.nii.gz"
    wmparc_path = hcp_subject_dir / "MNINonLinear" / "ROIs" / "wmparc.2.nii.gz"

    anat_dir = output_root / "compat_fmriprep" / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    out_path = anat_dir / f"{subject_id}_space-MNI152NLin6Asym_dseg.nii.gz"
    build_tissue_dseg(ribbon_path=ribbon_path, wmparc_path=wmparc_path, output_path=out_path)

    dseg_data = nib.load(str(out_path)).get_fdata()
    gm_voxels = int((dseg_data == 1).sum())
    wm_voxels = int((dseg_data == 2).sum())
    csf_voxels = int((dseg_data == 3).sum())
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
