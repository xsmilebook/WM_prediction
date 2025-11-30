#!/usr/bin/env python3
"""
Unified atlas reslicing script that automatically finds denoised_bold.nii.gz files
from dataset paths and reslices atlases to match their geometry.
"""

import argparse
import logging
import sys
from pathlib import Path
import nibabel as nib
import nibabel.processing
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_denoised_bold_reference(dataset_path):
    """
    Automatically find a denoised_bold.nii.gz file in the dataset path.
    Searches recursively through all subject directories.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Look for xcpd directory first
    xcpd_path = dataset_path / "xcpd"
    if not xcpd_path.exists():
        # If no xcpd directory, search the entire dataset path
        search_path = dataset_path
        logger.info(f"No xcpd directory found, searching entire dataset path: {dataset_path}")
    else:
        search_path = xcpd_path
        logger.info(f"Searching in xcpd directory: {xcpd_path}")
    
    # Search for denoised BOLD files recursively
    denoised_files = list(search_path.rglob("*desc-denoised_bold.nii.gz"))
    
    if not denoised_files:
        raise FileNotFoundError(f"No denoised_bold.nii.gz files found in {search_path}")
    
    # Use the first found file as reference
    ref_file = denoised_files[0]
    logger.info(f"Found reference image: {ref_file}")
    
    # Log some info about the found file
    try:
        ref_img = nib.load(ref_file)
        logger.info(f"Reference image shape: {ref_img.shape}")
        logger.info(f"Reference image affine: {ref_img.affine[:3, :3]}")
    except Exception as e:
        logger.warning(f"Could not load reference image for info: {e}")
    
    return ref_file


def reslice_to_ref(atlas_path, ref_path, output_path):
    """
    Reslices an atlas image to match the geometry (resolution, affine, shape) of a reference image.
    Uses nearest-neighbor interpolation (order=0) to preserve integer labels.
    """
    try:
        logger.info(f"Loading atlas: {atlas_path}")
        atlas_img = nib.load(atlas_path)
        
        logger.info(f"Loading reference: {ref_path}")
        ref_img = nib.load(ref_path)
        
        # Handle 4D reference image
        if len(ref_img.shape) > 3:
            logger.info(f"Reference image is {len(ref_img.shape)}D. Using first 3 dimensions as spatial reference.")
            ref_img_3d = nib.Nifti1Image(np.empty(ref_img.shape[:3], dtype=np.int8), ref_img.affine, ref_img.header)
            ref_img = ref_img_3d

        logger.info("Reslicing...")
        resliced_img = nibabel.processing.resample_from_to(atlas_img, ref_img, order=0)
        
        logger.info(f"Saving resliced atlas to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(resliced_img, output_path)
        logger.info("Done.")
        
    except Exception as e:
        logger.error(f"Error during reslicing: {e}")
        sys.exit(1)


def process_single_dataset(dataset_path, dataset_name, atlas_dir, output_base_dir):
    """
    Process a single dataset: find reference image and reslice atlases.
    """
    logger.info(f"\n=== Processing {dataset_name} dataset ===")
    
    try:
        # Find reference image automatically
        ref_img_path = find_denoised_bold_reference(dataset_path)
        
        # Define output directory
        output_dir = output_base_dir / f"resliced_{dataset_name.lower()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define atlas paths (original atlases in the atlas root directory)
        gm_atlas_path = atlas_dir / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
        wm_atlas_path = atlas_dir / "rICBM_DTI_81_WMPM_60p_FMRIB58.nii.gz"
        
        # Verify atlas files exist
        if not gm_atlas_path.exists():
            raise FileNotFoundError(f"GM atlas not found: {gm_atlas_path}")
        if not wm_atlas_path.exists():
            raise FileNotFoundError(f"WM atlas not found: {wm_atlas_path}")
        
        # Reslice GM atlas
        gm_output_path = output_dir / f"Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz"
        reslice_to_ref(gm_atlas_path, ref_img_path, gm_output_path)
        
        # Reslice WM atlas
        wm_output_path = output_dir / f"rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz"
        reslice_to_ref(wm_atlas_path, ref_img_path, wm_output_path)
        
        logger.info(f"Successfully processed {dataset_name} dataset")
        logger.info(f"Output saved to: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {dataset_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified atlas reslicing script. Automatically finds denoised_bold.nii.gz "
                   "files from dataset paths and reslices atlases to match their geometry."
    )
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to dataset directory (e.g., d:/code/WM_prediction/data/CCNP)")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Name of the dataset (e.g., CCNP, EFNY, HCPD, PNC)")
    parser.add_argument("--atlas_dir", type=str, default="d:/code/WM_prediction/data/atlas",
                       help="Directory containing atlas files")
    parser.add_argument("--output_dir", type=str, default="d:/code/WM_prediction/data/atlas",
                       help="Output directory for resliced atlases")
    parser.add_argument("--gm_atlas", type=str, 
                       default="contrast_abcd/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resample.nii.gz",
                       help="Relative path to GM atlas within atlas_dir")
    parser.add_argument("--wm_atlas", type=str,
                       default="contrast_abcd/rICBM_DTI_81_WMPM_60p_FMRIB58_resample.nii.gz", 
                       help="Relative path to WM atlas within atlas_dir")
    
    args = parser.parse_args()
    
    # Set up paths
    dataset_path = Path(args.dataset_path)
    atlas_dir = Path(args.atlas_dir)
    output_base_dir = Path(args.output_dir)
    
    # Process the dataset
    success = process_single_dataset(
        dataset_path, 
        args.dataset_name, 
        atlas_dir, 
        output_base_dir
    )
    
    if success:
        logger.info(f"\n=== Summary ===")
        logger.info(f"Successfully processed {args.dataset_name} dataset")
        logger.info(f"Resliced atlases saved to: {output_base_dir}/resliced_{args.dataset_name.lower()}")
        sys.exit(0)
    else:
        logger.error(f"Failed to process {args.dataset_name} dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()