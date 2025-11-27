#!/usr/bin/env python3
import argparse
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path

def setup_logger():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logger()

class MaskGenerator:    
    def __init__(self, subject_id, fmriprep_dir, xcpd_dir, output_dir):
        self.subject_id = subject_id
        self.fmriprep_dir = Path(fmriprep_dir)
        self.xcpd_dir = Path(xcpd_dir)
        self.output_dir = Path(output_dir)

    def _get_anat_files(self):
        """Finds the segmentation file for the subject."""
        # Search pattern based on user example: 
        # sub-NDARINV00HEV6HB/anat/sub-NDARINV00HEV6HB_space-MNI152NLin6Asym_dseg.nii.gz
        # Also compatible with standard BIDS: sub-XX/ses-YY/anat/...
        
        # Try finding under anat directly (no session)
        anat_path = self.fmriprep_dir / self.subject_id / 'anat'
        dseg_files = list(anat_path.glob(f'{self.subject_id}*space-MNI152NLin6Asym*dseg.nii.gz'))
        
        if not dseg_files:
            # Try searching with session directories if not found directly
            dseg_files = list(self.fmriprep_dir.glob(f'{self.subject_id}/*/anat/{self.subject_id}*space-MNI152NLin6Asym*dseg.nii.gz'))
            
        if not dseg_files:
            raise FileNotFoundError(f"Could not find segmentation file for {self.subject_id} in {self.fmriprep_dir}")
        
        # Return the first match (assuming one dseg per subject/session preference)
        return dseg_files[0]

    def _get_func_files(self):
        """Finds all functional runs for the subject."""
        # Search pattern based on user example:
        # sub-NDARINV00CY2MDM/ses-baselineYear1Arm1/func/sub-NDARINV00CY2MDM_ses-baselineYear1Arm1_task-rest_run-1_space-MNI152NLin6Asym_desc-denoised_bold.nii.gz
        
        # Search recursively for func files under the subject directory in xcpd_dir
        # We look for *desc-denoised_bold.nii.gz
        func_files = list(self.xcpd_dir.glob(f'{self.subject_id}/**/func/{self.subject_id}*space-MNI152NLin6Asym*desc-denoised_bold.nii.gz'))
        
        if not func_files:
             logger.warning(f"No functional files found for {self.subject_id} in {self.xcpd_dir}")
        
        return sorted(func_files)

    def create_tissue_masks(self, dseg_path):
        """Creates GM and WM masks from the dseg file."""
        logger.info(f"Loading segmentation file: {dseg_path}")
        dseg_img = nib.load(dseg_path)
        dseg_data = dseg_img.get_fdata()
        
        # 2 = GM, 3 = WM (Based on typical FreeSurfer/BIDS dseg, ABCD dataset is different)
        gm_mask = (dseg_data == 2).astype(np.float32)
        wm_mask = (dseg_data == 3).astype(np.float32)
        
        # Save masks
        anat_out_dir = self.output_dir / self.subject_id / 'anat'
        anat_out_dir.mkdir(parents=True, exist_ok=True)
        
        gm_mask_path = anat_out_dir / f'{self.subject_id}_space-MNI152NLin6Asym_dseg_GM.nii.gz'
        wm_mask_path = anat_out_dir / f'{self.subject_id}_space-MNI152NLin6Asym_dseg_WM.nii.gz'
        
        # Save using the affine header from original dseg
        nib.save(nib.Nifti1Image(gm_mask, dseg_img.affine, dseg_img.header), gm_mask_path)
        nib.save(nib.Nifti1Image(wm_mask, dseg_img.affine, dseg_img.header), wm_mask_path)
        
        return gm_mask, wm_mask, gm_mask_path, wm_mask_path

    def process_run(self, func_path, gm_mask, wm_mask):
        """Processes a single functional run: Mask -> Save (No Smoothing)."""
        logger.info(f"Processing run: {func_path.name}")
        
        # Prepare output directory
        func_out_dir = self.output_dir / self.subject_id / 'func'
        func_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load functional data
        func_img = nib.load(func_path)
        func_data = func_img.get_fdata()
        
        # --- Step 1: Masking (Vectorized) ---
        # Expand mask dimensions to match 4D functional data (x, y, z, t)
        
        gm_masked_data = func_data * gm_mask[..., np.newaxis]
        wm_masked_data = func_data * wm_mask[..., np.newaxis]
        
        # Define output filenames
        prefix = func_path.name.replace('.nii.gz', '')
        gm_out_path = func_out_dir / f'{prefix}_GM_masked.nii.gz'
        wm_out_path = func_out_dir / f'{prefix}_WM_masked.nii.gz'
        
        # Save masked data
        nib.save(nib.Nifti1Image(gm_masked_data, func_img.affine, func_img.header), gm_out_path)
        nib.save(nib.Nifti1Image(wm_masked_data, func_img.affine, func_img.header), wm_out_path)
        
        logger.info(f"Saved masked outputs: {gm_out_path.name}, {wm_out_path.name}")

    def run(self):
        """Main execution flow."""
        logger.info(f"Starting processing for subject: {self.subject_id}")
        
        # 1. Find anat file
        dseg_path = self._get_anat_files()
        
        # 2. Create masks
        gm_mask, wm_mask, _, _ = self.create_tissue_masks(dseg_path)
        
        # 3. Find functional files
        func_files = self._get_func_files()
        
        if not func_files:
            logger.warning("No functional files to process.")
            return

        # 4. Process each run
        for func_path in func_files:
            self.process_run(func_path, gm_mask, wm_mask)
            
        logger.info(f"Completed processing for {self.subject_id}")

def main():
    parser = argparse.ArgumentParser(description="Tissue-specific spatial smoothing for fMRI data.")
    
    parser.add_argument("--sub_id", required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument("--fmriprep_dir", required=True, help="Path to fMRIPrep/BIDS derivative directory containing anat/dseg")
    parser.add_argument("--xcpd_dir", required=True, help="Path to XCP-D or functional derivative directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    generator = MaskGenerator(
        subject_id=args.sub_id,
        fmriprep_dir=args.fmriprep_dir,
        xcpd_dir=args.xcpd_dir,
        output_dir=args.output_dir
    )
    
    generator.run()

if __name__ == "__main__":
    main()
