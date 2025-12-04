#!/usr/bin/env python3
"""
Unified script for processing fMRI data: generate masks, compute FC, and apply Fisher Z transform.
Supports multiple datasets (ABCD, CCNP, EFNY, HCPD, PNC) with dataset-specific handling.
"""

import argparse
import csv
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
import sys
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Unified processor for fMRI datasets with dataset-specific handling."""
    
    # Dataset-specific configurations
    DATASET_CONFIGS = {
        'ABCD': {
            'gm_label': 2, 'wm_label': 3,  # ABCD: GM=2, WM=3
            'run_order': ['1', '2'],  # Run numbers only
            'fd_summary_file': 'rest_fd_summary.csv',
            'func_pattern': 'task-rest_run-{run}_space-MNI152NLin6Asym_desc-denoised_bold.nii.gz'
        },
        'CCNP': {
            'gm_label': 1, 'wm_label': 2,  # XCPD standard: GM=1, WM=2
            'run_order': ['01', '02'],  # Run numbers only
            'fd_summary_file': 'rest_fd_summary.csv',
            'func_pattern': 'task-rest_run-{run}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz'
        },
        'EFNY': {
            'gm_label': 1, 'wm_label': 2,  # XCPD standard: GM=1, WM=2
            'run_order': ['1', '2', '3', '4'],  # Run numbers only
            'fd_summary_file': 'rest_fd_summary.csv',
            'func_pattern': 'task-rest_run-{run}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz'
        },
        'HCPD': {
            'gm_label': 1, 'wm_label': 2,  # XCPD standard: GM=1, WM=2
            'run_order': ['REST1_acq-AP', 'REST1_acq-PA', 'REST2_acq-AP', 'REST2_acq-PA'],
            'fd_summary_file': 'rest_fd_summary.csv',
            'func_pattern': 'task-{run}_space-MNI152NLin6Asym_desc-denoised_bold.nii.gz'
        },
        'PNC': {
            'gm_label': 1, 'wm_label': 2,  # XCPD standard: GM=1, WM=2
            'run_order': ['singleband'],  # Just the acquisition part
            'fd_summary_file': 'rest_fd_summary.csv',
            'func_pattern': 'task-rest_acq-{run}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz'
        }
    }
    
    def __init__(self, dataset_name: str, subject_id: str, dataset_path: str, 
                 mask_output_dir: str, fc_output_dir: str, z_output_dir: str):
        self.dataset_name = dataset_name.upper()
        self.subject_id = subject_id
        self.dataset_path = Path(dataset_path)
        self.mask_output_dir = Path(mask_output_dir)
        self.fc_output_dir = Path(fc_output_dir)
        self.z_output_dir = Path(z_output_dir)
        
        if self.dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {list(self.DATASET_CONFIGS.keys())}")
        
        self.config = self.DATASET_CONFIGS[self.dataset_name]
        
        # Setup paths based on dataset-specific structure
        self._setup_dataset_paths()
    
    def _setup_dataset_paths(self):
        """Setup dataset-specific paths for fmriprep and xcpd directories."""
        if self.dataset_name == 'PNC':
            # PNC has dual paths
            self.pnc_paths = [
                Path('/ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/datasets/PNC'),
                Path('/ibmgpfs/cuizaixu_lab/congjing/WM_prediction/PNC/results')
            ]
            self.fmriprep_dirs = []
            self.xcpd_dirs = []
            
            for path in self.pnc_paths:
                fmriprep_path = path / 'fmriprep'
                xcpd_path = path / 'xcpd' if (path / 'xcpd').exists() else path / 'results' / 'xcpd'
                # Add paths even if they don't exist during testing - will be checked at runtime
                self.fmriprep_dirs.append(fmriprep_path)
                self.xcpd_dirs.append(xcpd_path)
            
            # Filter out non-existent paths and warn
            self.fmriprep_dirs = [p for p in self.fmriprep_dirs if p.exists()]
            self.xcpd_dirs = [p for p in self.xcpd_dirs if p.exists()]
            
            if not self.fmriprep_dirs:
                logger.warning(f"No fmriprep directories found for PNC in {self.pnc_paths}")
            if not self.xcpd_dirs:
                logger.warning(f"No xcpd directories found for PNC in {self.pnc_paths}")
                
        elif self.dataset_name == 'HCPD':
            # HCPD specific paths
            base_path = Path('/ibmgpfs/cuizaixu_lab/zhaoshaoling/MSC_data/HCPD/code_xcpd0.7.1rc5_hcpMiniPrepData/final2025/data')
            self.fmriprep_dirs = [base_path / 'xcpd0.7.1rc5' / 'bids']  # dseg files are in bids
            self.xcpd_dirs = [base_path / 'xcpd0.7.1rc5' / 'step_2nd_24PcsfGlobal']
            
        elif self.dataset_name == 'CCNP':
            # CCNP has dual paths
            self.ccnp_paths = [
                Path('/ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/datasets/CCNP'),
                Path('/ibmgpfs/cuizaixu_lab/congjing/WM_prediction/CCNP/results')
            ]
            self.fmriprep_dirs = []
            self.xcpd_dirs = []
            
            for path in self.ccnp_paths:
                fmriprep_path = path / 'fmriprep'
                xcpd_path = path / 'xcpd' if (path / 'xcpd').exists() else path / 'results' / 'xcpd'
                # Add paths even if they don't exist during testing - will be checked at runtime
                self.fmriprep_dirs.append(fmriprep_path)
                self.xcpd_dirs.append(xcpd_path)
            
            # Filter out non-existent paths and warn
            self.fmriprep_dirs = [p for p in self.fmriprep_dirs if p.exists()]
            self.xcpd_dirs = [p for p in self.xcpd_dirs if p.exists()]
            
            if not self.fmriprep_dirs:
                logger.warning(f"No fmriprep directories found for CCNP in {self.ccnp_paths}")
            if not self.xcpd_dirs:
                logger.warning(f"No xcpd directories found for CCNP in {self.ccnp_paths}")
                
        elif self.dataset_name == 'EFNY':
            # EFNY specific paths
            base_path = Path('/ibmgpfs/cuizaixu_lab/congjing/WM_prediction/EFNY/results')
            self.fmriprep_dirs = [base_path / 'fmriprep']
            self.xcpd_dirs = [base_path / 'xcpd']
            
        else:
            # Fallback to original logic for other datasets
            self.fmriprep_dirs = [self.dataset_path / 'fmriprep'] if (self.dataset_path / 'fmriprep').exists() else [self.dataset_path]
            self.xcpd_dirs = [self.dataset_path / 'xcpd'] if (self.dataset_path / 'xcpd').exists() else [self.dataset_path]
        
        # Set table directory (for FD summary files)
        if self.dataset_name in ['PNC', 'CCNP']:
            # For datasets with dual paths, use the first path for tables
            base_table_path = self.pnc_paths[0] if self.dataset_name == 'PNC' else self.ccnp_paths[0]
            self.table_dir = base_table_path / 'table'
        else:
            self.table_dir = self.dataset_path / 'table'
        
    def get_valid_runs(self) -> List[str]:
        """Get valid runs for the subject based on FD summary file."""
        fd_file = self.table_dir / self.config['fd_summary_file']
        if not fd_file.exists():
            logger.warning(f"FD summary file not found: {fd_file}, using all runs")
            return self.config['run_order']
        
        valid_runs = []
        subject_found = False
        
        with open(fd_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['subid'] == self.subject_id:
                    subject_found = True
                    
                    # Handle different dataset formats
                    if self.dataset_name == 'CCNP':
                        # CCNP format: rest1_valid, rest2_valid
                        for i, run_num in enumerate(self.config['run_order'], 1):
                            col_name = f"rest{i}_valid"
                            if row.get(col_name, '0') == '1':
                                # Convert run number back to full run name
                                run_name = f"rest_run-{run_num}"
                                valid_runs.append(run_name)
                            else:
                                logger.info(f"Run rest_run-{run_num} marked as invalid for {self.subject_id}")
                    
                    elif self.dataset_name == 'HCPD':
                        # HCPD format: rest1_AP_valid, rest1_PA_valid, etc.
                        for run in self.config['run_order']:
                            # Convert REST1_acq-AP -> rest1_AP (remove the "acq-" part)
                            run_parts = run.split('_')
                            if len(run_parts) == 2 and run_parts[0].startswith('REST'):
                                rest_num = run_parts[0].replace('REST', '')
                                acq_type = run_parts[1].replace('acq-', '')  # Remove "acq-" prefix
                                col_name = f"rest{rest_num}_{acq_type}_valid"
                                col_value = row.get(col_name, '0')
                                logger.info(f"Checking run {run}: looking for column '{col_name}', found value = '{col_value}' (type: {type(col_value)})")
                                if col_value.strip() == '1':
                                    # For HCPD, the actual run name in files is REST1_acq-AP
                                    valid_runs.append(run)
                                    logger.info(f"Run {run} is valid for {self.subject_id}")
                                else:
                                    logger.info(f"Run {run} marked as invalid for {self.subject_id}")
                            else:
                                logger.warning(f"Unexpected run format for HCPD: {run}")
                    
                    else:  # ABCD, EFNY, PNC
                        # Standard format: rest{run_number}_valid
                        for i, run_num in enumerate(self.config['run_order'], 1):
                            col_name = f"rest{i}_valid"
                            if row.get(col_name, '0') == '1':
                                if self.dataset_name == 'ABCD':
                                    run_name = f"rest_run-{run_num}"
                                elif self.dataset_name == 'EFNY':
                                    run_name = f"rest_run-{run_num}"
                                elif self.dataset_name == 'PNC':
                                    run_name = f"rest_acq-{run_num}"
                                valid_runs.append(run_name)
                            else:
                                logger.info(f"Run marked as invalid for {self.subject_id}")
                    break
        
        if not subject_found:
            logger.warning(f"Subject {self.subject_id} not found in FD summary, using all runs")
            return self.config['run_order']
        
        logger.info(f"Valid runs for {self.subject_id}: {valid_runs}")
        return valid_runs
    
    def find_dseg_file(self) -> Path:
        """Find dseg file for the subject across multiple possible paths and naming conventions."""
        
        # Dataset-specific dseg patterns and search paths
        if self.dataset_name == 'PNC':
            patterns = [
                f'{self.subject_id}_ses-PNC1_acq-refaced_space-MNI152NLin6Asym_res-2_dseg.nii.gz',
                f'{self.subject_id}_ses-PNC1_space-MNI152NLin6Asym_res-2_dseg.nii.gz',
                f'{self.subject_id}_space-MNI152NLin6Asym_dseg.nii.gz',
                f'{self.subject_id}*space-MNI152NLin6Asym*dseg.nii.gz'
            ]
        elif self.dataset_name == 'HCPD':
            patterns = [
                f'{self.subject_id}_dseg.nii.gz',
                f'{self.subject_id}_space-MNI152NLin6Asym_dseg.nii.gz',
                f'{self.subject_id}*dseg.nii.gz'
            ]
        elif self.dataset_name == 'CCNP':
            patterns = [
                f'{self.subject_id}_ses-01_space-MNI152NLin6Asym_res-2_dseg.nii.gz',
                f'{self.subject_id}_ses-01_space-MNI152NLin6Asym_dseg.nii.gz',
                f'{self.subject_id}_space-MNI152NLin6Asym_dseg.nii.gz',
                f'{self.subject_id}*space-MNI152NLin6Asym*dseg.nii.gz'
            ]
        elif self.dataset_name == 'EFNY':
            patterns = [
                f'{self.subject_id}_run-1_space-MNI152NLin6Asym_res-2_dseg.nii.gz',
                f'{self.subject_id}_space-MNI152NLin6Asym_res-2_dseg.nii.gz',
                f'{self.subject_id}_space-MNI152NLin6Asym_dseg.nii.gz',
                f'{self.subject_id}*space-MNI152NLin6Asym*dseg.nii.gz'
            ]
        else:
            patterns = [f'{self.subject_id}*space-MNI152NLin6Asym*dseg.nii.gz']
        
        # Search in all fmriprep directories
        for fmriprep_dir in self.fmriprep_dirs:
            # Try specific anatomical paths first
            search_paths = [
                fmriprep_dir / self.subject_id / 'anat',
                fmriprep_dir / self.subject_id / 'ses-01' / 'anat',
                fmriprep_dir / self.subject_id / 'ses-baselineYear1Arm1' / 'anat',
                fmriprep_dir / self.subject_id / 'ses-PNC1' / 'anat',
            ]
            
            for search_path in search_paths:
                if search_path.exists():
                    for pattern in patterns:
                        dseg_files = list(search_path.glob(pattern))
                        if dseg_files:
                            logger.info(f"Found dseg file: {dseg_files[0]} in {search_path}")
                            return dseg_files[0]
            
            # Try recursive search in this fmriprep directory
            for pattern in patterns:
                dseg_files = list(fmriprep_dir.rglob(pattern))
                if dseg_files:
                    logger.info(f"Found dseg file via recursive search: {dseg_files[0]} in {fmriprep_dir}")
                    return dseg_files[0]
        
        raise FileNotFoundError(f"Could not find dseg file for {self.subject_id} in any of the configured paths")
    
    def find_functional_files(self, valid_runs: List[str]) -> List[Path]:
        """Find functional files for valid runs across multiple xcpd directories."""
        func_files = []
        
        # Dataset-specific xcpd step paths
        if self.dataset_name == 'PNC':
            xcpd_steps = ['step_2nd_24PcsfGlobal']
        elif self.dataset_name == 'HCPD':
            xcpd_steps = ['step_2nd_24PcsfGlobal']
        elif self.dataset_name == 'CCNP':
            xcpd_steps = ['step_2nd_24PcsfGlobal']
        elif self.dataset_name == 'EFNY':
            xcpd_steps = ['']
        else:
            xcpd_steps = ['']
        
        for run_name in valid_runs:
            # Construct search pattern based on dataset
            if self.dataset_name == 'HCPD':
                pattern = self.config['func_pattern'].format(run=run_name)
            else:
                pattern = self.config['func_pattern'].format(run=run_name.replace('rest_run-', '').replace('rest_acq-', ''))
            
            found = False
            
            # Search in all xcpd directories
            for xcpd_dir in self.xcpd_dirs:
                for step in xcpd_steps:
                    # Construct search paths
                    if step:
                        search_paths = [
                            xcpd_dir / step / self.subject_id / 'func',
                            xcpd_dir / step / self.subject_id / 'ses-01' / 'func',
                            xcpd_dir / step / self.subject_id / 'ses-baselineYear1Arm1' / 'func',
                            xcpd_dir / step / self.subject_id / 'ses-PNC1' / 'func',
                        ]
                    else:
                        search_paths = [
                            xcpd_dir / self.subject_id / 'func',
                            xcpd_dir / self.subject_id / 'ses-01' / 'func',
                            xcpd_dir / self.subject_id / 'ses-baselineYear1Arm1' / 'func',
                            xcpd_dir / self.subject_id / 'ses-PNC1' / 'func',
                        ]
                    
                    for search_path in search_paths:
                        if search_path.exists():
                            # Use more flexible glob pattern
                            files = list(search_path.glob(f'*{pattern}'))
                            if not files:
                                # Try alternative pattern matching
                                files = list(search_path.glob(f'{self.subject_id}*{pattern}'))
                            if files:
                                func_files.append(files[0])
                                found = True
                                logger.info(f"Found functional file: {files[0]} in {search_path}")
                                break
                    
                    if found:
                        break
                
                if found:
                    break
            
            if not found:
                # Try recursive search in all xcpd directories
                for xcpd_dir in self.xcpd_dirs:
                    # Always include subject ID in the search pattern to avoid matching other subjects
                    # First try starting with subject ID (standard BIDS)
                    files = list(xcpd_dir.rglob(f'{self.subject_id}*{pattern}'))
                    
                    if not files:
                        # Try matching with subject ID anywhere in the name
                        files = list(xcpd_dir.rglob(f'*{self.subject_id}*{pattern}'))
                    
                    if files:
                        func_files.append(files[0])
                        logger.info(f"Found functional file via recursive search: {files[0]} in {xcpd_dir}")
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Could not find functional file for run {run_name} with pattern {pattern}")
        
        return sorted(func_files)
    
    def create_tissue_masks(self, dseg_path: Path, func_shape: tuple = None, func_affine: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create GM and WM masks from dseg file, optionally resampling to match functional data."""
        logger.info(f"Loading dseg file: {dseg_path}")
        dseg_img = nib.load(dseg_path)
        dseg_data = dseg_img.get_fdata()
        
        # Remove any extra dimensions (e.g., from 4D to 3D)
        if dseg_data.ndim > 3:
            dseg_data = dseg_data.squeeze()
        
        # Check if resampling is needed
        if func_shape is not None and dseg_data.shape != func_shape[:3]:
            logger.info(f"Resampling dseg from {dseg_data.shape} to match functional data {func_shape[:3]}")
            
            # Create a reference image with the target shape and affine
            from nilearn.image import resample_to_img
            
            # Create a dummy functional image as reference
            if func_affine is None:
                func_affine = dseg_img.affine  # Fallback to original affine
            
            # Create reference image
            ref_data = np.zeros(func_shape[:3])
            ref_img = nib.Nifti1Image(ref_data, func_affine)
            
            # Resample dseg to match functional data
            try:
                import nilearn.image as nil_img
                resampled_dseg_img = nil_img.resample_to_img(dseg_img, ref_img, interpolation='nearest')
                dseg_data = resampled_dseg_img.get_fdata()
                # Remove any extra dimensions after resampling
                if dseg_data.ndim > 3:
                    dseg_data = dseg_data.squeeze()
                logger.info(f"Successfully resampled dseg to shape {dseg_data.shape}")
            except ImportError:
                logger.error("nilearn not available for resampling. Please install nilearn or ensure shapes match.")
                raise ValueError(f"Shape mismatch: dseg {dseg_data.shape} vs func {func_shape[:3]}. "
                                "Please install nilearn for automatic resampling.")
        
        # Create masks based on dataset-specific labels
        gm_mask = (dseg_data == self.config['gm_label']).astype(np.float32)
        wm_mask = (dseg_data == self.config['wm_label']).astype(np.float32)
        
        logger.info(f"Created GM mask with {np.sum(gm_mask)} voxels")
        logger.info(f"Created WM mask with {np.sum(wm_mask)} voxels")
        
        return gm_mask, wm_mask
    
    def apply_masks(self, func_file: Path, gm_mask: np.ndarray, wm_mask: np.ndarray) -> Tuple[Path, Path]:
        """Apply masks to functional data."""
        logger.info(f"Processing functional file: {func_file.name}")
        
        # Load functional data
        func_img = nib.load(func_file)
        func_data = func_img.get_fdata()
        
        # Check if mask and functional data have compatible shapes
        if gm_mask.shape != func_data.shape[:3]:
            logger.warning(f"Mask shape {gm_mask.shape} doesn't match functional data spatial shape {func_data.shape[:3]}")
            logger.warning("This might indicate different resolutions. Attempting to proceed with broadcasting...")
            
            # Try to handle shape mismatch by using broadcasting
            # This assumes the mask covers the same brain region but at different resolution
            try:
                gm_masked = func_data * gm_mask[..., np.newaxis]
                wm_masked = func_data * wm_mask[..., np.newaxis]
            except ValueError as e:
                logger.error(f"Cannot broadcast mask to functional data: {e}")
                raise ValueError(f"Incompatible shapes: mask {gm_mask.shape} vs func {func_data.shape[:3]}. "
                                "Please ensure masks are created from the same space as functional data.")
        else:
            # Normal case - shapes match
            gm_masked = func_data * gm_mask[..., np.newaxis]
            wm_masked = func_data * wm_mask[..., np.newaxis]
        
        # Save masked data
        output_dir = self.mask_output_dir / self.subject_id / 'func'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = func_file.name.replace('.nii.gz', '')
        gm_path = output_dir / f'{prefix}_GM_masked.nii.gz'
        wm_path = output_dir / f'{prefix}_WM_masked.nii.gz'
        
        nib.save(nib.Nifti1Image(gm_masked, func_img.affine, func_img.header), gm_path)
        nib.save(nib.Nifti1Image(wm_masked, func_img.affine, func_img.header), wm_path)
        
        return gm_path, wm_path
    
    def extract_timeseries(self, atlas_path: Path, masked_files: List[Path]) -> np.ndarray:
        """Extract time series from masked files using atlas."""
        logger.info(f"Loading atlas: {atlas_path}")
        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata().astype(int)
        
        # Get unique labels (exclude background)
        labels = np.unique(atlas_data)
        labels = labels[labels > 0]
        logger.info(f"Found {len(labels)} regions in atlas")
        
        all_timeseries = []
        
        for masked_file in masked_files:
            logger.info(f"Extracting timeseries from: {masked_file.name}")
            
            # Load masked data
            masked_img = nib.load(masked_file)
            masked_data = masked_img.get_fdata()
            n_timepoints = masked_data.shape[-1]
            
            # Extract timeseries for each region
            timeseries = np.zeros((len(labels), n_timepoints))
            
            for i, label in enumerate(labels):
                roi_mask = (atlas_data == label)
                
                if not np.any(roi_mask):
                    logger.warning(f"Label {label} has no voxels")
                    continue
                
                # Extract mean timeseries for this region
                roi_data = masked_data[roi_mask]
                timeseries[i, :] = np.mean(roi_data, axis=0)
            
            all_timeseries.append(timeseries)
        
        # Concatenate along time axis
        if all_timeseries:
            final_timeseries = np.concatenate(all_timeseries, axis=1)
            logger.info(f"Final timeseries shape: {final_timeseries.shape}")
            return final_timeseries
        else:
            raise ValueError("No timeseries extracted")
    
    def compute_functional_connectivity(self, gm_timeseries: np.ndarray, wm_timeseries: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute functional connectivity matrices."""
        logger.info("Computing functional connectivity matrices...")
        
        # Set error handling for correlation computation
        np.seterr(invalid='ignore')
        
        # GM-GM connectivity
        logger.info("Computing GM-GM FC...")
        gg_fc = np.corrcoef(gm_timeseries)
        gg_fc = np.nan_to_num(gg_fc)
        
        # WM-WM connectivity
        logger.info("Computing WM-WM FC...")
        ww_fc = np.corrcoef(wm_timeseries)
        ww_fc = np.nan_to_num(ww_fc)
        
        # GM-WM connectivity
        logger.info("Computing GM-WM FC...")
        combined_ts = np.vstack([gm_timeseries, wm_timeseries])
        combined_fc = np.corrcoef(combined_ts)
        combined_fc = np.nan_to_num(combined_fc)
        
        n_gm = gm_timeseries.shape[0]
        gw_fc = combined_fc[:n_gm, n_gm:]
        
        return {
            'GG_FC': gg_fc,
            'WW_FC': ww_fc,
            'GW_FC': gw_fc
        }
    
    def apply_fisher_z_transform(self, fc_matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Fisher Z transformation to FC matrices."""
        logger.info("Applying Fisher Z transformation...")
        
        z_matrices = {}
        epsilon = 1e-6  # To avoid infinity at r=±1
        
        for name, matrix in fc_matrices.items():
            # Clip correlation values
            matrix_clipped = np.clip(matrix, -1 + epsilon, 1 - epsilon)
            # Apply Fisher Z transform: arctanh(r)
            z_matrix = np.arctanh(matrix_clipped)
            # Handle any remaining NaNs
            z_matrix = np.nan_to_num(z_matrix, nan=0.0)
            
            z_matrices[name + '_Z'] = z_matrix
            logger.info(f"Applied Fisher Z transform to {name}")
        
        return z_matrices
    
    def save_results(self, gm_timeseries: np.ndarray, wm_timeseries: np.ndarray, 
                    fc_matrices: Dict[str, np.ndarray], z_matrices: Dict[str, np.ndarray]):
        """Save all results to appropriate output directories."""
        # Create subject directories in each output location
        mask_subj_dir = self.mask_output_dir / self.subject_id
        fc_subj_dir = self.fc_output_dir / self.subject_id
        z_subj_dir = self.z_output_dir / self.subject_id
        
        mask_subj_dir.mkdir(parents=True, exist_ok=True)
        fc_subj_dir.mkdir(parents=True, exist_ok=True)
        z_subj_dir.mkdir(parents=True, exist_ok=True)
        
        # Save timeseries to mask output directory (along with masked functional files)
        np.save(mask_subj_dir / f'{self.subject_id}_GM_timeseries.npy', gm_timeseries)
        np.save(mask_subj_dir / f'{self.subject_id}_WM_timeseries.npy', wm_timeseries)
        logger.info("Saved timeseries")
        
        # Save FC matrices to FC output directory
        for name, matrix in fc_matrices.items():
            np.save(fc_subj_dir / f'{self.subject_id}_{name}.npy', matrix)
        logger.info("Saved FC matrices")
        
        # Save Z-transformed matrices to Z output directory
        for name, matrix in z_matrices.items():
            np.save(z_subj_dir / f'{self.subject_id}_{name}.npy', matrix)
        logger.info("Saved Z-transformed matrices")
    
    def process_subject(self, gm_atlas_path: str, wm_atlas_path: str):
        """Complete processing pipeline for a single subject."""
        logger.info(f"\n=== Processing {self.subject_id} from {self.dataset_name} ===")
        
        try:
            # Step 1: Get valid runs
            valid_runs = self.get_valid_runs()
            if not valid_runs:
                logger.warning(f"No valid runs found for {self.subject_id}")
                return False
            
            # Step 2: Find necessary files
            dseg_file = self.find_dseg_file()
            func_files = self.find_functional_files(valid_runs)
            
            if not func_files:
                logger.error(f"No functional files found for {self.subject_id}")
                return False
            
            # Step 3: Create tissue masks (with shape matching to functional data)
            # Load one functional file to get shape and affine for resampling
            if func_files:
                sample_func = nib.load(func_files[0])
                gm_mask, wm_mask = self.create_tissue_masks(dseg_file, sample_func.shape, sample_func.affine)
            else:
                gm_mask, wm_mask = self.create_tissue_masks(dseg_file)
            
            # Step 4: Apply masks to functional data
            gm_masked_files = []
            wm_masked_files = []
            
            for func_file in func_files:
                gm_path, wm_path = self.apply_masks(func_file, gm_mask, wm_mask)
                gm_masked_files.append(gm_path)
                wm_masked_files.append(wm_path)
            
            # Step 5: Extract timeseries using atlases
            gm_timeseries = self.extract_timeseries(Path(gm_atlas_path), gm_masked_files)
            wm_timeseries = self.extract_timeseries(Path(wm_atlas_path), wm_masked_files)
            
            # Step 6: Compute functional connectivity
            fc_matrices = self.compute_functional_connectivity(gm_timeseries, wm_timeseries)
            
            # Step 7: Apply Fisher Z transformation
            z_matrices = self.apply_fisher_z_transform(fc_matrices)
            
            # Step 8: Save all results
            self.save_results(gm_timeseries, wm_timeseries, fc_matrices, z_matrices)
            
            logger.info(f"Successfully completed processing for {self.subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {self.subject_id}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified fMRI processing script: generate masks, compute FC, apply Fisher Z transform."
    )
    parser.add_argument("--dataset_name", type=str, required=True,
                       choices=['ABCD', 'CCNP', 'EFNY', 'HCPD', 'PNC'],
                       help="Dataset name")
    parser.add_argument("--subject_id", type=str, required=True,
                       help="Subject ID (e.g., sub-01)")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--mask_output_dir", type=str, required=True,
                       help="Output directory for mask files (mri_data/wm_postproc)")
    parser.add_argument("--fc_output_dir", type=str, required=True,
                       help="Output directory for FC matrices (fc_matrix/individual)")
    parser.add_argument("--z_output_dir", type=str, required=True,
                       help="Output directory for Z-transformed matrices (fc_matrix/individual_z)")
    parser.add_argument("--gm_atlas", type=str, required=True,
                       help="Path to GM atlas (resliced for this dataset)")
    parser.add_argument("--wm_atlas", type=str, required=True,
                       help="Path to WM atlas (resliced for this dataset)")
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = DatasetProcessor(
        dataset_name=args.dataset_name,
        subject_id=args.subject_id,
        dataset_path=args.dataset_path,
        mask_output_dir=args.mask_output_dir,
        fc_output_dir=args.fc_output_dir,
        z_output_dir=args.z_output_dir
    )
    
    success = processor.process_subject(args.gm_atlas, args.wm_atlas)
    
    if success:
        logger.info("Processing completed successfully")
        sys.exit(0)
    else:
        logger.error("Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()