#!/usr/bin/env python3
"""
Compute group average functional connectivity matrices from individual subject matrices.

This script loads individual FC matrices (GG, GW, WW) from a specified directory,
computes group averages for each matrix type, and saves the results along with
visualizations.

Usage:
    python compute_group_avg_fc.py --input_path /path/to/individual_z --dataset_name ABCD
Example:
    python compute_group_avg_fc.py --input_path /ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/data/ABCD/fc_matrix/individual_z --dataset_name ABCD
    python compute_group_avg_fc.py --input_path /ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/data/CCNP/fc_matrix/individual_z --dataset_name CCNP
    python compute_group_avg_fc.py --input_path /ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction/data/HCPD/fc_matrix/individual_z --dataset_name HCPD
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GroupAverageFC:
    """Compute group average functional connectivity matrices."""
    
    def __init__(self, input_path: str, dataset_name: str):
        """
        Initialize the group average calculator.
        
        Args:
            input_path: Path to directory containing individual FC matrices
            dataset_name: Name of the dataset (e.g., ABCD, CCNP, HCPD, PNC, EFNY)
        """
        self.input_path = Path(input_path)
        self.dataset_name = dataset_name
        self.dataset_root = self.input_path.parent.parent
        
        # Define output paths following ABCD structure
        self.group_avg_dir = self.dataset_root / "fc_matrix" / "group_avg"
        self.figs_dir = self.dataset_root / "figs"
        
        # Create output directories if they don't exist
        self.group_avg_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Group average output: {self.group_avg_dir}")
        logger.info(f"Figures output: {self.figs_dir}")
    
    def load_individual_matrices(self) -> Dict[str, List[np.ndarray]]:
        """
        Load all individual FC matrices from the input directory.
        
        Returns:
            Dictionary mapping matrix types ('GG', 'GW', 'WW') to lists of matrices
        """
        matrices = {'GG': [], 'GW': [], 'WW': []}
        
        # Find all subject directories
        subject_dirs = [d for d in self.input_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(subject_dirs)} subject directories")
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            
            # Look for all matrix types
            for matrix_type in ['GG', 'GW', 'WW']:
                # Search for matrix files with this type
                pattern = f"{subject_id}_{matrix_type}_FC_Z.npy"
                matrix_files = list(subject_dir.glob(pattern))
                
                if matrix_files:
                    try:
                        matrix = np.load(matrix_files[0])
                        matrices[matrix_type].append(matrix)
                        logger.debug(f"Loaded {matrix_type} matrix for {subject_id}: shape {matrix.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to load matrix for {subject_id} {matrix_type}: {e}")
        
        # Log summary statistics
        for matrix_type, matrix_list in matrices.items():
            logger.info(f"Loaded {len(matrix_list)} {matrix_type} matrices")
        
        return matrices
    
    def compute_group_averages(self, matrices: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Compute group average matrices for each type.
        
        Args:
            matrices: Dictionary of matrix lists by type
            
        Returns:
            Dictionary mapping matrix types to their group averages
        """
        group_averages = {}
        
        for matrix_type, matrix_list in matrices.items():
            if not matrix_list:
                logger.warning(f"No matrices found for type {matrix_type}")
                continue
            
            # Stack all matrices and compute element-wise average
            try:
                matrix_stack = np.stack(matrix_list)
                group_avg = np.mean(matrix_stack, axis=0)
                group_averages[matrix_type] = group_avg
                
                logger.info(f"Computed group average for {matrix_type}: shape {group_avg.shape}")
                logger.info(f"  Mean: {np.mean(group_avg):.4f}, Std: {np.std(group_avg):.4f}")
                logger.info(f"  Min: {np.min(group_avg):.4f}, Max: {np.max(group_avg):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to compute group average for {matrix_type}: {e}")
        
        return group_averages
    
    def save_group_averages(self, group_averages: Dict[str, np.ndarray]):
        """
        Save group average matrices to files.
        
        Args:
            group_averages: Dictionary of group average matrices
        """
        for matrix_type, group_avg in group_averages.items():
            output_file = self.group_avg_dir / f"group_avg_{matrix_type}_FC_Z.npy"
            try:
                np.save(output_file, group_avg)
                logger.info(f"Saved group average {matrix_type} to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save group average {matrix_type}: {e}")
    
    def create_heatmaps(self, group_averages: Dict[str, np.ndarray]):
        """
        Create and save heatmap visualizations for group averages.
        
        Args:
            group_averages: Dictionary of group average matrices
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        for matrix_type, group_avg in group_averages.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            im = ax.imshow(group_avg, cmap='coolwarm', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Fisher Z-transformed FC', rotation=270, labelpad=20)
            
            # Set labels and title
            ax.set_xlabel('ROI')
            ax.set_ylabel('ROI')
            ax.set_title(f'Group Average Functional Connectivity - {matrix_type}\n{self.dataset_name} Dataset')
            
            # Add text annotation with statistics
            stats_text = f"Mean: {np.mean(group_avg):.3f}\nStd: {np.std(group_avg):.3f}\nN = {len(group_avg)}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save figure
            output_file = self.figs_dir / f"group_avg_{matrix_type}_heatmap.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved heatmap for {matrix_type} to {output_file}")
    
    def run(self):
        """Execute the complete group average pipeline."""
        logger.info(f"Starting group average computation for {self.dataset_name}")
        
        # Load individual matrices
        matrices = self.load_individual_matrices()
        
        # Check if we found any matrices
        total_matrices = sum(len(matrix_list) for matrix_list in matrices.values())
        if total_matrices == 0:
            logger.error("No matrices found in the input directory")
            return
        
        # Compute group averages
        group_averages = self.compute_group_averages(matrices)
        
        # Save results
        self.save_group_averages(group_averages)
        
        # Create visualizations
        self.create_heatmaps(group_averages)
        
        logger.info("Group average computation completed successfully")


def main():
    """Main function to run the group average computation."""
    parser = argparse.ArgumentParser(
        description="Compute group average functional connectivity matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute group averages for ABCD dataset
    python compute_group_avg_fc.py --input_path /path/to/ABCD/fc_matrix/individual_z --dataset_name ABCD
    
    # Compute group averages for CCNP dataset
    python compute_group_avg_fc.py --input_path /path/to/CCNP/fc_matrix/individual_z --dataset_name CCNP
        """
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to directory containing individual FC matrices (e.g., /path/to/dataset/fc_matrix/individual_z)"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=['ABCD', 'CCNP', 'HCPD', 'PNC', 'EFNY'],
        help="Name of the dataset"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create and run the group average calculator
    calculator = GroupAverageFC(args.input_path, args.dataset_name)
    calculator.run()


if __name__ == "__main__":
    main()