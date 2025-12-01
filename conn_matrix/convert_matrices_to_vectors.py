#!/usr/bin/env python3
"""
Convert functional connectivity matrices to vectors for machine learning analysis.

This script converts GG (lower triangle), GW (flatten), and WW (lower triangle) 
matrices to vectors, then concatenates them according to subject list order to create
a subject × features matrix.

Reference MATLAB logic:
    FCvector = tril(FCmatrix, -1);                  % 提取严格下三角（对角线以下） 
    FCvector_WW = FCvector(tril(true(m), -1));      % 只保留下三角中的非零位置（即有效连接） 
    FCvector = reshape(FCmatrix, 1, []); 
    FCvector_GW = FCvector; 
    FCvector = tril(FCmatrix, -1); 
    FCvector_GG = FCvector(tril(true(m), -1));

Usage:
    python convert_matrices_to_vectors.py --input_path /path/to/matrices --sublist_file /path/to/sublist.txt --output_path /path/to/output
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MatrixToVectorConverter:
    """Convert FC matrices to vectors following MATLAB reference logic."""
    
    def __init__(self, input_path: str, sublist_file: str, output_path: str, dataset_name: str):
        """
        Initialize the converter.
        
        Args:
            input_path: Path to directory containing individual FC matrices
            sublist_file: Path to subject list file (sublist.txt)
            output_path: Path to output directory for vector matrices
            dataset_name: Name of the dataset (e.g., ABCD, CCNP, HCPD, PNC, EFNY)
        """
        self.input_path = Path(input_path)
        self.sublist_file = Path(sublist_file)
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Subject list: {self.sublist_file}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Dataset: {self.dataset_name}")
    
    def load_subject_list(self) -> List[str]:
        """
        Load subject list from file.
        
        Returns:
            List of subject IDs in order
        """
        try:
            with open(self.sublist_file, 'r') as f:
                subjects = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(subjects)} subjects from {self.sublist_file}")
            return subjects
        except Exception as e:
            logger.error(f"Failed to load subject list from {self.sublist_file}: {e}")
            raise
    
    def extract_gg_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Extract GG vector using lower triangle (excluding diagonal)."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"GG matrix must be square, got shape {matrix.shape}")
        
        # 直接提取下三角元素，更简单
        m = matrix.shape[0]
        lower_mask = np.tril(np.ones((m, m), dtype=bool), k=-1)
        vector = matrix[lower_mask]
        
        return vector
    
    def extract_gw_vector(self, matrix: np.ndarray) -> np.ndarray:
        """
        Extract GW vector by flattening the entire matrix.
        MATLAB: FCvector = reshape(FCmatrix, 1, []); FCvector_GW = FCvector;
        
        Args:
            matrix: GW matrix
            
        Returns:
            Flattened vector of all matrix elements
        """
        # Flatten the entire matrix (row-major order, like MATLAB)
        vector = matrix.flatten()
        
        logger.debug(f"GW matrix {matrix.shape} -> vector {vector.shape}")
        return vector
    
    def extract_ww_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Extract WW vector using lower triangle (excluding diagonal)."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"WW matrix must be square, got shape {matrix.shape}")
        
        # 直接提取下三角元素，更简单
        m = matrix.shape[0]
        lower_mask = np.tril(np.ones((m, m), dtype=bool), k=-1)
        vector = matrix[lower_mask]
        
        return vector
    
    def load_subject_matrices(self, subject_id: str) -> Dict[str, np.ndarray]:
        """
        Load all matrix types for a subject.
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Dictionary with matrix types as keys ('GG', 'GW', 'WW')
        """
        subject_dir = self.input_path / subject_id
        matrices = {}
        
        if not subject_dir.exists():
            logger.warning(f"Subject directory not found: {subject_dir}")
            return matrices
        
        # Look for matrix files
        matrix_patterns = {
            'GG': f"{subject_id}_GG_FC.npy",
            'GW': f"{subject_id}_GW_FC.npy", 
            'WW': f"{subject_id}_WW_FC.npy"
        }
        
        for matrix_type, pattern in matrix_patterns.items():
            matrix_files = list(subject_dir.glob(pattern))
            if matrix_files:
                try:
                    matrix = np.load(matrix_files[0])
                    matrices[matrix_type] = matrix
                    logger.debug(f"Loaded {matrix_type} matrix for {subject_id}: {matrix.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {matrix_type} matrix for {subject_id}: {e}")
            else:
                logger.warning(f"No {matrix_type} matrix found for {subject_id} (pattern: {pattern})")
        
        return matrices
    
    def convert_matrices_to_vectors(self, matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert matrices to vectors using appropriate methods.
        
        Args:
            matrices: Dictionary of matrices by type
            
        Returns:
            Dictionary of vectors by type
        """
        vectors = {}
        
        if 'GG' in matrices:
            vectors['GG'] = self.extract_gg_vector(matrices['GG'])
        
        if 'GW' in matrices:
            vectors['GW'] = self.extract_gw_vector(matrices['GW'])
        
        if 'WW' in matrices:
            vectors['WW'] = self.extract_ww_vector(matrices['WW'])
        
        return vectors
    
    def concatenate_vectors(self, vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate all vectors into a single feature vector.
        
        Args:
            vectors: Dictionary of vectors by type
            
        Returns:
            Concatenated feature vector
        """
        vector_list = []
        
        # Concatenate in consistent order: GG, GW, WW
        for matrix_type in ['GG', 'GW', 'WW']:
            if matrix_type in vectors:
                vector_list.append(vectors[matrix_type])
        
        if not vector_list:
            raise ValueError("No vectors to concatenate")
        
        concatenated = np.concatenate(vector_list)
        logger.debug(f"Concatenated vectors: {concatenated.shape}")
        return concatenated
    
    def process_all_subjects(self, subjects: List[str]) -> Dict[str, np.ndarray]:
        """
        Process all subjects and create feature matrix.
        
        Args:
            subjects: List of subject IDs in order
            
        Returns:
            Dictionary with subject IDs as keys and feature vectors as values
        """
        subject_features = {}
        valid_subjects = []
        
        for i, subject_id in enumerate(subjects):
            logger.info(f"Processing subject {i+1}/{len(subjects)}: {subject_id}")
            
            # Load matrices for this subject
            matrices = self.load_subject_matrices(subject_id)
            if not matrices:
                logger.warning(f"Skipping subject {subject_id}: no matrices found")
                continue
            
            # Convert to vectors
            try:
                vectors = self.convert_matrices_to_vectors(matrices)
                if not vectors:
                    logger.warning(f"Skipping subject {subject_id}: could not convert matrices to vectors")
                    continue
                
                # Concatenate vectors
                feature_vector = self.concatenate_vectors(vectors)
                subject_features[subject_id] = feature_vector
                valid_subjects.append(subject_id)
                
                logger.info(f"Processed {subject_id}: feature vector shape {feature_vector.shape}")
                
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(valid_subjects)} out of {len(subjects)} subjects")
        return subject_features, valid_subjects
    
    def create_feature_matrix(self, subject_features: Dict[str, np.ndarray], subjects: List[str]) -> np.ndarray:
        """
        Create subject × features matrix in the order specified by subjects list.
        
        Args:
            subject_features: Dictionary of feature vectors by subject
            subjects: Ordered list of subject IDs
            
        Returns:
            Subject × features matrix
        """
        # Filter subjects that have features
        valid_subjects = [s for s in subjects if s in subject_features]
        
        if not valid_subjects:
            raise ValueError("No subjects with valid features")
        
        # Get feature dimension from first valid subject
        feature_dim = subject_features[valid_subjects[0]].shape[0]
        
        # Create matrix: subjects × features
        n_subjects = len(valid_subjects)
        feature_matrix = np.zeros((n_subjects, feature_dim))
        
        for i, subject_id in enumerate(valid_subjects):
            feature_matrix[i, :] = subject_features[subject_id]
        
        logger.info(f"Created feature matrix: {feature_matrix.shape} (subjects × features)")
        return feature_matrix, valid_subjects
    
    def save_results(self, feature_matrix: np.ndarray, subjects: List[str], output_prefix: str = "fc_vectors"):
        """
        Save the feature matrix and subject list.
        
        Args:
            feature_matrix: Subject × features matrix
            subjects: Ordered list of subject IDs
            output_prefix: Prefix for output files
        """
        # Save feature matrix
        matrix_file = self.output_path / f"{output_prefix}.npy"
        np.save(matrix_file, feature_matrix)
        logger.info(f"Saved feature matrix to {matrix_file}")
        
        # Save subject list
        subjects_file = self.output_path / f"{output_prefix}_subjects.txt"
        with open(subjects_file, 'w') as f:
            for subject_id in subjects:
                f.write(f"{subject_id}\n")
        logger.info(f"Saved subject list to {subjects_file}")
        
        # Save feature dimensions info
        info_file = self.output_path / f"{output_prefix}_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Feature matrix shape: {feature_matrix.shape}\n")
            f.write(f"Number of subjects: {len(subjects)}\n")
            f.write(f"Number of features: {feature_matrix.shape[1]}\n")
            f.write(f"Feature types: GG (lower triangle), GW (flattened), WW (lower triangle)\n")
        logger.info(f"Saved feature info to {info_file}")
        
        # Also save as CSV for easy inspection
        csv_file = self.output_path / f"{output_prefix}.csv"
        df = pd.DataFrame(feature_matrix, index=subjects)
        df.to_csv(csv_file)
        logger.info(f"Saved feature matrix to {csv_file}")
    
    def run(self):
        """Execute the complete conversion pipeline."""
        logger.info(f"Starting matrix-to-vector conversion for {self.dataset_name}")
        
        # Load subject list
        subjects = self.load_subject_list()
        
        # Process all subjects
        subject_features, valid_subjects = self.process_all_subjects(subjects)
        
        if not subject_features:
            logger.error("No subjects with valid features found")
            return
        
        # Create feature matrix
        feature_matrix, final_subjects = self.create_feature_matrix(subject_features, valid_subjects)
        
        # Save results
        self.save_results(feature_matrix, final_subjects, f"{self.dataset_name}_fc_vectors")
        
        logger.info("Matrix-to-vector conversion completed successfully")


def main():
    """Main function to run the matrix-to-vector conversion."""
    parser = argparse.ArgumentParser(
        description="Convert FC matrices to vectors for machine learning analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert ABCD matrices to vectors
    python convert_matrices_to_vectors.py --input_path /path/to/ABCD/fc_matrix/individual --sublist_file /path/to/sublist.txt --output_path /path/to/ABCD/fc_vector --dataset_name ABCD
    
    # Convert with custom subject list
    python convert_matrices_to_vectors.py --input_path /path/to/matrices --sublist_file subjects.txt --output_path /path/to/output --dataset_name CUSTOM
        """
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to directory containing individual FC matrices (e.g., /path/to/dataset/fc_matrix/individual)"
    )
    
    parser.add_argument(
        "--sublist_file",
        type=str,
        required=True,
        help="Path to subject list file (sublist.txt)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory for vector matrices (e.g., /path/to/dataset/fc_vector)"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=['ABCD', 'CCNP', 'HCPD', 'PNC', 'EFNY', 'CUSTOM'],
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
    
    # Create and run the converter
    converter = MatrixToVectorConverter(
        args.input_path, 
        args.sublist_file, 
        args.output_path, 
        args.dataset_name
    )
    converter.run()


if __name__ == "__main__":
    main()