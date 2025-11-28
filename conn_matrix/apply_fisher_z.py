import argparse
import numpy as np
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fisher_z_transform(matrix):
    """
    Applies Fisher Z transformation: z = 0.5 * ln((1+r)/(1-r)) = arctanh(r).
    Handles NaNs and clip values to avoid infinity.
    """
    # Clip correlation values to avoid inf at r=1.0 or r=-1.0
    # Usually r is in [-1, 1].
    epsilon = 1e-6
    matrix_clipped = np.clip(matrix, -1 + epsilon, 1 - epsilon)
    
    z_matrix = np.arctanh(matrix_clipped)
    
    # Handle NaNs (if any original values were NaN)
    # The MATLAB script sets NaNs to 0
    if np.isnan(z_matrix).any():
        logger.warning("NaNs detected in Z-transformed matrix. Replacing with 0.")
        z_matrix = np.nan_to_num(z_matrix, nan=0.0)
        
    return z_matrix

def process_subject(subject_id, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    subj_in_dir = input_dir / subject_id
    subj_out_dir = output_dir / subject_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_process = [
        (f'{subject_id}_GG_FC.npy', 'GG_FC'),
        (f'{subject_id}_WW_FC.npy', 'WW_FC'),
        (f'{subject_id}_GW_FC.npy', 'GW_FC')
    ]
    
    for filename, var_name in files_to_process:
        file_path = subj_in_dir / filename
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Loading {filename}...")
        try:
            fc_matrix = np.load(file_path)
            
            # Apply Fisher Z
            logger.info("Applying Fisher Z transform...")
            z_matrix = fisher_z_transform(fc_matrix)
            
            # Save
            out_filename = filename.replace('.npy', '_Z.npy')
            out_path = subj_out_dir / out_filename
            
            np.save(out_path, z_matrix)
            logger.info(f"Saved {out_filename}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apply Fisher Z transform to FC matrices.")
    parser.add_argument("--sub_id", type=str, required=True, help="Subject ID")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing FC matrices")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Z-transformed matrices")
    
    args = parser.parse_args()
    
    process_subject(args.sub_id, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
