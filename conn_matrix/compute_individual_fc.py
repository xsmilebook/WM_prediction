import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_timeseries(func_img, atlas_data, labels):
    """
    Extracts mean time series for each ROI in the atlas.
    func_img: Nibabel image object (4D)
    atlas_data: 3D numpy array of labels
    labels: List of sorted ROI labels to extract
    
    Returns: (N_rois, N_timepoints)
    """
    func_data = func_img.get_fdata()
    n_timepoints = func_data.shape[-1]
    n_rois = len(labels)
    
    timeseries = np.zeros((n_rois, n_timepoints))
    
    for i, label in enumerate(labels):
        # Create mask for current ROI
        roi_mask = (atlas_data == label)
        
        if not np.any(roi_mask):
            logger.warning(f"Label {label} found in atlas but has no voxels.")
            continue
            
        # Extract data for ROI voxels: (N_voxels, N_timepoints)
        roi_data = func_data[roi_mask]
        
        # Compute mean time series
        timeseries[i, :] = np.mean(roi_data, axis=0)
        
    return timeseries

def compute_fc(subject_id, input_dir, gm_atlas_path, wm_atlas_path, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Define subject-specific directories
    subj_func_dir = input_dir / subject_id / 'func'
    subj_out_dir = output_dir / subject_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Atlases
    logger.info(f"Loading GM Atlas: {gm_atlas_path}")
    gm_atlas_img = nib.load(gm_atlas_path)
    gm_atlas_data = gm_atlas_img.get_fdata().astype(int)
    gm_labels = np.unique(gm_atlas_data)
    gm_labels = gm_labels[gm_labels > 0] # Exclude background
    logger.info(f"Found {len(gm_labels)} GM regions.")

    logger.info(f"Loading WM Atlas: {wm_atlas_path}")
    wm_atlas_img = nib.load(wm_atlas_path)
    wm_atlas_data = wm_atlas_img.get_fdata().astype(int)
    wm_labels = np.unique(wm_atlas_data)
    wm_labels = wm_labels[wm_labels > 0] # Exclude background
    logger.info(f"Found {len(wm_labels)} WM regions.")

    # Find masked functional files
    # Naming convention from generate_mask.py: {prefix}_GM_masked.nii.gz, {prefix}_WM_masked.nii.gz
    # We search for GM masked files and deduce WM ones
    gm_files = sorted(list(subj_func_dir.glob('*_GM_masked.nii.gz')))
    
    if not gm_files:
        logger.error(f"No GM masked files found in {subj_func_dir}")
        sys.exit(1)
        
    all_gm_timeseries = []
    all_wm_timeseries = []
    
    for gm_file in gm_files:
        # Deduce WM file name
        wm_file = gm_file.parent / gm_file.name.replace('_GM_masked.nii.gz', '_WM_masked.nii.gz')
        
        if not wm_file.exists():
            logger.warning(f"WM masked file not found for {gm_file.name}, skipping run.")
            continue
            
        logger.info(f"Processing run: {gm_file.name}")
        
        # Load GM data
        gm_img = nib.load(gm_file)
        gm_ts = extract_timeseries(gm_img, gm_atlas_data, gm_labels)
        all_gm_timeseries.append(gm_ts)
        
        # Load WM data
        wm_img = nib.load(wm_file)
        wm_ts = extract_timeseries(wm_img, wm_atlas_data, wm_labels)
        all_wm_timeseries.append(wm_ts)
        
    # Concatenate time series along time axis (axis 1)
    if not all_gm_timeseries:
        logger.error("No valid runs processed.")
        sys.exit(1)
        
    final_gm_ts = np.concatenate(all_gm_timeseries, axis=1) # (N_gm, Total_Time)
    final_wm_ts = np.concatenate(all_wm_timeseries, axis=1) # (N_wm, Total_Time)
    
    logger.info(f"Final GM Timeseries shape: {final_gm_ts.shape}")
    logger.info(f"Final WM Timeseries shape: {final_wm_ts.shape}")
    
    # Compute Correlations
    # Handle division by zero/invalid values warning
    np.seterr(invalid='ignore')
    
    # 1. GM-GM (GG)
    logger.info("Computing GM-GM FC...")
    gg_fc = np.corrcoef(final_gm_ts)
    gg_fc = np.nan_to_num(gg_fc) # Replace NaNs with 0
    
    # 2. WM-WM (WW)
    logger.info("Computing WM-WM FC...")
    ww_fc = np.corrcoef(final_wm_ts)
    ww_fc = np.nan_to_num(ww_fc) # Replace NaNs with 0
    
    # 3. GM-WM (GW)
    # corr(A, B) returns a matrix [[corr(A,A), corr(A,B)], [corr(B,A), corr(B,B)]]
    # We want corr(A, B) submatrix.
    # Manual calculation is cleaner: A_norm * B_norm^T / N
    # Or just use numpy corrcoef on stacked array and extract subblock
    logger.info("Computing GM-WM FC...")
    combined_ts = np.vstack([final_gm_ts, final_wm_ts])
    combined_fc = np.corrcoef(combined_ts)
    combined_fc = np.nan_to_num(combined_fc) # Replace NaNs with 0
    n_gm = len(gm_labels)
    gw_fc = combined_fc[:n_gm, n_gm:] # Top-right block
    
    # Save Results (.npy)
    # Save timeseries
    np.save(subj_out_dir / f'{subject_id}_GM_timeseries.npy', final_gm_ts)
    np.save(subj_out_dir / f'{subject_id}_WM_timeseries.npy', final_wm_ts)
    
    # Save FC matrices
    np.save(subj_out_dir / f'{subject_id}_GG_FC.npy', gg_fc)
    np.save(subj_out_dir / f'{subject_id}_WW_FC.npy', ww_fc)
    np.save(subj_out_dir / f'{subject_id}_GW_FC.npy', gw_fc)
    
    # Also save as CSV for readability if needed (optional, but requested "npy or csv")
    # np.savetxt(subj_out_dir / f'{subject_id}_GG_FC.csv', gg_fc, delimiter=',')
    
    logger.info("All files saved.")

def main():
    parser = argparse.ArgumentParser(description="Compute individual FC matrices.")
    parser.add_argument("--sub_id", type=str, required=True, help="Subject ID")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing masked functional data")
    parser.add_argument("--gm_atlas", type=str, required=True, help="Path to Resliced GM Atlas")
    parser.add_argument("--wm_atlas", type=str, required=True, help="Path to Resliced WM Atlas")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for FC matrices")
    
    args = parser.parse_args()
    
    compute_fc(args.sub_id, args.input_dir, args.gm_atlas, args.wm_atlas, args.output_dir)

if __name__ == "__main__":
    main()
