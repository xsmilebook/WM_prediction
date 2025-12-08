import argparse
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster execution
import matplotlib.pyplot as plt
import warnings

"""
python compute_haufe_median.py --dataset ABCD --project_folder /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/ABCD/prediction --targets nihtbx_cryst_uncorrected
python compute_haufe_median.py --age_combined
"""

def get_dimension_from_tril_len(length):
    """
    Calculate the dimension N of a square matrix from the length of its 
    strict lower triangular vector (k=-1).
    L = N(N-1)/2 => N^2 - N - 2L = 0
    N = (1 + sqrt(1 + 8L)) / 2
    """
    delta = 1 + 8 * length
    sqrt_delta = np.sqrt(delta)
    if not np.isclose(sqrt_delta, int(sqrt_delta)):
        return None
    n = (1 + int(sqrt_delta)) // 2
    return int(n)

def reconstruct_matrix(vector, fc_type, dims=None):
    """
    Reconstruct matrix from vector based on FC type and known dimensions.
    """
    vector = np.array(vector).flatten()
    l = len(vector)
    
    if fc_type in ['GGFC', 'WWFC']:
        n = get_dimension_from_tril_len(l)
        if n is None:
            raise ValueError(f"Vector length {l} does not correspond to a strict lower triangular matrix.")
        
        # Reconstruct symmetric matrix
        matrix = np.zeros((n, n))
        # tril indices k=-1
        tril_indices = np.tril_indices(n, k=-1)
        matrix[tril_indices] = vector
        # Symmetrize (Haufe weights are typically symmetric for symmetric connectivity)
        matrix = matrix + matrix.T
        return matrix, n
        
    elif fc_type == 'GWFC':
        # Needs N (GM) and M (WM) dimensions
        # Strategy: Use known dimensions from GG/WW if available
        n, m = None, None
        
        if dims:
            if 'GGFC' in dims:
                n = dims['GGFC']
            if 'WWFC' in dims:
                m = dims['WWFC']
        
        # Case 1: Both known
        if n and m:
            if l == n * m:
                return vector.reshape(n, m), (n, m)
            elif l == m * n: # Should be NxM based on creation logic but check consistency
                 # Typically GW is (n_gm, n_wm)
                 return vector.reshape(n, m), (n, m)
            else:
                raise ValueError(f"GWFC length {l} matches neither {n}x{m} nor {m}x{n}")
        
        # Case 2: Only N known (common if GG processed first)
        if n and not m:
            if l % n == 0:
                m = l // n
                return vector.reshape(n, m), (n, m)
        
        # Case 3: Neither known - try to guess square or common aspect ratio?
        # This is risky. Check if square first.
        root = np.sqrt(l)
        if np.isclose(root, int(root)):
            k = int(root)
            return vector.reshape(k, k), (k, k)
            
        raise ValueError(f"Cannot reconstruct GWFC with length {l} without known dimensions. Please process GGFC/WWFC first.")
        
    return None, None

def process_age_across_datasets(project_root, sort_idx_gm=None, sort_idx_wm=None, num_cv=101, num_folds=5):
    """
    Process age target across EFNY, ABCD, PNC, HCPD datasets and combine results.
    """
    datasets = ["EFNY", "ABCD", "PNC", "HCPD"]
    target_str = "age"
    
    print(f"Processing age target across datasets: {datasets}")
    
    # Store all Haufe vectors for each FC type across all datasets
    all_haufe_by_fc = {'GGFC': [], 'WWFC': [], 'GWFC': []}
    dataset_counts = {}
    
    # Process each FC type
    fc_types = ['GGFC', 'WWFC', 'GWFC']
    
    for fc_type in fc_types:
        print(f"\nProcessing {fc_type} across all datasets...")
        
        all_haufe_vectors = []
        dataset_counts = {}
        
        # Collect data from all datasets
        for dataset in datasets:
            print(f"  Processing {dataset}...")
            
            # Construct base folder path for this dataset
            base_folder = os.path.join(project_root, 'data', dataset, 'prediction', target_str, 'RegressCovariates_RandomCV')
            
            if not os.path.exists(base_folder):
                print(f"    Warning: Base folder not found for {dataset}: {base_folder}. Skipping.")
                continue
                    
            dataset_vectors = []
            
            # Iterate over all CV runs and folds
            for i in range(num_cv):  # 101 CV runs
                for k in range(num_folds):  # 5 folds
                    mat_path = os.path.join(base_folder, f"Time_{i}", fc_type, f"Fold_{k}_Score.mat")
                    
                    if os.path.isfile(mat_path):
                        try:
                            mat_data = sio.loadmat(mat_path)
                            if 'w_Brain_Haufe' in mat_data:
                                vec = mat_data['w_Brain_Haufe'].flatten()
                                dataset_vectors.append(vec)
                                all_haufe_vectors.append(vec)
                        except Exception as e:
                            print(f"    Error reading {mat_path}: {e}")
            
            dataset_counts[dataset] = len(dataset_vectors)
            print(f"    Loaded {len(dataset_vectors)} vectors from {dataset}")
        
        if not all_haufe_vectors:
            print(f"  No data found for {fc_type} across all datasets. Skipping.")
            continue
            
        # Stack and compute median across all datasets
        try:
            all_haufe_matrix = np.vstack(all_haufe_vectors)
            median_vector = np.median(all_haufe_matrix, axis=0)
            
            # Reconstruct matrix
            dims = {}
            if fc_type == 'GGFC':
                # For GGFC, we expect 100 regions
                n = get_dimension_from_tril_len(len(median_vector))
                if n == 100:
                    dims['GGFC'] = n
            elif fc_type == 'WWFC':
                # For WWFC, we expect 68 regions  
                n = get_dimension_from_tril_len(len(median_vector))
                if n == 68:
                    dims['WWFC'] = n
            elif fc_type == 'GWFC':
                # For GWFC, we expect 100x68 = 6800 elements
                if len(median_vector) == 6800:
                    dims['GGFC'] = 100
                    dims['WWFC'] = 68
            
            recon_matrix, dim_info = reconstruct_matrix(median_vector, fc_type, dims)
            
            # Apply sorting if available and dimensions match
            if sort_idx_gm is not None and sort_idx_wm is not None:
                try:
                    if fc_type == 'GGFC' and recon_matrix.shape == (100, 100) and len(sort_idx_gm) == 100:
                        recon_matrix = recon_matrix[sort_idx_gm][:, sort_idx_gm]
                        print("    Reordered GGFC matrix by network.")
                    elif fc_type == 'WWFC' and recon_matrix.shape == (68, 68) and len(sort_idx_wm) == 68:
                        recon_matrix = recon_matrix[sort_idx_wm][:, sort_idx_wm]
                        print("    Reordered WWFC matrix by tracts.")
                    elif fc_type == 'GWFC' and recon_matrix.shape == (100, 68) and len(sort_idx_gm) == 100 and len(sort_idx_wm) == 68:
                        recon_matrix = recon_matrix[sort_idx_gm][:, sort_idx_wm]
                        print("    Reordered GWFC matrix by network/tracts.")
                except Exception as sort_e:
                    print(f"    Warning: Failed to reorder matrix: {sort_e}")
            
            # Save combined results
            output_dir = os.path.join(project_root, 'results', 'age_combined')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"Haufe_Median_age_{fc_type}_combined.mat")
            sio.savemat(output_file, {
                'w_Brain_Haufe': median_vector,
                'w_Brain_Haufe_Matrix': recon_matrix,
                'dims': dim_info,
                'dataset_counts': dataset_counts,
                'total_samples': len(all_haufe_vectors)
            })
            print(f"  Saved combined results to {output_file}")
            
            # Visualization for combined data
            if fc_type == 'GWFC':
                # Square display for GWFC matrix
                plt.figure(figsize=(10, 10))
                max_val = np.max(np.abs(recon_matrix))
                plt.imshow(recon_matrix, cmap='RdBu_r', vmin=-max_val, vmax=max_val, aspect='auto')
                plt.colorbar(label='Haufe Weight', fraction=0.046, pad=0.04)
            else:
                plt.figure(figsize=(10, 8))
                max_val = np.max(np.abs(recon_matrix))
                plt.imshow(recon_matrix, cmap='RdBu_r', vmin=-max_val, vmax=max_val)
                plt.colorbar(label='Haufe Weight')
            
            # Create title with dataset counts
            dataset_info = ", ".join([f"{ds}: {count}" for ds, count in dataset_counts.items()])
            plt.title(f"Median Haufe Matrix - Age Combined - {fc_type}\n({len(all_haufe_vectors)} total samples from {len(dataset_counts)} datasets)\n{dataset_info}")
            
            vis_file = os.path.join(output_dir, f"Haufe_Median_age_{fc_type}_combined.png")
            plt.savefig(vis_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved combined visualization to {vis_file}")
            
        except Exception as e:
            print(f"  Error during processing/reconstruction for {fc_type}: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Extract and reconstruct Haufe weights from CV results.")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g. ABCD, HCPD)")
    parser.add_argument("--project_folder", type=str, help="Base project folder containing prediction results")
    parser.add_argument("--targets", type=str, nargs='+', 
                        default=["nihtbx_cryst_uncorrected", "nihtbx_fluidcomp_uncorrected", "nihtbx_totalcomp_uncorrected"], 
                        help="List of target variables")
    parser.add_argument("--num_cv", type=int, default=101, help="Number of CV runs (default: 101)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--age_combined", action="store_true", help="Process age target across all datasets (EFNY, ABCD, PNC, HCPD)")
    
    args = parser.parse_args()
    
    if args.age_combined:
        print("Running in age_combined mode - processing across all datasets")
        # Load Atlas info for sorting
        sort_idx_gm, sort_idx_wm = None, None
        try:
            # Assuming script is in src/results_vis/
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to project root: src/results_vis -> src -> WM_prediction
            project_root = os.path.dirname(os.path.dirname(script_dir))
            atlas_dir = os.path.join(project_root, 'data', 'atlas')
            
            schaefer_path = os.path.join(atlas_dir, 'Schaefer100_info.mat')
            jhu_path = os.path.join(atlas_dir, 'JHU68_info.mat')
            
            if os.path.exists(schaefer_path) and os.path.exists(jhu_path):
                schaefer_info = sio.loadmat(schaefer_path)
                jhu_info = sio.loadmat(jhu_path)
                
                # MATLAB 1-based -> Python 0-based
                if 'regionID_sortedByNetwork' in schaefer_info:
                    sort_idx_gm = schaefer_info['regionID_sortedByNetwork'].flatten() - 1
                if 'regionID_sortedByTracts' in jhu_info:
                    sort_idx_wm = jhu_info['regionID_sortedByTracts'].flatten() - 1
                
                if sort_idx_gm is not None and sort_idx_wm is not None:
                    print(f"Loaded Atlas sorting info. GM: {len(sort_idx_gm)}, WM: {len(sort_idx_wm)}")
            else:
                print(f"Warning: Atlas files not found at {atlas_dir}. Sorting will be skipped.")
                
        except Exception as e:
            print(f"Warning: Failed to load Atlas info: {e}")
        
        process_age_across_datasets(project_root, sort_idx_gm, sort_idx_wm, args.num_cv, args.num_folds)
        return
    
    # 验证参数逻辑
    if not args.dataset or not args.project_folder:
        parser.error("--dataset and --project_folder are required when not using --age_combined")
    
    # 原有的单数据集处理逻辑
    print(f"Dataset: {args.dataset}")
    print(f"Project Folder: {args.project_folder}")
    print(f"Targets: {args.targets}")
    
    # Load Atlas Info for sorting
    sort_idx_gm, sort_idx_wm = None, None
    try:
        # Assuming script is in src/results_vis/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root: src/results_vis -> src -> WM_prediction
        project_root = os.path.dirname(os.path.dirname(script_dir))
        atlas_dir = os.path.join(project_root, 'data', 'atlas')
        
        schaefer_path = os.path.join(atlas_dir, 'Schaefer100_info.mat')
        jhu_path = os.path.join(atlas_dir, 'JHU68_info.mat')
        
        if os.path.exists(schaefer_path) and os.path.exists(jhu_path):
            schaefer_info = sio.loadmat(schaefer_path)
            jhu_info = sio.loadmat(jhu_path)
            
            # MATLAB 1-based -> Python 0-based
            if 'regionID_sortedByNetwork' in schaefer_info:
                sort_idx_gm = schaefer_info['regionID_sortedByNetwork'].flatten() - 1
            if 'regionID_sortedByTracts' in jhu_info:
                sort_idx_wm = jhu_info['regionID_sortedByTracts'].flatten() - 1
            
            if sort_idx_gm is not None and sort_idx_wm is not None:
                print(f"Loaded Atlas sorting info. GM: {len(sort_idx_gm)}, WM: {len(sort_idx_wm)}")
        else:
            print(f"Warning: Atlas files not found at {atlas_dir}. Sorting will be skipped.")
            
    except Exception as e:
        print(f"Warning: Failed to load Atlas info: {e}")

    # Enforce processing order to ensure dimensions are found for GWFC
    fc_types = ['GGFC', 'WWFC', 'GWFC']
    
    for target_str in args.targets:
        print(f"\nProcessing target: {target_str}")
        base_folder = os.path.join(args.project_folder, target_str, 'RegressCovariates_RandomCV')
        
        if not os.path.exists(base_folder):
            print(f"Warning: Base folder not found: {base_folder}. Skipping.")
            continue
            
        # Store dimensions for this target
        dims = {}
        
        for fc_type in fc_types:
            print(f"  Processing {fc_type}...")
            
            all_haufe_vectors = []
            missing_count = 0
            
            # Iterate over all CV runs and folds
            for i in range(args.num_cv):
                for k in range(args.num_folds):
                    # Path format: base_folder/Time_{i}/{fc_type}/Fold_{k}_Score.mat
                    mat_path = os.path.join(base_folder, f"Time_{i}", fc_type, f"Fold_{k}_Score.mat")
                    
                    if os.path.isfile(mat_path):
                        try:
                            mat_data = sio.loadmat(mat_path)
                            if 'w_Brain_Haufe' in mat_data:
                                vec = mat_data['w_Brain_Haufe'].flatten()
                                all_haufe_vectors.append(vec)
                            else:
                                missing_count += 1
                        except Exception as e:
                            print(f"    Error reading {mat_path}: {e}")
                            missing_count += 1
                    else:
                        missing_count += 1
            
            total_expected = args.num_cv * args.num_folds
            print(f"    Loaded {len(all_haufe_vectors)} vectors (Missing: {missing_count}/{total_expected})")
            
            if not all_haufe_vectors:
                print(f"    No data found for {fc_type}. Skipping.")
                continue
                
            # Stack and compute median
            try:
                all_haufe_matrix = np.vstack(all_haufe_vectors) # shape (N_samples, N_features)
                median_vector = np.median(all_haufe_matrix, axis=0)
                
                # Reconstruct matrix
                recon_matrix, dim_info = reconstruct_matrix(median_vector, fc_type, dims)
                
                # Apply sorting if available and dimensions match
                if sort_idx_gm is not None and sort_idx_wm is not None:
                    try:
                        if fc_type == 'GGFC' and recon_matrix.shape == (100, 100) and len(sort_idx_gm) == 100:
                            recon_matrix = recon_matrix[sort_idx_gm][:, sort_idx_gm]
                            print("    Reordered GGFC matrix by network.")
                        elif fc_type == 'WWFC' and recon_matrix.shape == (68, 68) and len(sort_idx_wm) == 68:
                            recon_matrix = recon_matrix[sort_idx_wm][:, sort_idx_wm]
                            print("    Reordered WWFC matrix by tracts.")
                        elif fc_type == 'GWFC' and recon_matrix.shape == (100, 68) and len(sort_idx_gm) == 100 and len(sort_idx_wm) == 68:
                            recon_matrix = recon_matrix[sort_idx_gm][:, sort_idx_wm]
                            print("    Reordered GWFC matrix by network/tracts.")
                    except Exception as sort_e:
                        print(f"    Warning: Failed to reorder matrix: {sort_e}")
                
                # Update dimensions
                if fc_type == 'GGFC':
                    dims['GGFC'] = dim_info # n
                elif fc_type == 'WWFC':
                    dims['WWFC'] = dim_info # m
                
                # Save results
                output_dir = os.path.join(base_folder, 'Haufe_Analysis')
                os.makedirs(output_dir, exist_ok=True)
                
                output_file = os.path.join(output_dir, f"Haufe_Median_{fc_type}.mat")
                sio.savemat(output_file, {
                    'w_Brain_Haufe': median_vector, # The vector as requested
                    'w_Brain_Haufe_Matrix': recon_matrix, # The reconstructed matrix
                    'dims': dim_info
                })
                print(f"    Saved results to {output_file}")
                
                # Visualization - Square display for GWFC matrix
                if fc_type == 'GWFC':
                    # For GWFC (68x100), create square figure with aspect='auto'
                    plt.figure(figsize=(10, 10))
                    max_val = np.max(np.abs(recon_matrix))
                    plt.imshow(recon_matrix, cmap='RdBu_r', vmin=-max_val, vmax=max_val, aspect='auto')
                    plt.colorbar(label='Haufe Weight', fraction=0.046, pad=0.04)
                    plt.title(f"Median Haufe Matrix - {target_str} - {fc_type}\n(Median of {len(all_haufe_vectors)} folds)")
                else:
                    # Original visualization for GGFC and WWFC
                    plt.figure(figsize=(10, 8))
                    max_val = np.max(np.abs(recon_matrix))
                    plt.imshow(recon_matrix, cmap='RdBu_r', vmin=-max_val, vmax=max_val)
                    plt.colorbar(label='Haufe Weight')
                    plt.title(f"Median Haufe Matrix - {target_str} - {fc_type}\n(Median of {len(all_haufe_vectors)} folds)")
                
                vis_file = os.path.join(output_dir, f"Haufe_Median_{fc_type}.png")
                plt.savefig(vis_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    Saved visualization to {vis_file}")
                
            except Exception as e:
                print(f"    Error during processing/reconstruction for {fc_type}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()