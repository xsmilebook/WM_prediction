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

def main():
    parser = argparse.ArgumentParser(description="Extract and reconstruct Haufe weights from CV results.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. ABCD, HCPD)")
    parser.add_argument("--project_folder", type=str, required=True, help="Base project folder containing prediction results")
    parser.add_argument("--targets", type=str, nargs='+', 
                        default=["nihtbx_cryst_uncorrected", "nihtbx_fluidcomp_uncorrected", "nihtbx_totalcomp_uncorrected"], 
                        help="List of target variables")
    parser.add_argument("--num_cv", type=int, default=101, help="Number of CV runs (default: 101)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds (default: 5)")
    
    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset}")
    print(f"Project Folder: {args.project_folder}")
    print(f"Targets: {args.targets}")
    
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
                
                # Visualization
                plt.figure(figsize=(10, 8))
                # Use divergent colormap centered at 0
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