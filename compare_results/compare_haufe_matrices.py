import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster execution
import matplotlib.pyplot as plt
import argparse

def load_atlas_info():
    """Load atlas information for matrix reordering."""
    try:
        # Load Schaefer100 atlas info (for GM regions)
        schaefer_file = "d:/code/WM_prediction/data/atlas/Schaefer100_info.mat"
        schaefer_data = sio.loadmat(schaefer_file)
        sort_idx_gm = schaefer_data['regionID_sortedByNetwork'].flatten() - 1  # Convert from 1-based to 0-based indexing
        
        # Load JHU68 atlas info (for WM regions)
        jhu_file = "d:/code/WM_prediction/data/atlas/JHU68_info.mat"
        jhu_data = sio.loadmat(jhu_file)
        sort_idx_wm = jhu_data['regionID_sortedByTracts'].flatten() - 1  # Convert from 1-based to 0-based indexing
        
        print(f"Loaded atlas info: GM sort indices shape {sort_idx_gm.shape}, WM sort indices shape {sort_idx_wm.shape}")
        return sort_idx_gm, sort_idx_wm
        
    except Exception as e:
        print(f"Warning: Failed to load atlas info: {e}")
        return None, None

def reorder_matrix_by_atlas(matrix, fc_type, sort_idx_gm, sort_idx_wm):
    """Reorder matrix based on atlas network/tract organization."""
    if sort_idx_gm is None or sort_idx_wm is None:
        return matrix
        
    try:
        recon_matrix = matrix.copy()
        
        if fc_type == 'GGFC' and recon_matrix.shape == (100, 100) and len(sort_idx_gm) == 100:
            recon_matrix = recon_matrix[sort_idx_gm][:, sort_idx_gm]
            print("    Reordered GGFC matrix by network.")
        elif fc_type == 'WWFC' and recon_matrix.shape == (68, 68) and len(sort_idx_wm) == 68:
            recon_matrix = recon_matrix[sort_idx_wm][:, sort_idx_wm]
            print("    Reordered WWFC matrix by tracts.")
        elif fc_type == 'GWFC' and recon_matrix.shape == (100, 68) and len(sort_idx_gm) == 100 and len(sort_idx_wm) == 68:
            recon_matrix = recon_matrix[sort_idx_gm][:, sort_idx_wm]
            print("    Reordered GWFC matrix by network/tracts.")
        else:
            print(f"    No reordering applied for {fc_type} with shape {recon_matrix.shape}")
            
        return recon_matrix
        
    except Exception as sort_e:
        print(f"    Warning: Failed to reorder matrix: {sort_e}")
        return matrix

def load_matrix_from_mat(mat_path, field_name):
    """Load matrix from mat file, trying different field names."""
    try:
        mat_data = sio.loadmat(mat_path)
        if field_name in mat_data:
            return mat_data[field_name]
        else:
            print(f"Warning: Field '{field_name}' not found in {mat_path}")
            print(f"Available fields: {list(mat_data.keys())}")
            return None
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

def compute_matrix_differences(matrix1, matrix2):
    """Compute various difference metrics between two matrices."""
    if matrix1.shape != matrix2.shape:
        print(f"Matrix shapes don't match: {matrix1.shape} vs {matrix2.shape}")
        return None
    
    # Absolute difference
    abs_diff = np.abs(matrix1 - matrix2)
    
    # Relative difference (avoid division by zero)
    mask = np.abs(matrix2) > 1e-10
    rel_diff = np.zeros_like(matrix1)
    rel_diff[mask] = np.abs((matrix1[mask] - matrix2[mask]) / matrix2[mask])
    
    # Correlation
    corr_coef = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
    
    # Statistics
    stats = {
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'std_abs_diff': np.std(abs_diff),
        'max_rel_diff': np.max(rel_diff) if np.any(mask) else 0,
        'mean_rel_diff': np.mean(rel_diff[mask]) if np.any(mask) else 0,
        'correlation': corr_coef,
        'shape': matrix1.shape
    }
    
    return {
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'stats': stats
    }

def visualize_differences(matrix1, matrix2, diff_results, target_name, fc_type, output_dir):
    """Visualize the differences between matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original matrix 1 (haufe_weights)
    im1 = axes[0, 0].imshow(matrix1, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title(f'Haufe Weights - {target_name} - {fc_type}')
    axes[0, 0].set_xlabel('Region')
    axes[0, 0].set_ylabel('Region')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Original matrix 2 (compare_results)
    im2 = axes[0, 1].imshow(matrix2, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title(f'Compare Results - {target_name} - {fc_type}')
    axes[0, 1].set_xlabel('Region')
    axes[0, 1].set_ylabel('Region')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Absolute difference
    max_diff = np.max(np.abs(diff_results['abs_diff']))
    im3 = axes[1, 0].imshow(diff_results['abs_diff'], cmap='Reds', vmin=0, vmax=max_diff, aspect='auto')
    axes[1, 0].set_title(f'Absolute Difference - {target_name} - {fc_type}')
    axes[1, 0].set_xlabel('Region')
    axes[1, 0].set_ylabel('Region')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Relative difference
    max_rel_diff = np.max(diff_results['rel_diff'])
    im4 = axes[1, 1].imshow(diff_results['rel_diff'], cmap='Reds', vmin=0, vmax=max_rel_diff, aspect='auto')
    axes[1, 1].set_title(f'Relative Difference - {target_name} - {fc_type}')
    axes[1, 1].set_xlabel('Region')
    axes[1, 1].set_ylabel('Region')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f"comparison_{target_name}_{fc_type}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison visualization to {output_file}")
    
    return output_file

def save_difference_results(diff_results, target_name, fc_type, output_dir):
    """Save difference results to mat file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"differences_{target_name}_{fc_type}.mat")
    
    save_data = {
        'abs_diff': diff_results['abs_diff'],
        'rel_diff': diff_results['rel_diff'],
        'stats': diff_results['stats']
    }
    
    sio.savemat(output_file, save_data)
    print(f"  Saved difference results to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Compare Haufe weights matrices between two sources.")
    parser.add_argument("--haufe_dir", type=str, default="d:/code/WM_prediction/data/haufe_weights", 
                        help="Directory containing haufe_weights results")
    parser.add_argument("--compare_dir", type=str, default="d:/code/WM_prediction/data/compare_results", 
                        help="Directory containing compare_results")
    parser.add_argument("--output_dir", type=str, default="d:/code/WM_prediction/data/compare_results", 
                        help="Output directory for comparison results")
    parser.add_argument("--targets", type=str, nargs='+', default=['age', 'cognition', 'pfactor'],
                        help="Target names to compare")
    parser.add_argument("--fc_types", type=str, nargs='+', default=['GGFC', 'WWFC', 'GWFC'],
                        help="FC types to compare")
    
    args = parser.parse_args()
    
    print("Starting matrix comparison...")
    print(f"Haufe weights directory: {args.haufe_dir}")
    print(f"Compare results directory: {args.compare_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Targets: {args.targets}")
    print(f"FC types: {args.fc_types}")
    
    # Load atlas information for matrix reordering
    print("Loading atlas information...")
    sort_idx_gm, sort_idx_wm = load_atlas_info()
    
    comparison_summary = {}
    
    # Process each target and FC type combination
    for target in args.targets:
        print(f"\nProcessing target: {target}")
        comparison_summary[target] = {}
        
        for fc_type in args.fc_types:
            print(f"  Processing FC type: {fc_type}")
            
            # Construct file paths based on actual naming conventions
            if target == 'age':
                haufe_file = os.path.join(args.haufe_dir, target, f"Haufe_Median_age_{fc_type}_combined.mat")
                # Use correct naming for age files
                if fc_type == 'GGFC':
                    compare_file = os.path.join(args.compare_dir, target, "datasetsAveraged_Haufe_FCmatrix_gg.mat")
                elif fc_type == 'WWFC':
                    compare_file = os.path.join(args.compare_dir, target, "datasetsAveraged_Haufe_FCmatrix_ww.mat")
                elif fc_type == 'GWFC':
                    compare_file = os.path.join(args.compare_dir, target, "datasetsAveraged_Haufe_FCmatrix_gw.mat")
            else:
                haufe_file = os.path.join(args.haufe_dir, target, f"Haufe_Median_{fc_type}.mat")
                # Use correct naming for cognition and pfactor files
                if target == 'cognition':
                    # Special case for cognition files
                    if fc_type == 'GGFC':
                        compare_file = os.path.join(args.compare_dir, target, "ABCDcog_Haufe_FCmatrix_gg_Schaefer100.mat")
                    elif fc_type == 'WWFC':
                        compare_file = os.path.join(args.compare_dir, target, "ABCDcog_Haufe_FCmatrix_ww_Schaefer100.mat")
                    elif fc_type == 'GWFC':
                        compare_file = os.path.join(args.compare_dir, target, "ABCDcog_Haufe_FCmatrix_gw_Schaefer100.mat")
                else:
                    # Standard naming for pfactor
                    if fc_type == 'GGFC':
                        compare_file = os.path.join(args.compare_dir, target, f"ABCD{target}_Haufe_FCmatrix_gg_Schaefer100.mat")
                    elif fc_type == 'WWFC':
                        compare_file = os.path.join(args.compare_dir, target, f"ABCD{target}_Haufe_FCmatrix_ww_Schaefer100.mat")
                    elif fc_type == 'GWFC':
                        compare_file = os.path.join(args.compare_dir, target, f"ABCD{target}_Haufe_FCmatrix_gw_Schaefer100.mat")
            
            # Check if files exist
            if not os.path.exists(haufe_file):
                print(f"    Warning: Haufe weights file not found: {haufe_file}")
                continue
                
            if not os.path.exists(compare_file):
                print(f"    Warning: Compare results file not found: {compare_file}")
                continue
            
            # Load matrices
            print(f"    Loading matrices...")
            haufe_matrix = load_matrix_from_mat(haufe_file, 'w_Brain_Haufe_Matrix')
            compare_matrix = load_matrix_from_mat(compare_file, 'FCmatrix_full')
            
            if haufe_matrix is None or compare_matrix is None:
                print(f"    Error: Failed to load matrices")
                continue
            
            print(f"    Original Haufe matrix shape: {haufe_matrix.shape}")
            print(f"    Original Compare matrix shape: {compare_matrix.shape}")
            
            # Apply atlas-based reordering only to compare matrix
            # Haufe weights are already reordered when saved
            print(f"    Reordering compare matrix by atlas (haufe matrix already reordered)...")
            compare_matrix_reordered = reorder_matrix_by_atlas(compare_matrix, fc_type, sort_idx_gm, sort_idx_wm)
            
            print(f"    Haufe matrix shape: {haufe_matrix.shape}")
            print(f"    Reordered Compare matrix shape: {compare_matrix_reordered.shape}")
            
            # Compute differences using haufe matrix (already reordered) and reordered compare matrix
            print(f"    Computing differences...")
            diff_results = compute_matrix_differences(haufe_matrix, compare_matrix_reordered)
            
            if diff_results is None:
                continue
            
            # Print statistics
            stats = diff_results['stats']
            print(f"    Statistics:")
            print(f"      Max absolute difference: {stats['max_abs_diff']:.6f}")
            print(f"      Mean absolute difference: {stats['mean_abs_diff']:.6f}")
            print(f"      Std absolute difference: {stats['std_abs_diff']:.6f}")
            print(f"      Max relative difference: {stats['max_rel_diff']:.6f}")
            print(f"      Mean relative difference: {stats['mean_rel_diff']:.6f}")
            print(f"      Correlation coefficient: {stats['correlation']:.6f}")
            
            # Visualize differences
            print(f"    Creating visualizations...")
            vis_file = visualize_differences(haufe_matrix, compare_matrix_reordered, diff_results, 
                                           target, fc_type, args.output_dir)
            
            # Save difference results
            diff_file = save_difference_results(diff_results, target, fc_type, args.output_dir)
            
            # Store summary
            comparison_summary[target][fc_type] = {
                'stats': stats,
                'visualization': vis_file,
                'difference_file': diff_file,
                'haufe_file': haufe_file,
                'compare_file': compare_file
            }
    
    # Save overall summary
    summary_file = os.path.join(args.output_dir, "comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Matrix Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for target in args.targets:
            if target not in comparison_summary:
                continue
                
            f.write(f"Target: {target}\n")
            f.write("-" * 30 + "\n")
            
            for fc_type in args.fc_types:
                if fc_type not in comparison_summary[target]:
                    continue
                    
                result = comparison_summary[target][fc_type]
                stats = result['stats']
                
                f.write(f"  FC Type: {fc_type}\n")
                f.write(f"    Matrix shape: {stats['shape']}\n")
                f.write(f"    Max absolute difference: {stats['max_abs_diff']:.6f}\n")
                f.write(f"    Mean absolute difference: {stats['mean_abs_diff']:.6f}\n")
                f.write(f"    Std absolute difference: {stats['std_abs_diff']:.6f}\n")
                f.write(f"    Max relative difference: {stats['max_rel_diff']:.6f}\n")
                f.write(f"    Mean relative difference: {stats['mean_rel_diff']:.6f}\n")
                f.write(f"    Correlation coefficient: {stats['correlation']:.6f}\n")
                f.write(f"    Visualization: {os.path.basename(result['visualization'])}\n")
                f.write(f"    Difference file: {os.path.basename(result['difference_file'])}\n")
                f.write("\n")
            
            f.write("\n")
    
    print(f"\nComparison complete! Summary saved to {summary_file}")
    print("All results saved to:", args.output_dir)

if __name__ == "__main__":
    main()