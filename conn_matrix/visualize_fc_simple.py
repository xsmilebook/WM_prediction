#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified visualization of FC matrix comparison
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_matrix(file_path):
    """Load matrix file"""
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def create_simple_comparison_plots(contrast_matrix, individual_matrix, matrix_type, subject_id, output_dir):
    """Create simplified comparison visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original matrices
    im1 = axes[0, 0].imshow(contrast_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im1, ax=axes[0, 0])
    axes[0, 0].set_title(f'Contrast Version - {matrix_type}', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Region')
    axes[0, 0].set_ylabel('Region')
    
    im2 = axes[0, 1].imshow(individual_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].set_title(f'Individual Version - {matrix_type}', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Region')
    axes[0, 1].set_ylabel('Region')
    
    # 2. Difference matrix
    diff_matrix = individual_matrix - contrast_matrix
    im3 = axes[0, 2].imshow(diff_matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im3, ax=axes[0, 2])
    axes[0, 2].set_title(f'Difference Matrix', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Region')
    axes[0, 2].set_ylabel('Region')
    
    # 3. Correlation scatter plot
    flat1 = contrast_matrix.flatten()
    flat2 = individual_matrix.flatten()
    
    # Remove NaN values
    valid_mask = ~np.isnan(flat1) & ~np.isnan(flat2)
    if valid_mask.any():
        x_vals = flat1[valid_mask]
        y_vals = flat2[valid_mask]
        
        axes[1, 0].scatter(x_vals, y_vals, alpha=0.6, s=1)
        
        # Add diagonal line
        min_val = min(x_vals.min(), y_vals.min())
        max_val = max(x_vals.max(), y_vals.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        # Calculate correlation
        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Pearson r = {correlation:.4f}', 
                       transform=axes[1, 0].transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        axes[1, 0].set_xlabel('Contrast Values')
        axes[1, 0].set_ylabel('Individual Values')
        axes[1, 0].set_title(f'Correlation Scatter Plot', fontsize=12)
        axes[1, 0].legend()
    
    # 4. Distribution histograms
    axes[1, 1].hist(flat1[valid_mask], bins=50, alpha=0.7, label='Contrast', density=True)
    axes[1, 1].hist(flat2[valid_mask], bins=50, alpha=0.7, label='Individual', density=True)
    axes[1, 1].set_xlabel('FC Values')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'Value Distribution Comparison', fontsize=12)
    axes[1, 1].legend()
    
    # 5. Statistics text
    axes[1, 2].axis('off')
    
    stats_text = f"""
    Statistics Comparison - {matrix_type}
    
    Contrast Version:
    - Mean: {np.nanmean(contrast_matrix):.6f}
    - Std: {np.nanstd(contrast_matrix):.6f}
    - Min: {np.nanmin(contrast_matrix):.6f}
    - Max: {np.nanmax(contrast_matrix):.6f}
    
    Individual Version:
    - Mean: {np.nanmean(individual_matrix):.6f}
    - Std: {np.nanstd(individual_matrix):.6f}
    - Min: {np.nanmin(individual_matrix):.6f}
    - Max: {np.nanmax(individual_matrix):.6f}
    
    Difference Statistics:
    - MAE: {np.nanmean(np.abs(diff_matrix)):.6f}
    - Max Diff: {np.nanmax(np.abs(diff_matrix)):.6f}
    - Correlation: {correlation:.6f}
    """
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'{subject_id}_{matrix_type}_simple_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Simple visualization of FC matrix comparison')
    parser.add_argument('--contrast_dir', required=True, 
                       help='Contrast version directory path')
    parser.add_argument('--individual_dir', required=True,
                       help='Individual version directory path')
    parser.add_argument('--subject_id', required=True,
                       help='Subject ID')
    parser.add_argument('--output_dir', default='./fc_visualization_simple',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    print(f"Starting simple visualization for subject {args.subject_id}")
    
    # Convert paths
    contrast_dir = Path(args.contrast_dir)
    individual_dir = Path(args.individual_dir)
    output_dir = Path(args.output_dir)
    
    # Check directories exist
    if not contrast_dir.exists():
        print(f"Error: Contrast directory not found: {contrast_dir}")
        return 1
    if not individual_dir.exists():
        print(f"Error: Individual directory not found: {individual_dir}")
        return 1
    
    # Find matrix files
    contrast_files = list(contrast_dir.glob("*.npy"))
    individual_files = list(individual_dir.glob("*.npy"))
    
    # Filter out timeseries files
    contrast_matrices = [f for f in contrast_files if 'timeseries' not in f.name]
    individual_matrices = [f for f in individual_files if 'timeseries' not in f.name]
    
    print(f"\nFound {len(contrast_matrices)} contrast matrix files")
    print(f"Found {len(individual_matrices)} individual matrix files")
    
    # Match matrix pairs
    matches = []
    matrix_types = [
        ('GM', 'GG'),
        ('WM', 'WW'), 
        ('GM_WM', 'GW')
    ]
    
    for contrast_type, individual_type in matrix_types:
        contrast_matches = [f for f in contrast_matrices if contrast_type in f.name and 'FC' in f.name]
        individual_matches = [f for f in individual_matrices if individual_type in f.name and 'FC' in f.name]
        
        if contrast_matches and individual_matches:
            matches.append((contrast_matches[0], individual_matches[0], contrast_type, individual_type))
    
    if not matches:
        print("Error: No matching matrix pairs found")
        return 1
    
    print(f"\nFound {len(matches)} matching matrix pairs")
    
    # Create visualizations for each pair
    for contrast_file, individual_file, contrast_type, individual_type in matches:
        print(f"\nProcessing matrix pair: {contrast_type} vs {individual_type}")
        
        contrast_matrix = load_matrix(contrast_file)
        individual_matrix = load_matrix(individual_file)
        
        if contrast_matrix is not None and individual_matrix is not None:
            # Check shape compatibility
            if contrast_matrix.shape != individual_matrix.shape:
                print(f"Warning: Shape mismatch {contrast_matrix.shape} vs {individual_matrix.shape}")
                if contrast_matrix.shape == individual_matrix.T.shape:
                    individual_matrix = individual_matrix.T
                    print("Transposed individual matrix")
                else:
                    print("Skipping: Incompatible shapes")
                    continue
            
            create_simple_comparison_plots(
                contrast_matrix, individual_matrix, 
                f"{contrast_type}_vs_{individual_type}", 
                args.subject_id, output_dir
            )
    
    print(f"\nVisualization complete! Figures saved in: {output_dir}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())