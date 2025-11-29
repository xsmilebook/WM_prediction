#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比两个图谱文件是否完全一致（基于文件内容匹配）
Compare atlas files between contrast_abcd and resliced_abcd directories

command:
python d:/code/WM_prediction/src/conn_matrix/compare_atlas_files.py --contrast_dir d:/code/WM_prediction/data/atlas/contrast_abcd --resliced_dir d:/code/WM_prediction/data/atlas/resliced_abcd --output_dir ./atlas_comparison 
"""

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_nifti(file_path):
    """加载NIfTI文件"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return img, data
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None, None

def find_matching_files(contrast_dir, resliced_dir):
    """根据文件内容找到匹配的文件对"""
    contrast_files = list(contrast_dir.glob("*.nii.gz"))
    resliced_files = list(resliced_dir.glob("*.nii.gz"))
    
    matches = []
    
    # 对比版本文件：rICBM_DTI_81_WMPM_60p_FMRIB58_resample.nii.gz
    # 重切片版本文件：rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz
    # 对比版本文件：Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resample.nii.gz
    # 重切片版本文件：Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz
    
    # 手动匹配文件对
    file_mappings = [
        ("rICBM_DTI_81_WMPM_60p_FMRIB58_resample.nii.gz", "rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz"),
        ("Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resample.nii.gz", "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz")
    ]
    
    for contrast_name, resliced_name in file_mappings:
        contrast_file = contrast_dir / contrast_name
        resliced_file = resliced_dir / resliced_name
        
        if contrast_file.exists() and resliced_file.exists():
            matches.append((contrast_file, resliced_file, contrast_name, resliced_name))
        else:
            if not contrast_file.exists():
                print(f"警告: 对比版本文件不存在: {contrast_file}")
            if not resliced_file.exists():
                print(f"警告: 重切片版本文件不存在: {resliced_file}")
    
    return matches

def compare_atlas_files(contrast_file, resliced_file, contrast_name, resliced_name):
    """对比两个图谱文件"""
    print(f"\n{'='*60}")
    print(f"对比文件:")
    print(f"对比版本: {contrast_name}")
    print(f"重切片版本: {resliced_name}")
    print(f"{'='*60}")
    
    # 加载文件
    contrast_img, contrast_data = load_nifti(contrast_file)
    resliced_img, resliced_data = load_nifti(resliced_file)
    
    if contrast_data is None or resliced_data is None:
        return False
    
    # 1. 检查形状
    print(f"对比版本形状: {contrast_data.shape}")
    print(f"重切片版本形状: {resliced_data.shape}")
    
    if contrast_data.shape != resliced_data.shape:
        print("❌ 形状不匹配！")
        return False
    else:
        print("✅ 形状匹配")
    
    # 2. 检查数据类型
    print(f"对比版本数据类型: {contrast_data.dtype}")
    print(f"重切片版本数据类型: {resliced_data.dtype}")
    
    if contrast_data.dtype != resliced_data.dtype:
        print("⚠️  数据类型不同")
    else:
        print("✅ 数据类型相同")
    
    # 3. 检查仿射矩阵
    contrast_affine = contrast_img.affine
    resliced_affine = resliced_img.affine
    
    print(f"\n仿射矩阵对比:")
    print(f"对比版本仿射矩阵:\n{contrast_affine}")
    print(f"重切片版本仿射矩阵:\n{resliced_affine}")
    
    if np.allclose(contrast_affine, resliced_affine, rtol=1e-6):
        print("✅ 仿射矩阵相同")
    else:
        print("❌ 仿射矩阵不同")
        print(f"最大差异: {np.max(np.abs(contrast_affine - resliced_affine))}")
        print(f"相对差异: {np.max(np.abs(contrast_affine - resliced_affine)) / np.max(np.abs(contrast_affine)):.2e}")
    
    # 4. 检查数据内容
    # 移除NaN值进行对比
    valid_mask = ~np.isnan(contrast_data) & ~np.isnan(resliced_data)
    
    if not valid_mask.any():
        print("❌ 没有有效数据可对比")
        return False
    
    contrast_valid = contrast_data[valid_mask]
    resliced_valid = resliced_data[valid_mask]
    
    # 检查是否完全相同
    tolerance = 1e-6
    if np.allclose(contrast_valid, resliced_valid, rtol=tolerance, atol=tolerance):
        print(f"✅ 数据内容在容差 {tolerance} 下完全相同！")
        return True
    else:
        print("❌ 数据内容不同")
        
        # 计算统计差异
        mae = np.mean(np.abs(contrast_valid - resliced_valid))
        max_diff = np.max(np.abs(contrast_valid - resliced_valid))
        correlation = np.corrcoef(contrast_valid, resliced_valid)[0, 1]
        
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"最大差异: {max_diff:.6f}")
        print(f"Pearson相关性: {correlation:.6f}")
        
        # 统计信息对比
        print(f"\n统计信息对比:")
        print(f"对比版本 - 均值: {np.mean(contrast_valid):.6f}, 标准差: {np.std(contrast_valid):.6f}")
        print(f"重切片版本 - 均值: {np.mean(resliced_valid):.6f}, 标准差: {np.std(resliced_valid):.6f}")
        
        # 检查唯一值
        contrast_unique = np.unique(contrast_valid)
        resliced_unique = np.unique(resliced_valid)
        
        print(f"对比版本唯一值数量: {len(contrast_unique)}")
        print(f"重切片版本唯一值数量: {len(resliced_unique)}")
        
        if len(contrast_unique) == len(resliced_unique):
            if np.array_equal(contrast_unique, resliced_unique):
                print("✅ 唯一值集合相同")
            else:
                print("❌ 唯一值集合不同")
                print(f"对比版本前10个唯一值: {contrast_unique[:10]}")
                print(f"重切片版本前10个唯一值: {resliced_unique[:10]}")
        else:
            print("❌ 唯一值数量不同")
        
        return False

def create_visualization(contrast_data, resliced_data, file_name, output_dir):
    """创建可视化对比"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 选择中间切片进行可视化
    mid_slice = contrast_data.shape[2] // 2
    
    # 1. 原始数据对比
    im1 = axes[0, 0].imshow(contrast_data[:, :, mid_slice], cmap='viridis')
    plt.colorbar(im1, ax=axes[0, 0])
    axes[0, 0].set_title(f'Contrast - Slice {mid_slice}', fontweight='bold')
    
    im2 = axes[0, 1].imshow(resliced_data[:, :, mid_slice], cmap='viridis')
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].set_title(f'Resliced - Slice {mid_slice}', fontweight='bold')
    
    # 2. 差异图
    diff_data = resliced_data - contrast_data
    im3 = axes[0, 2].imshow(diff_data[:, :, mid_slice], cmap='RdBu_r', 
                           vmin=-np.max(np.abs(diff_data)), vmax=np.max(np.abs(diff_data)))
    plt.colorbar(im3, ax=axes[0, 2])
    axes[0, 2].set_title(f'Difference (Resliced - Contrast)', fontweight='bold')
    
    # 3. 值分布直方图
    valid_mask = ~np.isnan(contrast_data) & ~np.isnan(resliced_data)
    if valid_mask.any():
        axes[1, 0].hist(contrast_data[valid_mask].flatten(), bins=50, alpha=0.7, 
                       label='Contrast', density=True)
        axes[1, 0].hist(resliced_data[valid_mask].flatten(), bins=50, alpha=0.7, 
                       label='Resliced', density=True)
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Value Distribution Comparison')
        axes[1, 0].legend()
    
    # 4. 相关性散点图（采样显示）
    if valid_mask.any():
        contrast_flat = contrast_data[valid_mask].flatten()
        resliced_flat = resliced_data[valid_mask].flatten()
        
        # 采样显示，避免点太多
        sample_size = min(10000, len(contrast_flat))
        indices = np.random.choice(len(contrast_flat), sample_size, replace=False)
        
        axes[1, 1].scatter(contrast_flat[indices], resliced_flat[indices], alpha=0.6, s=1)
        
        # 添加对角线
        min_val = min(contrast_flat.min(), resliced_flat.min())
        max_val = max(contrast_flat.max(), resliced_flat.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        correlation = np.corrcoef(contrast_flat, resliced_flat)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Pearson r = {correlation:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        axes[1, 1].set_xlabel('Contrast Values')
        axes[1, 1].set_ylabel('Resliced Values')
        axes[1, 1].set_title('Correlation Scatter Plot (Sampled)')
    
    # 5. 统计信息
    axes[1, 2].axis('off')
    
    if valid_mask.any():
        contrast_flat = contrast_data[valid_mask].flatten()
        resliced_flat = resliced_data[valid_mask].flatten()
        
        stats_text = f"""
        Statistics Comparison
        
        Contrast Version:
        - Mean: {np.mean(contrast_flat):.6f}
        - Std: {np.std(contrast_flat):.6f}
        - Min: {np.min(contrast_flat):.6f}
        - Max: {np.max(contrast_flat):.6f}
        
        Resliced Version:
        - Mean: {np.mean(resliced_flat):.6f}
        - Std: {np.std(resliced_flat):.6f}
        - Min: {np.min(resliced_flat):.6f}
        - Max: {np.max(resliced_flat):.6f}
        
        Difference Statistics:
        - MAE: {np.mean(np.abs(resliced_flat - contrast_flat)):.6f}
        - Max Diff: {np.max(np.abs(resliced_flat - contrast_flat)):.6f}
        - Correlation: {correlation:.6f}
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图形
    output_file = output_dir / f'{file_name}_atlas_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"保存可视化图形: {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='对比两个图谱文件是否完全一致')
    parser.add_argument('--contrast_dir', required=True, 
                       help='对比版本目录路径')
    parser.add_argument('--resliced_dir', required=True,
                       help='重切片版本目录路径')
    parser.add_argument('--output_dir', default='./atlas_comparison',
                       help='输出结果目录')
    
    args = parser.parse_args()
    
    # 转换路径
    contrast_dir = Path(args.contrast_dir)
    resliced_dir = Path(args.resliced_dir)
    output_dir = Path(args.output_dir)
    
    # 检查目录是否存在
    if not contrast_dir.exists():
        print(f"错误: 对比版本目录不存在: {contrast_dir}")
        return 1
    if not resliced_dir.exists():
        print(f"错误: 重切片版本目录不存在: {resliced_dir}")
        return 1
    
    print(f"对比图谱目录:")
    print(f"对比版本: {contrast_dir}")
    print(f"重切片版本: {resliced_dir}")
    
    # 找到匹配的文件对
    matches = find_matching_files(contrast_dir, resliced_dir)
    
    if not matches:
        print("错误: 未找到匹配的文件对")
        return 1
    
    print(f"\n找到 {len(matches)} 对匹配的文件")
    
    # 对比每对文件
    all_identical = True
    for contrast_file, resliced_file, contrast_name, resliced_name in matches:
        # 对比文件
        is_identical = compare_atlas_files(contrast_file, resliced_file, contrast_name, resliced_name)
        
        if not is_identical:
            all_identical = False
            # 创建可视化
            contrast_img, contrast_data = load_nifti(contrast_file)
            resliced_img, resliced_data = load_nifti(resliced_file)
            
            if contrast_data is not None and resliced_data is not None:
                base_name = contrast_name.replace('.nii.gz', '').replace('_resample', '')
                create_visualization(contrast_data, resliced_data, base_name, output_dir)
    
    print(f"\n{'='*60}")
    if all_identical:
        print("✅ 所有图谱文件完全相同！")
    else:
        print("❌ 发现差异，图谱文件不完全相同")
    print(f"{'='*60}")
    
    return 0 if all_identical else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())