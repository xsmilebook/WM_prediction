#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较两个版本的功能连接矩阵是否一致，如果不一致计算相关性
Compare FC matrices between contrast and individual versions
"""

import argparse
import logging
import numpy as np
from pathlib import Path
import sys

def setup_logging(output_dir):
    """设置日志"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'compare_fc_versions.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_matrix(file_path):
    """加载矩阵文件"""
    try:
        return np.load(file_path)
    except Exception as e:
        logger.error(f"加载文件失败 {file_path}: {e}")
        return None

def compare_matrices(mat1, mat2, name1, name2):
    """比较两个矩阵"""
    logger.info(f"\n{'='*60}")
    logger.info(f"比较: {name1} vs {name2}")
    logger.info(f"矩阵1形状: {mat1.shape}")
    logger.info(f"矩阵2形状: {mat2.shape}")
    
    # 检查形状
    if mat1.shape != mat2.shape:
        logger.warning(f"形状不匹配！尝试转置...")
        if mat1.shape == mat2.T.shape:
            mat2 = mat2.T
            logger.info("已转置矩阵2，形状现在匹配")
        else:
            logger.error("形状不兼容，无法比较")
            return None
    
    # 检查NaN值
    nan1 = np.isnan(mat1).sum()
    nan2 = np.isnan(mat2).sum()
    if nan1 > 0:
        logger.warning(f"矩阵1包含 {nan1} 个NaN值")
    if nan2 > 0:
        logger.warning(f"矩阵2包含 {nan2} 个NaN值")
    
    # 展平矩阵用于比较
    flat1 = mat1.flatten()
    flat2 = mat2.flatten()
    
    # 处理NaN值
    valid_mask = ~np.isnan(flat1) & ~np.isnan(flat2)
    if not valid_mask.all():
        logger.info(f"忽略 {np.sum(~valid_mask)} 个无效值对")
        flat1 = flat1[valid_mask]
        flat2 = flat2[valid_mask]
    
    if len(flat1) == 0:
        logger.error("没有有效数据可比较")
        return None
    
    # 检查是否完全相同
    if np.allclose(mat1, mat2, equal_nan=True):
        logger.info("结果: 矩阵完全相同！")
        return {"identical": True, "correlation": 1.0, "mae": 0.0}
    else:
        logger.info("结果: 矩阵不同")
        
        # 计算相关性
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        logger.info(f"Pearson相关性: {correlation:.6f}")
        
        # 计算平均绝对误差
        mae = np.mean(np.abs(flat1 - flat2))
        logger.info(f"平均绝对误差: {mae:.6f}")
        
        # 计算最大差异
        max_diff = np.max(np.abs(flat1 - flat2))
        logger.info(f"最大差异: {max_diff:.6f}")
        
        # 计算相对误差
        mean_val = np.mean(np.abs(flat1))
        if mean_val > 0:
            relative_error = mae / mean_val * 100
            logger.info(f"相对误差: {relative_error:.2f}%")
        
        return {
            "identical": False,
            "correlation": correlation,
            "mae": mae,
            "max_diff": max_diff
        }

def find_matching_files(contrast_dir, individual_dir, subject_id):
    """查找匹配的矩阵文件"""
    contrast_files = list(contrast_dir.glob("*.npy"))
    individual_files = list(individual_dir.glob("*.npy"))
    
    # 过滤掉时间序列文件（不比较时间序列）
    contrast_matrices = [f for f in contrast_files if 'timeseries' not in f.name]
    individual_matrices = [f for f in individual_files if 'timeseries' not in f.name]
    
    logger.info(f"\n找到 {len(contrast_matrices)} 个对比版本矩阵文件")
    logger.info(f"找到 {len(individual_matrices)} 个个體版本矩阵文件")
    
    matches = []
    
    # 尝试匹配不同类型的矩阵
    matrix_types = [
        ('GM', 'GG'),
        ('WM', 'WW'), 
        ('GM_WM', 'GW')
    ]
    
    for contrast_type, individual_type in matrix_types:
        contrast_pattern = f"*{contrast_type}*FC.npy"
        individual_pattern = f"*{individual_type}*FC.npy"
        
        contrast_matches = [f for f in contrast_matrices if contrast_type in f.name and 'FC' in f.name]
        individual_matches = [f for f in individual_matrices if individual_type in f.name and 'FC' in f.name]
        
        if contrast_matches and individual_matches:
            # 取第一个匹配的文件
            matches.append((contrast_matches[0], individual_matches[0], contrast_type, individual_type))
    
    return matches

def main():
    parser = argparse.ArgumentParser(description='比较两个版本的功能连接矩阵')
    parser.add_argument('--contrast_dir', required=True, 
                       help='对比版本目录路径')
    parser.add_argument('--individual_dir', required=True,
                       help='个体版本目录路径')
    parser.add_argument('--subject_id', required=True,
                       help='被试ID')
    parser.add_argument('--output_dir', default='./log',
                       help='输出日志目录')
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logging(args.output_dir)
    
    logger.info(f"开始比较被试 {args.subject_id} 的功能连接矩阵")
    logger.info(f"对比版本目录: {args.contrast_dir}")
    logger.info(f"个体版本目录: {args.individual_dir}")
    
    # 转换路径
    contrast_dir = Path(args.contrast_dir)
    individual_dir = Path(args.individual_dir)
    
    # 检查目录是否存在
    if not contrast_dir.exists():
        logger.error(f"对比版本目录不存在: {contrast_dir}")
        return 1
    if not individual_dir.exists():
        logger.error(f"个体版本目录不存在: {individual_dir}")
        return 1
    
    # 查找匹配的矩阵文件
    matches = find_matching_files(contrast_dir, individual_dir, args.subject_id)
    
    if not matches:
        logger.error("未找到匹配的矩阵文件对")
        return 1
    
    logger.info(f"\n找到 {len(matches)} 对匹配的矩阵文件")
    
    # 比较每对矩阵
    results = {}
    for contrast_file, individual_file, contrast_type, individual_type in matches:
        logger.info(f"\n处理矩阵对: {contrast_type} vs {individual_type}")
        
        mat1 = load_matrix(contrast_file)
        mat2 = load_matrix(individual_file)
        
        if mat1 is not None and mat2 is not None:
            result = compare_matrices(mat1, mat2, contrast_file.name, individual_file.name)
            if result is not None:
                results[f"{contrast_type}_vs_{individual_type}"] = result
    
    # 总结结果
    logger.info(f"\n{'='*60}")
    logger.info("总结报告")
    logger.info(f"{'='*60}")
    
    for name, result in results.items():
        logger.info(f"\n{name}:")
        if result["identical"]:
            logger.info("  完全相同")
        else:
            logger.info(f"  不同")
            logger.info(f"  相关性: {result['correlation']:.6f}")
            logger.info(f"  平均绝对误差: {result['mae']:.6f}")
            logger.info(f"  最大差异: {result['max_diff']:.6f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())