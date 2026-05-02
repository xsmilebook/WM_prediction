#coding: utf-8
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import sys

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
# 入口脚本位于 prediction/V_hcppipeline/，PLS 实现在上一级 prediction/ 目录。
sys.path.append(os.path.dirname(script_dir))

import PLSr1_CZ_Random_RegressCovariates

def load_sublist(sublist_path):
    """加载sublist文件"""
    if not os.path.exists(sublist_path):
        print(f"警告: sublist文件不存在: {sublist_path}")
        return None
    
    with open(sublist_path, 'r') as f:
        sublist = [line.strip() for line in f if line.strip()]
    return sublist

def filter_data_by_sublist(data_df, sublist, subid_col='subid'):
    """根据sublist过滤数据"""
    if sublist is None:
        return data_df
    
    # 检查subid列是否存在
    if subid_col not in data_df.columns:
        print(f"警告: 列'{subid_col}'不存在于数据中")
        print(f"可用列: {list(data_df.columns)}")
        return data_df
    
    # 过滤数据
    filtered_data = data_df[data_df[subid_col].isin(sublist)].copy()
    
    # 按照sublist的顺序重新排序
    sublist_order = {subid: i for i, subid in enumerate(sublist)}
    filtered_data['sort_order'] = filtered_data[subid_col].map(sublist_order)
    filtered_data = filtered_data.sort_values('sort_order').drop('sort_order', axis=1)
    
    # 检查是否有缺失的subjects
    missing_subjects = set(sublist) - set(filtered_data[subid_col])
    if missing_subjects:
        print(f"警告: {len(missing_subjects)}个subjects在数据中缺失")
        print(f"缺失样本: {list(missing_subjects)[:5]}...")  # 只显示前5个
    
    return filtered_data

def vectorize_matrix(matrix, matrix_type):
    """将 FC 矩阵转为与 fc_vector 一致的一维特征。"""
    if matrix_type in ['GG', 'WW']:
        row_idx, col_idx = np.triu_indices(matrix.shape[0], k=1)
        return matrix[row_idx, col_idx]
    if matrix_type == 'GW':
        return matrix.reshape(-1)
    raise ValueError(f"未知矩阵类型: {matrix_type}")

def load_vectors_from_individual_z(individual_z_root, sublist, matrix_type):
    """按 sublist 顺序从 individual_z 目录读取并向量化 FC 矩阵。"""
    vectors = []
    loaded_subjects = []
    missing_subjects = []

    for subject_id in sublist:
        matrix_path = os.path.join(
            individual_z_root,
            subject_id,
            f'{subject_id}_{matrix_type}_FC_Z.npy',
        )
        if not os.path.exists(matrix_path):
            missing_subjects.append(subject_id)
            continue

        matrix = np.load(matrix_path)
        vectors.append(vectorize_matrix(matrix, matrix_type))
        loaded_subjects.append(subject_id)

    if missing_subjects:
        print(f"警告: {matrix_type} 缺少 {len(missing_subjects)} 个subjects的矩阵")
        print(f"缺失样本: {missing_subjects[:5]}...")

    if not vectors:
        raise RuntimeError(f"未能从 {individual_z_root} 读取任何 {matrix_type} 向量")

    return np.vstack(vectors), loaded_subjects

# 配置参数 - 可以轻松修改为其他数据集
dataset = 'EFNY'
targetStr = 'age'

# 基础路径配置
base_path = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}'
hcppipeline_base_path = f'{base_path}/hcppipeline_fc'
outFolder = f'{base_path}/prediction/{targetStr}/V_hcppipeline'
os.makedirs(outFolder, exist_ok=True)

sublist_path = f'{base_path}/table/sublist_xcpd_ready505.txt'

# 先读取目标 sublist，再用标签表确定最终可用被试
sublist = load_sublist(sublist_path)
if not sublist:
    raise RuntimeError(f"未能加载 sublist: {sublist_path}")
print(f"加载了 {len(sublist)} 个subjects的sublist")

# 2. subject label: prediction score - 需要根据 sublist 过滤，new是调整过subID后的表格
labelpath = f'{base_path}/table/subid_meanFD_age_sex_new.csv'

# 检查标签文件是否存在
if not os.path.exists(labelpath):
    print(f"警告: 标签文件不存在: {labelpath}")

label_files_all = pd.read_csv(labelpath)
dimention = targetStr 

# 检查目标列是否存在
if dimention not in label_files_all.columns:
    print(f"警告: 目标列 '{dimention}' 不存在于标签文件中")
    print(f"可用列: {list(label_files_all.columns)}")

# 3. covariates - 需要根据 sublist 过滤
covariatespath = f'{base_path}/table/subid_meanFD_age_sex_new.csv'

# 检查协变量文件是否存在
if not os.path.exists(covariatespath):
    print(f"警告: 协变量文件不存在: {covariatespath}")

Covariates_all = pd.read_csv(covariatespath, header=0)
individual_z_root = f'{hcppipeline_base_path}/fc_matrix/individual_z'
available_subjects = []
missing_fc_subjects = []
for subject_id in sublist:
    gg_path = os.path.join(individual_z_root, subject_id, f'{subject_id}_GG_FC_Z.npy')
    gw_path = os.path.join(individual_z_root, subject_id, f'{subject_id}_GW_FC_Z.npy')
    ww_path = os.path.join(individual_z_root, subject_id, f'{subject_id}_WW_FC_Z.npy')
    if os.path.exists(gg_path) and os.path.exists(gw_path) and os.path.exists(ww_path):
        available_subjects.append(subject_id)
    else:
        missing_fc_subjects.append(subject_id)

if missing_fc_subjects:
    print(f"警告: hcppipeline individual_z 中缺少 {len(missing_fc_subjects)} 个subjects")
    print(f"缺失样本: {missing_fc_subjects[:5]}...")

label_subjects = set(label_files_all['subid'])
effective_sublist = [subject_id for subject_id in available_subjects if subject_id in label_subjects]
missing_label_subjects = [subject_id for subject_id in available_subjects if subject_id not in label_subjects]

if missing_label_subjects:
    print(f"警告: 标签表中缺少 {len(missing_label_subjects)} 个subjects")
    print(f"缺失样本: {missing_label_subjects[:5]}...")

if not effective_sublist:
    raise RuntimeError("sublist 中没有同时具备 hcppipeline FC 和标签信息的subjects")

print(f"最终用于预测的subjects数量: {len(effective_sublist)}")

label_files_filtered = filter_data_by_sublist(label_files_all, effective_sublist)
covariates_filtered = filter_data_by_sublist(Covariates_all, effective_sublist)
print(f"标签数据过滤后剩余 {len(label_files_filtered)} 个subjects")
print(f"协变量过滤后剩余 {len(covariates_filtered)} 个subjects")

# Import data
# 1. atlas loading - 直接从 hcppipeline individual_z 向量化
GG_data_files, gg_subjects = load_vectors_from_individual_z(individual_z_root, effective_sublist, 'GG')
GW_data_files, gw_subjects = load_vectors_from_individual_z(individual_z_root, effective_sublist, 'GW')
WW_data_files, ww_subjects = load_vectors_from_individual_z(individual_z_root, effective_sublist, 'WW')

if gg_subjects != effective_sublist or gw_subjects != effective_sublist or ww_subjects != effective_sublist:
    raise RuntimeError("向量化后的 subjects 顺序与 effective_sublist 不一致")

print(f"向量数据形状: GG={GG_data_files.shape}, GW={GW_data_files.shape}, WW={WW_data_files.shape}")

SubjectsData = []
SubjectsData.append(GG_data_files)
SubjectsData.append(GW_data_files)
SubjectsData.append(WW_data_files)

label = label_files_filtered[dimention]
y_label = np.array(label)
OverallPsyFactor = y_label

Covariates = covariates_filtered.values

# 添加调试信息来检查数据质量
print(f"标签数据范围: min={np.min(y_label)}, max={np.max(y_label)}, mean={np.mean(y_label):.2f}, std={np.std(y_label):.2f}")
Covariates = Covariates[:, [2, 3]].astype(float)  # sex, motion

# subID,age,sex,meanFD
# Range of parameters
ComponentNumber_Range = np.arange(10) + 1
FoldQuantity = 5
Parallel_Quantity = 1
CVtimes = 101

print(f"数据集: {dataset}")
print(f"目标变量: {targetStr}")
print(f"数据路径: {base_path}")
print(f"HCP pipeline FC 路径: {individual_z_root}")
print(f"输入 sublist: {sublist_path}")
print(f"输出路径: {outFolder}")
print(f"组件数量范围: {ComponentNumber_Range}")
print(f"交叉验证次数: {CVtimes}")
print(f"最终样本数量: {len(OverallPsyFactor)}")

# # Predict
ResultantFolder = outFolder + '/RegressCovariates_RandomCV'
print(f"结果文件夹: {ResultantFolder}")

# 确保输出目录存在
os.makedirs(ResultantFolder, exist_ok=True)

PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, CVtimes, ResultantFolder, Parallel_Quantity, 0)

# Permutation
# ResultantFolder = outFolder + '/RegressCovariates_RandomCV_Permutation';
# PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, 1000, ResultantFolder, Parallel_Quantity, 1)
