#coding: utf-8
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import sys

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

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

# 配置参数 - 可以轻松修改为其他数据集
dataset = 'HCPD'
targetStr = 'age'

# 基础路径配置
base_path = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}'
outFolder = f'{base_path}/prediction/{targetStr}'
os.makedirs(outFolder, exist_ok=True)

# 添加sublist路径配置
sublist_path = f'{base_path}/table/sublist.txt'  # 假设sublist文件名为age_sublist.txt

# Import data
# 1. atlas loading - 向量文件已根据sublist生成，直接使用
GG_datapath = f'{base_path}/fc_vector/{dataset}_GG_vectors.npy'
GW_datapath = f'{base_path}/fc_vector/{dataset}_GW_vectors.npy'
WW_datapath = f'{base_path}/fc_vector/{dataset}_WW_vectors.npy'

# 检查数据文件是否存在
for filepath, name in [(GG_datapath, 'GG'), (GW_datapath, 'GW'), (WW_datapath, 'WW')]:
    if not os.path.exists(filepath):
        print(f"警告: {name} 数据文件不存在: {filepath}")

# 直接加载向量数据（已根据sublist生成）
GG_data_files = np.load(GG_datapath)
GW_data_files = np.load(GW_datapath)
WW_data_files = np.load(WW_datapath)

print(f"向量数据形状: GG={GG_data_files.shape}, GW={GW_data_files.shape}, WW={WW_data_files.shape}")

SubjectsData = []
SubjectsData.append(GG_data_files)
SubjectsData.append(GW_data_files)
SubjectsData.append(WW_data_files)

# 2. subject label: prediction score - 需要根据sublist过滤
labelpath = f'{base_path}/table/subid_meanFD_age_sex_site.csv'

# 检查标签文件是否存在
if not os.path.exists(labelpath):
    print(f"警告: 标签文件不存在: {labelpath}")

label_files_all = pd.read_csv(labelpath)
dimention = targetStr 

# 检查目标列是否存在
if dimention not in label_files_all.columns:
    print(f"警告: 目标列 '{dimention}' 不存在于标签文件中")
    print(f"可用列: {list(label_files_all.columns)}")

# 加载sublist并过滤标签数据
sublist = load_sublist(sublist_path)
if sublist:
    print(f"加载了 {len(sublist)} 个subjects的sublist")
    label_files_filtered = filter_data_by_sublist(label_files_all, sublist)
    print(f"标签数据过滤后剩余 {len(label_files_filtered)} 个subjects")
else:
    print("未找到sublist文件，使用所有标签数据")
    label_files_filtered = label_files_all

label = label_files_filtered[dimention]
y_label = np.array(label)
OverallPsyFactor = y_label

# 3. covariates - 需要根据sublist过滤  
covariatespath = f'{base_path}/table/subid_meanFD_age_sex_site.csv'

# 检查协变量文件是否存在
if not os.path.exists(covariatespath):
    print(f"警告: 协变量文件不存在: {covariatespath}")

Covariates_all = pd.read_csv(covariatespath, header=0)

# 根据sublist过滤协变量数据
if sublist:
    covariates_filtered = filter_data_by_sublist(Covariates_all, sublist)
    print(f"协变量过滤后剩余 {len(covariates_filtered)} 个subjects")
else:
    covariates_filtered = Covariates_all

# 确保标签和协变量数据对齐
if len(label_files_filtered) != len(covariates_filtered):
    print(f"警告: 标签数据({len(label_files_filtered)})和协变量数据({len(covariates_filtered)})长度不一致")
    # 取交集
    common_subids = set(label_files_filtered['subid']) & set(covariates_filtered['subid'])
    if common_subids:
        label_files_filtered = filter_data_by_sublist(label_files_filtered, list(common_subids))
        covariates_filtered = filter_data_by_sublist(covariates_filtered, list(common_subids))
        print(f"取交集后剩余 {len(common_subids)} 个subjects")

# 重新提取标签和协变量
label = label_files_filtered[dimention]
y_label = np.array(label)
OverallPsyFactor = y_label

Covariates = covariates_filtered.values

# 检查是否有足够的列用于协变量
if Covariates.shape[1] < 4:
    print(f"警告: 协变量文件列数不足。期望至少4列，实际有 {Covariates.shape[1]} 列")
    print(f"将使用可用的列进行协变量处理")
    # 根据实际可用的列调整协变量选择
    if Covariates.shape[1] >= 2:
        Covariates = Covariates[:, [0, 1]].astype(float)  # 使用前两列
    else:
        Covariates = Covariates.astype(float)  # 使用所有列
elif dataset == "HCPD":
    Covariates = Covariates[:, [2, 3, 4]]  # sex, motion, site
else:
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