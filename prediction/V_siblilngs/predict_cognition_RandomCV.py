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
from familywise_inputs import filter_and_sort_by_subids, load_feature_subset

dataset = 'ABCD'  # 可以修改为 'ABCD' 或其他数据集
targetStr_list = ["nihtbx_cryst_uncorrected", "nihtbx_fluidcomp_uncorrected", "nihtbx_totalcomp_uncorrected"]
original_sublist_file = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}/table/cognition_sublist.txt'
familywise_sublist_file = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}/table/cognition_sublist_unique_family.txt'

for targetStr in targetStr_list:
    # 基础路径配置
    base_path = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}'
    outFolder = f'{base_path}/prediction/{targetStr}/V_siblilngs'
    os.makedirs(outFolder, exist_ok=True)
    # Import data
    # 1. atlas loading
    GG_datapath = f'{base_path}/cog_fc_vector/{dataset}_GG_vectors.npy'
    GW_datapath = f'{base_path}/cog_fc_vector/{dataset}_GW_vectors.npy'
    WW_datapath = f'{base_path}/cog_fc_vector/{dataset}_WW_vectors.npy'

    # 检查数据文件是否存在
    for filepath, name in [(GG_datapath, 'GG'), (GW_datapath, 'GW'), (WW_datapath, 'WW')]:
        if not os.path.exists(filepath):
            print(f"警告: {name} 数据文件不存在: {filepath}")
            # 可以在这里添加替代逻辑或抛出异常

    SubjectsData, target_subids = load_feature_subset(
        [GG_datapath, GW_datapath, WW_datapath],
        original_sublist_file,
        familywise_sublist_file,
    )
    
    # 2. subject label: prediction score
    labelpath = f'{base_path}/table/nc_y_nihtb_baseline.csv'

    # 检查标签文件是否存在
    if not os.path.exists(labelpath):
        print(f"警告: 标签文件不存在: {labelpath}")
        # 可以在这里添加替代逻辑或抛出异常

    label_files_all = pd.read_csv(labelpath, low_memory=False)
    label_files_filtered = filter_and_sort_by_subids(label_files_all, target_subids)
    
    dimention = targetStr 

    # 检查目标列是否存在
    if dimention not in label_files_filtered.columns:
        print(f"警告: 目标列 '{dimention}' 不存在于标签文件中")
        print(f"可用列: {list(label_files_filtered.columns)}")
        # 可以在这里添加替代逻辑或抛出异常

    label = label_files_filtered[dimention]
    y_label = np.array(label)
    OverallPsyFactor = y_label

    # 3. covariates  
    covariatespath = f'{base_path}/table/subid_meanFD_age_sex.csv'

    # 检查协变量文件是否存在
    if not os.path.exists(covariatespath):
        print(f"警告: 协变量文件不存在: {covariatespath}")
        # 可以在这里添加替代逻辑或抛出异常

    Covariates_all = pd.read_csv(covariatespath, header=0)
    Covariates_filtered = filter_and_sort_by_subids(Covariates_all, target_subids)
    
    # 调试信息：显示实际列数和列名
    print(f"协变量文件列数: {Covariates_filtered.shape[1]}")
    print(f"可用列: {list(Covariates_filtered.columns)}")
    
    # ABCD: 选择age, sex, motion, site列，保持原始数据类型（site是分类变量）
    Covariates_selected = Covariates_filtered.iloc[:, [2, 3, 4, 1]].values  # shape: (n_samples, 4) sex motion site age

    site_labels = Covariates_selected[:, 2]  # 提取 site 列
    site_dict = {site: i for i, site in enumerate(np.unique(site_labels))}
    Covariates_selected[:, 2] = np.array([site_dict[site] for site in site_labels])

    # 4. 转换为 float 类型（确保可用于回归等数值计算）
    Covariates = Covariates_selected.astype(float)


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
    print(f"family-wise 样本量: {len(target_subids)}")
    print(f"组件数量范围: {ComponentNumber_Range}")
    print(f"交叉验证次数: {CVtimes}")

    # # Predict
    ResultantFolder = outFolder + '/RegressCovariates_RandomCV'
    print(f"结果文件夹: {ResultantFolder}")

    # 确保输出目录存在
    os.makedirs(ResultantFolder, exist_ok=True)

    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, CVtimes, ResultantFolder, Parallel_Quantity, 0)

    # Permutation
    ResultantFolder = outFolder + '/RegressCovariates_RandomCV_Permutation'
    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, 1000, ResultantFolder, Parallel_Quantity, 1)
