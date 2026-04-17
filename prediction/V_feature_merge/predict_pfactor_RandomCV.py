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

dataset = 'ABCD'
targetStr_list = ['General','Ext','ADHD','Int']
sublist_file = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}/table/pfactor_sublist.txt'

for targetStr in targetStr_list:
    # 基础路径配置
    base_path = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}'
    outFolder = f'{base_path}/prediction/{targetStr}'
    os.makedirs(outFolder, exist_ok=True)
    # Import data
    # 1. atlas loading
    GG_datapath = f'{base_path}/pfactor_fc_vector/{dataset}_GG_vectors.npy'
    GW_datapath = f'{base_path}/pfactor_fc_vector/{dataset}_GW_vectors.npy'
    WW_datapath = f'{base_path}/pfactor_fc_vector/{dataset}_WW_vectors.npy'

    # 检查数据文件是否存在
    for filepath, name in [(GG_datapath, 'GG'), (GW_datapath, 'GW'), (WW_datapath, 'WW')]:
        if not os.path.exists(filepath):
            print(f"警告: {name} 数据文件不存在: {filepath}")
            # 可以在这里添加替代逻辑或抛出异常

    GG_data_files_all = np.load(GG_datapath)
    
    GW_data_files_all = np.load(GW_datapath)
    
    WW_data_files_all = np.load(WW_datapath)

    SubjectsData = []
    SubjectsData.append(GG_data_files_all)
    SubjectsData.append(GW_data_files_all)
    SubjectsData.append(WW_data_files_all)

    # 读取sublist以确保数据顺序一致
    with open(sublist_file, 'r') as f:
        target_subids = [line.strip() for line in f if line.strip()]
    
    # 2. subject label: prediction score
    labelpath = f'{base_path}/table/Pfactor_score_wx.csv'

    # 检查标签文件是否存在
    if not os.path.exists(labelpath):
        print(f"警告: 标签文件不存在: {labelpath}")
        # 可以在这里添加替代逻辑或抛出异常

    label_files_all = pd.read_csv(labelpath, low_memory=False)
    
    # 根据sublist筛选和排序数据
    # 确保subid列存在
    if 'subid' not in label_files_all.columns:
        print(f"警告: 标签文件中缺少subid列")
        print(f"可用列: {list(label_files_all.columns)}")
        # 尝试从src_subject_id生成subid
        if 'src_subject_id' in label_files_all.columns:
            label_files_all['subid'] = label_files_all['src_subject_id'].apply(lambda x: f"sub-{x.replace('_', '')}")
        else:
            print(f"错误: 标签文件中既无subid列也无src_subject_id列")
            sys.exit(1)
    
    # 只保留sublist中的subjects并按sublist顺序排序
    label_files_filtered = label_files_all[label_files_all['subid'].isin(target_subids)].copy()
    label_files_filtered['sort_order'] = label_files_filtered['subid'].map({subid: i for i, subid in enumerate(target_subids)})
    label_files_filtered = label_files_filtered.sort_values('sort_order').drop('sort_order', axis=1)
    
    # 检查是否有缺失的subjects
    missing_subjects = set(target_subids) - set(label_files_filtered['subid'])
    if missing_subjects:
        print(f"警告: 以下subjects在标签文件中缺失: {missing_subjects}")
        print(f"将使用可用的{len(label_files_filtered)}个subjects进行预测")
    
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
    
    # 根据sublist筛选和排序协变量数据
    # 确保subid列存在
    if 'subid' not in Covariates_all.columns:
        print(f"警告: 协变量文件中缺少subid列")
        print(f"可用列: {list(Covariates_all.columns)}")
        sys.exit(1)
    
    # 只保留sublist中的subjects并按sublist顺序排序
    Covariates_filtered = Covariates_all[Covariates_all['subid'].isin(target_subids)].copy()
    Covariates_filtered['sort_order'] = Covariates_filtered['subid'].map({subid: i for i, subid in enumerate(target_subids)})
    Covariates_filtered = Covariates_filtered.sort_values('sort_order').drop('sort_order', axis=1)
    
    # 检查是否有缺失的subjects
    missing_covariates = set(target_subids) - set(Covariates_filtered['subid'])
    if missing_covariates:
        print(f"警告: 以下subjects在协变量文件中缺失: {missing_covariates}")
        print(f"将使用可用的{len(Covariates_filtered)}个subjects进行预测")
    
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
    print(f"组件数量范围: {ComponentNumber_Range}")
    print(f"交叉验证次数: {CVtimes}")

    # # Predict
    ResultantFolder = outFolder + '/RegressCovariates_RandomCV'
    print(f"结果文件夹: {ResultantFolder}")

    # 确保输出目录存在
    os.makedirs(ResultantFolder, exist_ok=True)

    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, CVtimes, ResultantFolder, Parallel_Quantity, 0)

    # Permutation
    # ResultantFolder = outFolder + '/RegressCovariates_RandomCV_Permutation';
    # PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, 1000, ResultantFolder, Parallel_Quantity, 1)