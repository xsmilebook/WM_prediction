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

dataset = 'ABCD'  # 可以修改为 'ABCD' 或其他数据集
targetStr_list = ["nihtbx_cryst_uncorrected", "nihtbx_fluidcomp_uncorrected", "nihtbx_totalcomp_uncorrected"]

for targetStr in targetStr_list:
    # 基础路径配置
    base_path = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}'
    outFolder = f'{base_path}/prediction/{targetStr}'
    os.makedirs(outFolder, exist_ok=True)
    # Import data
    # 1. atlas loading
    GG_datapath = f'{base_path}/fc_vector/{dataset}_GG_vectors.npy'
    GW_datapath = f'{base_path}/fc_vector/{dataset}_GW_vectors.npy'
    WW_datapath = f'{base_path}/fc_vector/{dataset}_WW_vectors.npy'

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

    # 2. subject label: prediction score
    labelpath = f'{base_path}/table/subid_meanFD_age_sex.csv'

    # 检查标签文件是否存在
    if not os.path.exists(labelpath):
        print(f"警告: 标签文件不存在: {labelpath}")
        # 可以在这里添加替代逻辑或抛出异常

    label_files_all = pd.read_csv(labelpath)
    dimention = targetStr 

    # 检查目标列是否存在
    if dimention not in label_files_all.columns:
        print(f"警告: 目标列 '{dimention}' 不存在于标签文件中")
        print(f"可用列: {list(label_files_all.columns)}")
        # 可以在这里添加替代逻辑或抛出异常

    label = label_files_all[dimention]
    y_label = np.array(label)
    OverallPsyFactor = y_label

    # 3. covariates  
    covariatespath = f'{base_path}/table/subid_meanFD_age_sex.csv'

    # 检查协变量文件是否存在
    if not os.path.exists(covariatespath):
        print(f"警告: 协变量文件不存在: {covariatespath}")
        # 可以在这里添加替代逻辑或抛出异常

    Covariates = pd.read_csv(covariatespath, header=0)
    Covariates = Covariates.values

    # 检查是否有足够的列用于协变量
    if Covariates.shape[1] < 4:
        print(f"警告: 协变量文件列数不足。期望至少4列，实际有 {Covariates.shape[1]} 列")
        print(f"将使用可用的列进行协变量处理")
        # 根据实际可用的列调整协变量选择
        if Covariates.shape[1] >= 2:
            Covariates = Covariates[:, [0, 1]].astype(float)  # 使用前两列
        else:
            Covariates = Covariates.astype(float)  # 使用所有列
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

    # # Predict
    ResultantFolder = outFolder + '/RegressCovariates_RandomCV'
    print(f"结果文件夹: {ResultantFolder}")

    # 确保输出目录存在
    os.makedirs(ResultantFolder, exist_ok=True)

    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, CVtimes, ResultantFolder, Parallel_Quantity, 0)

    # Permutation
    # ResultantFolder = outFolder + '/RegressCovariates_RandomCV_Permutation';
    # PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, 1000, ResultantFolder, Parallel_Quantity, 1)