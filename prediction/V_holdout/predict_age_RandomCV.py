#coding: utf-8
import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import StratifiedKFold

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

    if subid_col not in data_df.columns:
        print(f"警告: 列'{subid_col}'不存在于数据中")
        print(f"可用列: {list(data_df.columns)}")
        return data_df

    filtered_data = data_df[data_df[subid_col].isin(sublist)].copy()

    sublist_order = {subid: i for i, subid in enumerate(sublist)}
    filtered_data['sort_order'] = filtered_data[subid_col].map(sublist_order)
    filtered_data = filtered_data.sort_values('sort_order').drop('sort_order', axis=1)

    missing_subjects = set(sublist) - set(filtered_data[subid_col])
    if missing_subjects:
        print(f"警告: {len(missing_subjects)}个subjects在数据中缺失")
        print(f"缺失样本: {list(missing_subjects)[:5]}...")

    return filtered_data


def build_stratification_labels(subjects_score, n_splits):
    subjects_score = np.asarray(subjects_score).reshape(-1)
    max_bins = min(10, len(np.unique(subjects_score)))
    for bin_count in range(max_bins, 1, -1):
        labels = pd.qcut(subjects_score, q=bin_count, labels=False, duplicates='drop')
        labels = np.asarray(labels, dtype=int)
        bincount = np.bincount(labels)
        if bincount.size >= 2 and np.min(bincount) >= n_splits:
            return labels
    raise ValueError('Unable to create stratification labels for half-split.')


def save_half_split(subjects_score, output_file, random_state=1234):
    labels = build_stratification_labels(subjects_score, n_splits=2)
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    train_index, test_index = next(splitter.split(np.zeros(len(labels)), labels))
    sio.savemat(
        output_file,
        {
            'Train_Index': train_index.astype(int),
            'Test_Index': test_index.astype(int),
        },
    )


parser = argparse.ArgumentParser(
    description='Run non-ABCD age holdout prediction with an optional reproducible seed.'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='EFNY',
    choices=['EFNY', 'HCPD', 'CCNP', 'PNC'],
    help='Dataset for age holdout prediction.',
)
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='Optional seed controlling the outer holdout split and inner CV shuffles.',
)
args = parser.parse_args()

dataset = args.dataset
targetStr = 'age'
seed = args.seed

if seed is not None:
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(1001)
    observed_random_seeds = [int(child_sequences[0].generate_state(1, dtype=np.uint32)[0])]
    permutation_random_seeds = [
        int(child_sequence.generate_state(1, dtype=np.uint32)[0])
        for child_sequence in child_sequences[1:]
    ]
else:
    observed_random_seeds = ''
    permutation_random_seeds = ''

# 基础路径配置
base_path = f'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/{dataset}'
if seed is None:
    outFolder = f'{base_path}/prediction/{targetStr}/V_holdout'
else:
    outFolder = f'{base_path}/prediction/{targetStr}/V_holdout_{seed}'
os.makedirs(outFolder, exist_ok=True)

sublist_path = f'{base_path}/table/sublist.txt'

# Import data
# 1. atlas loading - 向量文件已根据sublist生成，直接使用
GG_datapath = f'{base_path}/fc_vector/{dataset}_GG_vectors.npy'
GW_datapath = f'{base_path}/fc_vector/{dataset}_GW_vectors.npy'
WW_datapath = f'{base_path}/fc_vector/{dataset}_WW_vectors.npy'

for filepath, name in [(GG_datapath, 'GG'), (GW_datapath, 'GW'), (WW_datapath, 'WW')]:
    if not os.path.exists(filepath):
        print(f"警告: {name} 数据文件不存在: {filepath}")

GG_data_files = np.load(GG_datapath)
GW_data_files = np.load(GW_datapath)
WW_data_files = np.load(WW_datapath)

print(f"向量数据形状: GG={GG_data_files.shape}, GW={GW_data_files.shape}, WW={WW_data_files.shape}")

SubjectsData = []
SubjectsData.append(GG_data_files)
SubjectsData.append(GW_data_files)
SubjectsData.append(WW_data_files)

# 2. subject label: prediction score - 需要根据sublist过滤
if dataset == "HCPD":
    labelpath = f'{base_path}/table/subid_meanFD_age_sex_site.csv'
else:
    labelpath = f'{base_path}/table/subid_meanFD_age_sex.csv'

if not os.path.exists(labelpath):
    print(f"警告: 标签文件不存在: {labelpath}")

label_files_all = pd.read_csv(labelpath)
dimention = targetStr

if dimention not in label_files_all.columns:
    print(f"警告: 目标列 '{dimention}' 不存在于标签文件中")
    print(f"可用列: {list(label_files_all.columns)}")

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
if dataset == "HCPD":
    covariatespath = f'{base_path}/table/subid_meanFD_age_sex_site.csv'
else:
    covariatespath = f'{base_path}/table/subid_meanFD_age_sex.csv'

if not os.path.exists(covariatespath):
    print(f"警告: 协变量文件不存在: {covariatespath}")

Covariates_all = pd.read_csv(covariatespath, header=0)

if sublist:
    covariates_filtered = filter_data_by_sublist(Covariates_all, sublist)
    print(f"协变量过滤后剩余 {len(covariates_filtered)} 个subjects")
else:
    covariates_filtered = Covariates_all

if len(label_files_filtered) != len(covariates_filtered):
    print(f"警告: 标签数据({len(label_files_filtered)})和协变量数据({len(covariates_filtered)})长度不一致")
    common_subids = set(label_files_filtered['subid']) & set(covariates_filtered['subid'])
    if common_subids:
        common_subids = [subid for subid in sublist if subid in common_subids]
        label_files_filtered = filter_data_by_sublist(label_files_filtered, common_subids)
        covariates_filtered = filter_data_by_sublist(covariates_filtered, common_subids)
        print(f"取交集后剩余 {len(common_subids)} 个subjects")

label = label_files_filtered[dimention]
y_label = np.array(label)
OverallPsyFactor = y_label

Covariates = covariates_filtered.values

print(f"标签数据范围: min={np.min(y_label)}, max={np.max(y_label)}, mean={np.mean(y_label):.2f}, std={np.std(y_label):.2f}")
if dataset == "HCPD":
    print(f"协变量维度: {Covariates.shape}")
    print(f"站点信息: {sorted(set(Covariates[:, 4])) if Covariates.shape[1] > 4 else 'No site info'}")
    print(f"性别分布: {np.unique(Covariates[:, 2] if Covariates.shape[1] > 2 else 'No sex info')}")

if dataset == "HCPD":
    Covariates_selected = Covariates[:, [2, 3, 4]]  # sex, motion, site
    site_dict = {site: i for i, site in enumerate(np.unique(Covariates_selected[:, 2]))}
    Covariates_selected[:, 2] = np.array([site_dict[site] for site in Covariates_selected[:, 2]])
    Covariates = Covariates_selected.astype(float)
else:
    Covariates = Covariates[:, [2, 3]].astype(float)  # sex, motion

# subID,age,sex,meanFD
# Range of parameters
ComponentNumber_Range = np.arange(10) + 1
FoldQuantity = 5
Parallel_Quantity = 1
CVtimes = 1

print(f"数据集: {dataset}")
print(f"目标变量: {targetStr}")
print(f"数据路径: {base_path}")
print(f"输出路径: {outFolder}")
if seed is not None:
    print(f"随机种子: {seed}")
print(f"组件数量范围: {ComponentNumber_Range}")
print("Holdout 划分: stratified half-split train/test = 1:1")
print("超参数选择: outer train half 内部 5-fold CV")
print(f"重复次数: {CVtimes}")
print(f"最终样本数量: {len(OverallPsyFactor)}")

shared_split_file = os.path.join(outFolder, 'SharedSplitIndex.mat')
if seed is None:
    save_half_split(OverallPsyFactor, shared_split_file)
else:
    save_half_split(OverallPsyFactor, shared_split_file, random_state=seed)
print(f"固定 holdout 划分文件: {shared_split_file}")
observed_split_files = [shared_split_file]
permutation_split_files = [shared_split_file] * 1000

# Predict
ResultantFolder = outFolder + '/RegressCovariates_Holdout'
print(f"结果文件夹: {ResultantFolder}")
os.makedirs(ResultantFolder, exist_ok=True)

if seed is None:
    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(
        SubjectsData,
        OverallPsyFactor,
        Covariates,
        FoldQuantity,
        ComponentNumber_Range,
        CVtimes,
        ResultantFolder,
        Parallel_Quantity,
        0,
        observed_split_files,
    )
else:
    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(
        SubjectsData,
        OverallPsyFactor,
        Covariates,
        FoldQuantity,
        ComponentNumber_Range,
        CVtimes,
        ResultantFolder,
        Parallel_Quantity,
        0,
        observed_split_files,
        observed_random_seeds,
    )

# Permutation
ResultantFolder = outFolder + '/RegressCovariates_Holdout_Permutation'
print(f"结果文件夹: {ResultantFolder}")
os.makedirs(ResultantFolder, exist_ok=True)
if seed is None:
    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(
        SubjectsData,
        OverallPsyFactor,
        Covariates,
        FoldQuantity,
        ComponentNumber_Range,
        1000,
        ResultantFolder,
        Parallel_Quantity,
        1,
        permutation_split_files,
    )
else:
    PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(
        SubjectsData,
        OverallPsyFactor,
        Covariates,
        FoldQuantity,
        ComponentNumber_Range,
        1000,
        ResultantFolder,
        Parallel_Quantity,
        1,
        permutation_split_files,
        permutation_random_seeds,
    )
