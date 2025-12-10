#coding: utf-8
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import glob
import sys
sys.path.append('/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/test_confounds_regress');
import PLSr1_CZ_Random_RegressCovariates

targetStr = 'interview_age'
outFolder = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/prediction/test_confounds_regress_age_year_fix_randindex/'+ targetStr

# Import data
# 1. atlas loading
GG_datapath = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/table/test_confounds_regress/total_FCvector_GG.txt';
GW_datapath = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/table/test_confounds_regress/total_FCvector_GW.txt';
WW_datapath = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/table/test_confounds_regress/total_FCvector_WW.txt';

GG_data_files_all = np.loadtxt(GG_datapath,delimiter=",")
GG_data_files_all = np.float32(GG_data_files_all) 
GW_data_files_all = np.loadtxt(GW_datapath,delimiter=",")
GW_data_files_all = np.float32(GW_data_files_all)
WW_data_files_all = np.loadtxt(WW_datapath,delimiter=",")
WW_data_files_all = np.float32(WW_data_files_all)

SubjectsData = []
SubjectsData.append(GG_data_files_all)
SubjectsData.append(GW_data_files_all)
SubjectsData.append(WW_data_files_all)

# 2. subject label: prediction score
labelpath =  '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/table/test_confounds_regress/HCPD_demoCombined_531.csv';
label_files_all = pd.read_csv(labelpath)
dimention = targetStr 
label = label_files_all[dimention]
y_label = np.array(label)
y_label = y_label / 12
OverallPsyFactor = y_label

# 3. covariates  
covariatespath = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/table/test_confounds_regress/HCPD_covariates_531.csv';
Covariates = pd.read_csv(covariatespath, header=0)
Covariates = Covariates.values
Covariates = Covariates[:, [1,2,3]].astype(float) # sex, motion
# subID,age,sex,meanFD
# Range of parameters
ComponentNumber_Range = np.arange(10) + 1
FoldQuantity = 5
Parallel_Quantity = 1
CVtimes = 101

rand_root = r'/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/prediction/age/RegressCovariates_RandomCV'

# 2) 找出所有 Time_i 里的 RandIndex.mat
randindex_files = sorted(
    glob.glob(os.path.join(rand_root, 'Time_*', 'RandIndex.mat')),
    key=lambda x: int(os.path.basename(os.path.dirname(x)).split('_')[-1])  # Time_i -> i
)

print(f'找到 {len(randindex_files)} 份 RandIndex')

# Predict
ResultantFolder = outFolder + '/RegressCovariates_RandomCV'
PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, CVtimes, ResultantFolder, Parallel_Quantity, 0, randindex_files)

# Permutation
# ResultantFolder = outFolder + '/RegressCovariates_RandomCV_Permutation';
# PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, 1000, ResultantFolder, Parallel_Quantity, 1)


