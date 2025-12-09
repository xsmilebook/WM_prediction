#coding: utf-8
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/test_confounds_regress');
import PLSr1_CZ_Random_RegressCovariates

targetStr = 'interview_age'
outFolder = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/prediction/test_confounds_regress/'+ targetStr

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

# Predict
ResultantFolder = outFolder + '/RegressCovariates_RandomCV'
PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, CVtimes, ResultantFolder, Parallel_Quantity, 0)

# Permutation
# ResultantFolder = outFolder + '/RegressCovariates_RandomCV_Permutation';
# PLSr1_CZ_Random_RegressCovariates.PLSr1_KFold_RandomCV_MultiTimes(SubjectsData, OverallPsyFactor, Covariates, FoldQuantity, ComponentNumber_Range, 1000, ResultantFolder, Parallel_Quantity, 1)


