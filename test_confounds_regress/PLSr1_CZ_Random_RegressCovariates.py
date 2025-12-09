# -*- coding: utf-8 -*-
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import time
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_decomposition
from joblib import Parallel, delayed
import statsmodels.formula.api as sm
CodesPath = '/ibmgpfs/cuizaixu_lab/congjing/WM_prediction/HCPD/code/4th_prediction/s02_PLSprediction/nosmooth';
 
def PLSr1_KFold_RandomCV_MultiTimes(Subjects_Data_List, Subjects_Score, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes, ResultantFolder, Parallel_Quantity, Permutation_Flag, RandIndex_File_List=''):
    """
    PLSr1_KFold_RandomCV_MultiTimes: The function is used to do the PLS regression with K-fold cross-validation and random permutation.
    Parameters:
    Subjects_Data: list of FC matices, [GG_FC, GW_FC, WW_FC]
    ResultantFolder: the folder to save the results
    """
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)

    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']

    Subjects_Data_Mat_Path_List = []
    for i in np.arange(len(Subjects_Data_List)):
        Subjects_Data_Mat = {'Subjects_Data': Subjects_Data_List[i]}
        Subjects_Data_Mat_Path = ResultantFolder + '/' + conn_cate_list[i] + '_Subjects_Data.mat'
        Subjects_Data_Mat_Path_List.append(Subjects_Data_Mat_Path)
        sio.savemat(Subjects_Data_Mat_Path, Subjects_Data_Mat)

    Finish_File = []
    Corr_MTimes = np.zeros(CVRepeatTimes)
    MAE_MTimes = np.zeros(CVRepeatTimes)
    for i in np.arange(CVRepeatTimes):
        ResultantFolder_TimeI = ResultantFolder + '/Time_' + str(i)
        if not os.path.exists(ResultantFolder_TimeI):
            os.makedirs(ResultantFolder_TimeI);
            
        if RandIndex_File_List != '':
            RandIndex_File = RandIndex_File_List[i]
        else:
            RandIndex_File = ''

        # MARK: modify(Subjects_Data_Mat_Path_List, ResultsFolder_TimeI_List)
        Configuration_Mat = {'Subjects_Data_Mat_Path_List': Subjects_Data_Mat_Path_List,   'Subjects_Score': Subjects_Score, 'Covariates': Covariates, 'Fold_Quantity': Fold_Quantity, 'ComponentNumber_Range': ComponentNumber_Range, 'CVRepeatTimes': CVRepeatTimes, 'ResultantFolder_TimeI': ResultantFolder_TimeI, 'Parallel_Quantity': Parallel_Quantity, 'Permutation_Flag': Permutation_Flag, 'RandIndex_File': RandIndex_File}

        sio.savemat(ResultantFolder_TimeI + '/Configuration.mat', Configuration_Mat)

        system_cmd = 'python3 -c ' + '\'import sys;\
            sys.path.append("' + CodesPath + '");\
            from PLSr1_CZ_Random_RegressCovariates import PLSr1_KFold_RandomCV_MultiTimes_Sub; \
            import os;\
            import scipy.io as sio;\
            Configuration = sio.loadmat("' + ResultantFolder_TimeI + '/Configuration.mat");\
            Subjects_Data_Mat_Path_List = Configuration["Subjects_Data_Mat_Path_List"];\
            Subjects_Score = Configuration["Subjects_Score"];\
            Covariates = Configuration["Covariates"];\
            Fold_Quantity = Configuration["Fold_Quantity"];\
            ComponentNumber_Range = Configuration["ComponentNumber_Range"];\
            ResultantFolder_TimeI = Configuration["ResultantFolder_TimeI"];\
            Permutation_Flag = Configuration["Permutation_Flag"];\
            RandIndex_File = Configuration["RandIndex_File"];\
            Parallel_Quantity = Configuration["Parallel_Quantity"];\
            PLSr1_KFold_RandomCV_MultiTimes_Sub(Subjects_Data_Mat_Path_List, Subjects_Score[0], Covariates, Fold_Quantity[0][0], ComponentNumber_Range[0], 20, ResultantFolder_TimeI[0], Parallel_Quantity[0][0], Permutation_Flag[0][0], RandIndex_File)\' '
        system_cmd = system_cmd + ' > "' + ResultantFolder_TimeI + '/Time_' + str(i) + '.log" 2>&1\n'
        Finish_File.append(ResultantFolder_TimeI + '/Res_NFold.mat');
        script = open(ResultantFolder_TimeI + '/script.sh', 'w');      
        # # Submit jobs
        script.write('#!/bin/bash\n');
        #script.write('#SBATCH --job-name=PLSca' + str(i) + '\n');
        script.write('#SBATCH --job-name=HCPD' + str(i) + '\n');
        script.write('#SBATCH --nodes=1\n');
        script.write('#SBATCH --ntasks=1\n');
        script.write('#SBATCH --cpus-per-task=2\n');
        # script.write('#SBATCH --mem-per-cpu 5G\n');
        script.write('#SBATCH -p q_cn\n');
        # script.write('#SBATCH -q high_c\n');
        script.write('#SBATCH -o ' + ResultantFolder_TimeI + '/job.%j.out\n');
        script.write('#SBATCH -e ' + ResultantFolder_TimeI + '/job.%j.error.txt\n\n');
        script.write(system_cmd);
        script.close();
        os.system('chmod +x ' + ResultantFolder_TimeI + '/script.sh');
        os.system('sbatch ' + ResultantFolder_TimeI + '/script.sh');
        # # Submit jobs
        # Option = ' -V -o "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.o" -e "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.e" ';
        # #os.system('chmod +x ' + ResultantFolder_TimeI + '/script.sh');
        # os.system(' -l h_vmem=10G,s_vmem=10G -q ' + Queue + ' -N RandomCV_' + str(i) + Option + ResultantFolder_TimeI + '/script.sh')  
        # #os.system('qsub -l h_vmem=10G ' + ResultantFolder_TimeI + '/script.sh')

def PLSr1_KFold_RandomCV_MultiTimes_Sub(Subjects_Data_Mat_Path_List, Subjects_Score, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes, ResultantFolder, Parallel_Quantity, Permutation_Flag, RandIndex_File=''):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    Subjects_Data_List = []
    for i in range(len(Subjects_Data_Mat_Path_List)):
        data = sio.loadmat(Subjects_Data_Mat_Path_List[i])
        Subjects_Data = data['Subjects_Data']
        Subjects_Data_List.append(Subjects_Data)

    PLSr1_KFold_RandomCV(Subjects_Data_List, Subjects_Score, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes, ResultantFolder, Parallel_Quantity, Permutation_Flag, RandIndex_File);

def PLSr1_KFold_RandomCV(Subjects_Data_List, Subjects_Score, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity, Permutation_Flag, RandIndex_File=''):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    ResultantFolder_List = []
    for conn_cate in conn_cate_list:
        ResultantFolder_FC = os.path.join(ResultantFolder, conn_cate)
        if not os.path.exists(ResultantFolder_FC):
            os.makedirs(ResultantFolder_FC)
        ResultantFolder_List.append(ResultantFolder_FC)
    
    Subjects_Quantity = len(Subjects_Score)
    EachFold_Size = int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    if len(RandIndex_File) == 0:
        # MARK: 分层抽样：使各个数据集的y_label分布尽量相似
        sorted_indices_desc = np.argsort(Subjects_Score)[::-1]

        N = len(sorted_indices_desc)
        m  = N % Fold_Quantity
        num_bins = (N + Fold_Quantity - 1) // Fold_Quantity

        bins = [
            sorted_indices_desc[i*Fold_Quantity : (i+1)*Fold_Quantity]
            for i in range(num_bins)
        ]
        shuffled_bins = [np.random.permutation(bin_i) for bin_i in bins]
        RandIndex = np.zeros(N)
        print('num_bins:', num_bins)
        if m== 0:
            for j in range(num_bins):
                for i in range(Fold_Quantity):
                    RandIndex[i * num_bins + j] = shuffled_bins[j][i]
        else:
            for j in range(num_bins-1):
                for i in range(Fold_Quantity):
                    if i * (num_bins-1) + j >= N:
                        continue
                    RandIndex[i * (num_bins-1) + j] = shuffled_bins[j][i]
            for k in range(len(shuffled_bins[-1])):
                RandIndex[(num_bins - 1) * Fold_Quantity + k] = shuffled_bins[-1][k]
        
        RandIndex = RandIndex.astype(int)

    else:
        tmpData = sio.loadmat(RandIndex_File[0])
        RandIndex = tmpData['RandIndex'][0];
    RandIndex_Mat = {'RandIndex': RandIndex}
    sio.savemat(ResultantFolder + '/RandIndex.mat', RandIndex_Mat);
    
    Fold_Corr = [];
    Fold_MAE = [];
    Fold_Weight = [];
    Features_Quantity_List = []
    for conn_index in np.arange(len(Subjects_Data_List)):
        Features_Quantity = np.shape(Subjects_Data_List[conn_index])[1]
        Features_Quantity_List.append(Features_Quantity)

    # split dataset into K folds         
    for j in np.arange(Fold_Quantity):
        Fold_J_Index = RandIndex[EachFold_Size * j + np.arange(EachFold_Size)]
        if Remain > j:
            Fold_J_Index = np.insert(Fold_J_Index, len(Fold_J_Index), RandIndex[EachFold_Size * Fold_Quantity + j])

        Subjects_Data_test_List = []
        Subjects_Score_test = Subjects_Score[Fold_J_Index]
        Subjects_Data_train_List = []
        Subjects_Score_train = np.delete(Subjects_Score, Fold_J_Index)
        for conn_index in np.arange(len(Subjects_Data_List)):
            Subjects_Data_test = Subjects_Data_List[conn_index][Fold_J_Index, :]
            Subjects_Data_train = np.delete(Subjects_Data_List[conn_index], Fold_J_Index, axis=0) # delete the test data from the training data
            Subjects_Data_test_List.append(Subjects_Data_test)
            Subjects_Data_train_List.append(Subjects_Data_train)
            Covariates_test = Covariates[Fold_J_Index, :]
            Covariates_train = np.delete(Covariates, Fold_J_Index, axis=0)

# ========= 回归协变量：
            Covariates_Quantity = Covariates_train.shape[1]

            # 先构造训练和测试的 DataFrame，列名统一用 Covariate_0/1/2...
            df = pd.DataFrame({
                f'Covariate_{k}': Covariates_train[:, k]
                for k in range(Covariates_Quantity)
            })
            df_test = pd.DataFrame({
                f'Covariate_{k}': Covariates_test[:, k]
                for k in range(Covariates_Quantity)
            })

            # 这里约定：Covariates = [sex, motion, site]，
            # 所以 0 和 2 列是类别变量（sex, site），1 列是连续变量（motion）
            categorical_idx = [0, 2]

            # LOGGING: 输出每次train和test中site类别(index 2)的区别
            if 2 < Covariates_Quantity:
                train_sites = sorted(np.unique(Covariates_train[:, 2]))
                test_sites = sorted(np.unique(Covariates_test[:, 2]))
                print(f"Fold {j} Site Distribution:")
                print(f"  Train sites: {train_sites}")
                print(f"  Test sites:  {test_sites}")
                diff = set(test_sites) - set(train_sites)
                if diff:
                    print(f"  WARNING: Sites in Test but NOT in Train: {diff}")

            terms = []
            for k in range(Covariates_Quantity):
                if k in categorical_idx:
                    # sex 和 site 用 C() 当类别变量
                    # 显式指定所有可能的levels，防止Train/Test集合中某些level缺失导致报错或不一致
                    all_levels = sorted(np.unique(Covariates[:, k]))
                    terms.append(f'C(Covariate_{k}, levels={list(all_levels)})')
                else:
                    # 其他协变量当连续变量
                    terms.append(f'Covariate_{k}')

            Formula = 'Data ~ ' + ' + '.join(terms)
            # ===========================================================
            for k in np.arange(Features_Quantity_List[conn_index]):
                df['Data'] = Subjects_Data_train_List[conn_index][:,k]
                df = pd.DataFrame(df)
                df_test['Data'] = Subjects_Data_test_List[conn_index][:,k]
                LinModel_Res = sm.ols(formula=Formula, data=df).fit()
                Subjects_Data_train_List[conn_index][:,k] = LinModel_Res.resid

                y_test_pred = LinModel_Res.predict(df_test)
                Subjects_Data_test_List[conn_index][:,k] =  Subjects_Data_test_List[conn_index][:,k]  - y_test_pred


        # MARK: waiting for modification         
        if Permutation_Flag:
            # If do permutation, the training scores should be permuted, while the testing scores remain
            Subjects_Index_Random = np.arange(len(Subjects_Score_train))
            np.random.shuffle(Subjects_Index_Random)
            Subjects_Score_train = Subjects_Score_train[Subjects_Index_Random]
            if j == 0:
                PermutationIndex = {'Fold_0': Subjects_Index_Random}
            else:
                PermutationIndex['Fold_' + str(j)] = Subjects_Index_Random
        
        for conn_index in np.arange(len(Subjects_Data_List)):
            normalize = preprocessing.MinMaxScaler()
            Subjects_Data_train_List[conn_index] = normalize.fit_transform(Subjects_Data_train_List[conn_index])
            Subjects_Data_test_List[conn_index] = normalize.transform(Subjects_Data_test_List[conn_index])

        Optimal_ComponentNumber_List, Inner_Corr_List, Inner_MAE_inv_List = PLSr1_OptimalComponentNumber_KFold(Subjects_Data_train_List, Subjects_Score_train, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity)

        Fold_J_Corr_List = []
        Fold_J_MAE_List = []
        # 根据内部交叉验证的超参数进行交叉验证
        for conn_index in np.arange(len(Subjects_Data_List)):
            clf = cross_decomposition.PLSRegression(n_components = Optimal_ComponentNumber_List[conn_index])
            clf.fit(Subjects_Data_train_List[conn_index], Subjects_Score_train)
            Fold_J_Score = clf.predict(Subjects_Data_test_List[conn_index])
            Fold_J_Score = np.transpose(Fold_J_Score)

            Fold_J_Corr = np.corrcoef(Fold_J_Score, Subjects_Score_test)
            Fold_J_Corr = Fold_J_Corr[0,1]
            Fold_J_Corr_List.append(Fold_J_Corr)
            # Fold_Corr.append(Fold_J_Corr)
            Fold_J_MAE = np.mean(np.abs(np.subtract(Fold_J_Score,Subjects_Score_test)))
            Fold_J_MAE_List.append(Fold_J_MAE)
            # Fold_MAE.append(Fold_J_MAE)

            Weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ **2))
            ################# this is the added#############################
            Weight_Haufe = np.dot(np.cov(np.transpose(Subjects_Data_train_List[conn_index])), clf.coef_);
            Weight_Haufe = Weight_Haufe / np.sqrt(np.sum(Weight_Haufe ** 2));
            ############################################################
            Fold_J_result = {'Index':Fold_J_Index, 'Test_Score':Subjects_Score_test, 'Predict_Score':Fold_J_Score, 'Corr':Fold_J_Corr, 'MAE':Fold_J_MAE, 'ComponentNumber':Optimal_ComponentNumber_List[conn_index], 'Inner_Corr':Inner_Corr_List[conn_index], 'Inner_MAE_inv':Inner_MAE_inv_List[conn_index], 'w_Brain':Weight, 'w_Brain_Haufe': Weight_Haufe}
            Fold_J_FileName = 'Fold_' + str(j) + '_Score.mat'
            ResultantFile = os.path.join(ResultantFolder, conn_cate_list[conn_index], Fold_J_FileName)
            sio.savemat(ResultantFile, Fold_J_result)
        
        Fold_Corr.append(Fold_J_Corr_List)
        Fold_MAE.append(Fold_J_MAE_List)

    Fold_Corr_array = np.array(Fold_Corr)
    Mean_Corr = Fold_Corr_array.mean(axis=0)
    Fold_J_MAE_array = np.array(Fold_MAE)
    Mean_MAE = Fold_J_MAE_array.mean(axis=0)
    for conn_index in np.arange(len(Subjects_Data_List)):
        Res_NFold = {'Mean_Corr':Mean_Corr[conn_index], 'Mean_MAE':Mean_MAE[conn_index]}
        ResultantFile = os.path.join(ResultantFolder_List[conn_index], 'Res_NFold.mat')
        sio.savemat(ResultantFile, Res_NFold)
    
    # MARK: wait for modification
    if Permutation_Flag:
        sio.savemat(ResultantFolder + '/PermutationIndex.mat', PermutationIndex)

    return (Mean_Corr, Mean_MAE)

def PLSr1_OptimalComponentNumber_KFold(Training_Data_List, Training_Score, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes, ResultantFolder, Parallel_Quantity):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)
 
    Subjects_Quantity = len(Training_Score)
    Inner_EachFold_Size = int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    Remain = np.mod(Subjects_Quantity, Fold_Quantity);    

    Inner_Corr_List = []
    Inner_MAE_inv_List = []
    for conn_index in np.arange(len(Training_Data_List)):
        Inner_Corr = np.zeros((CVRepeatTimes, Fold_Quantity, len(ComponentNumber_Range)))
        Inner_MAE_inv = np.zeros((CVRepeatTimes, Fold_Quantity, len(ComponentNumber_Range)))
        Inner_Corr_List.append(Inner_Corr)
        Inner_MAE_inv_List.append(Inner_MAE_inv)
    
    ComponentNumber_Quantity = len(ComponentNumber_Range)
    for i in np.arange(CVRepeatTimes):
        
        RandIndex = np.arange(Subjects_Quantity)
        np.random.shuffle(RandIndex)

        ComponentNumber_Quantity = len(ComponentNumber_Range)

        for k in np.arange(Fold_Quantity):

            Inner_Fold_K_Index = RandIndex[Inner_EachFold_Size * k + np.arange(Inner_EachFold_Size)]
            if Remain > k:
                Inner_Fold_K_Index = np.insert(Inner_Fold_K_Index, len(Inner_Fold_K_Index), RandIndex[Inner_EachFold_Size * Fold_Quantity + k])

            Inner_Fold_K_Data_test_List = []
            Inner_Fold_K_Score_test = Training_Score[Inner_Fold_K_Index]
            Inner_Fold_K_Data_train_List = []
            Inner_Fold_K_Score_train = np.delete(Training_Score, Inner_Fold_K_Index)

            for conn_index in np.arange(len(Training_Data_List)):
                Inner_Fold_K_Data_test = Training_Data_List[conn_index][Inner_Fold_K_Index, :]
                Inner_Fold_K_Data_test_List.append(Inner_Fold_K_Data_test)
                Inner_Fold_K_Data_train = np.delete(Training_Data_List[conn_index], Inner_Fold_K_Index, axis = 0)
                Inner_Fold_K_Data_train_List.append(Inner_Fold_K_Data_train)
            
            for conn_index in np.arange(len(Training_Data_List)):
                Scale = preprocessing.MinMaxScaler()
                Inner_Fold_K_Data_train_List[conn_index] = Scale.fit_transform(Inner_Fold_K_Data_train_List[conn_index])
                Inner_Fold_K_Data_test_List[conn_index] = Scale.transform(Inner_Fold_K_Data_test_List[conn_index])
            
            Parallel(n_jobs=Parallel_Quantity,backend="threading")(delayed(PLSr1_SubComponentNumber)(Inner_Fold_K_Data_train_List, Inner_Fold_K_Score_train, Inner_Fold_K_Data_test_List, Inner_Fold_K_Score_test, ComponentNumber_Range[l], l, ResultantFolder) for l in np.arange(len(ComponentNumber_Range)))        

            for conn_index in np.arange(len(Training_Data_List)):
                for l in np.arange(ComponentNumber_Quantity):
                    print(l)
                    ComponentNumber_l_Mat_Path = os.path.join(ResultantFolder, conn_cate_list[conn_index], 'ComponentNumber_' + str(l) + '.mat')
                    ComponentNumber_l_Mat = sio.loadmat(ComponentNumber_l_Mat_Path)
                    Inner_Corr_List[conn_index][i, k, l] = ComponentNumber_l_Mat['Corr']
                    Inner_MAE_inv_List[conn_index][i, k, l] = ComponentNumber_l_Mat['MAE_inv']
                    os.remove(ComponentNumber_l_Mat_Path)
                Inner_Corr_List[conn_index] = np.nan_to_num(Inner_Corr_List[conn_index])

    
    Inner_Corr_FoldMean_List = []
    Inner_MAE_inv_FoldMean_List = []
    Optimal_ComponentNumber_List = []
    for conn_index in np.arange(len(Training_Data_List)):
        Inner_Corr = Inner_Corr_List[conn_index] # shape: (CVTimes, k, l)
        Inner_MAE_inv = Inner_MAE_inv_List[conn_index]

        Inner_Corr_CVMean = np.mean(Inner_Corr, axis = 0)
        Inner_MAE_inv_CVMean = np.mean(Inner_MAE_inv, axis = 0)
        
        Inner_Corr_FoldMean = np.mean(Inner_Corr_CVMean, axis = 0)
        Inner_MAE_inv_FoldMean = np.mean(Inner_MAE_inv_CVMean, axis = 0)
        Inner_Corr_FoldMean = (Inner_Corr_FoldMean - np.mean(Inner_Corr_FoldMean)) / np.std(Inner_Corr_FoldMean)
        Inner_MAE_inv_FoldMean = (Inner_MAE_inv_FoldMean - np.mean(Inner_MAE_inv_FoldMean)) / np.std(Inner_MAE_inv_FoldMean)
        Inner_Evaluation = Inner_Corr_FoldMean + Inner_MAE_inv_FoldMean
        Inner_Corr_FoldMean_List.append(Inner_Corr_FoldMean)
        Inner_MAE_inv_FoldMean_List.append(Inner_MAE_inv_FoldMean)
    
        Inner_Evaluation_Mat = {'Inner_Corr_CVMean':Inner_Corr_CVMean, 'Inner_MAE_inv_CVMean':Inner_MAE_inv_CVMean, 'Inner_Corr_FoldMean': Inner_Corr_FoldMean, 'Inner_MAE_inv_FoldMean': Inner_MAE_inv_FoldMean, 'Inner_Evaluation':Inner_Evaluation}
        sio.savemat(ResultantFolder + '/' + conn_cate_list[conn_index] + '/Inner_Evaluation.mat', Inner_Evaluation_Mat)
    
        Optimal_ComponentNumber_Index = np.argmax(Inner_Evaluation) 
        Optimal_ComponentNumber = ComponentNumber_Range[Optimal_ComponentNumber_Index]
        Optimal_ComponentNumber_List.append(Optimal_ComponentNumber)
    return (Optimal_ComponentNumber_List, Inner_Corr_List, Inner_MAE_inv_List)


def PLSr1_SubComponentNumber(Training_Data_List, Training_Score, Testing_Data_List, Testing_Score, ComponentNumber, ComponentNumber_ID, ResultantFolder):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    for conn_index in np.arange(len(Training_Data_List)):
        clf = cross_decomposition.PLSRegression(n_components=ComponentNumber)
        clf.fit(Training_Data_List[conn_index], Training_Score)
        Predict_Score = clf.predict(Testing_Data_List[conn_index])
        Predict_Score = np.transpose(Predict_Score)
        Corr = np.corrcoef(Predict_Score, Testing_Score)
        Corr = Corr[0,1]
        MAE_inv = np.divide(1, np.mean(np.abs(Predict_Score - Testing_Score)))
        result = {'Corr': Corr, 'MAE_inv':MAE_inv}
        ResultantFile = os.path.join(ResultantFolder, conn_cate_list[conn_index], 'ComponentNumber_' + str(ComponentNumber_ID) + '.mat')
        sio.savemat(ResultantFile, result)
    

