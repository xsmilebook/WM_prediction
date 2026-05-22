# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from sklearn import cross_decomposition
from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold
import statsmodels.formula.api as sm

CodesPath = os.path.dirname(os.path.abspath(__file__))


def PLSr1_KFold_RandomCV_MultiTimes(
    Subjects_Data_List,
    Subjects_Score,
    Covariates,
    Fold_Quantity,
    ComponentNumber_Range,
    CVRepeatTimes,
    ResultantFolder,
    Parallel_Quantity,
    Permutation_Flag,
    RandIndex_File_List='',
    RandomSeed_List='',
):
    """
    保持原有入口签名不变，但在 V_holdout 中改为：
    1. family-aware 的固定 half-split 外层 train/test；
    2. outer train 内部的 5-fold CV 选择最佳 component；
    3. permutation 仅打乱 outer train 标签，outer test 标签保持真实。
    """
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)

    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    subjects_data_mat_path_list = []
    for i in np.arange(len(Subjects_Data_List)):
        subjects_data_mat = {'Subjects_Data': Subjects_Data_List[i]}
        subjects_data_mat_path = os.path.join(ResultantFolder, conn_cate_list[i] + '_Subjects_Data.mat')
        subjects_data_mat_path_list.append(subjects_data_mat_path)
        sio.savemat(subjects_data_mat_path, subjects_data_mat)

    for i in np.arange(CVRepeatTimes):
        resultantfolder_timei = os.path.join(ResultantFolder, 'Time_' + str(i))
        if not os.path.exists(resultantfolder_timei):
            os.makedirs(resultantfolder_timei)

        if RandIndex_File_List != '':
            randindex_file = RandIndex_File_List[i]
        else:
            randindex_file = ''

        if RandomSeed_List != '':
            random_seed = int(RandomSeed_List[i])
        else:
            random_seed = -1

        configuration_mat = {
            'Subjects_Data_Mat_Path_List': subjects_data_mat_path_list,
            'Subjects_Score': Subjects_Score,
            'Covariates': Covariates,
            'Fold_Quantity': Fold_Quantity,
            'ComponentNumber_Range': ComponentNumber_Range,
            'CVRepeatTimes': CVRepeatTimes,
            'ResultantFolder_TimeI': resultantfolder_timei,
            'Parallel_Quantity': Parallel_Quantity,
            'Permutation_Flag': Permutation_Flag,
            'RandIndex_File': randindex_file,
            'Random_Seed': random_seed,
        }
        sio.savemat(os.path.join(resultantfolder_timei, 'Configuration.mat'), configuration_mat)

        system_cmd = 'python3 -c ' + '\'import sys;\
            sys.path.append("' + CodesPath + '");\
            from PLSr1_CZ_Random_RegressCovariates import PLSr1_KFold_RandomCV_MultiTimes_Sub; \
            import scipy.io as sio;\
            Configuration = sio.loadmat("' + resultantfolder_timei + '/Configuration.mat");\
            Subjects_Data_Mat_Path_List = Configuration["Subjects_Data_Mat_Path_List"];\
            Subjects_Score = Configuration["Subjects_Score"];\
            Covariates = Configuration["Covariates"];\
            Fold_Quantity = Configuration["Fold_Quantity"];\
            ComponentNumber_Range = Configuration["ComponentNumber_Range"];\
            ResultantFolder_TimeI = Configuration["ResultantFolder_TimeI"];\
            Permutation_Flag = Configuration["Permutation_Flag"];\
            RandIndex_File = Configuration["RandIndex_File"];\
            Parallel_Quantity = Configuration["Parallel_Quantity"];\
            Random_Seed = Configuration["Random_Seed"];\
            PLSr1_KFold_RandomCV_MultiTimes_Sub(Subjects_Data_Mat_Path_List, Subjects_Score[0], Covariates, Fold_Quantity[0][0], ComponentNumber_Range[0], 20, ResultantFolder_TimeI[0], Parallel_Quantity[0][0], Permutation_Flag[0][0], RandIndex_File, Random_Seed[0][0])\' '
        system_cmd = system_cmd + ' > "' + os.path.join(resultantfolder_timei, 'Time_' + str(i) + '.log') + '" 2>&1\n'

        script_path = os.path.join(resultantfolder_timei, 'script.sh')
        script = open(script_path, 'w')
        script.write('#!/bin/bash\n')
        script.write('#SBATCH --job-name=prediction' + str(i) + '\n')
        script.write('#SBATCH --cpus-per-task=1\n')
        script.write('#SBATCH -p q_fat_c,q_fat_l,q_fat\n')
        script.write('#SBATCH -o ' + os.path.join(resultantfolder_timei, 'job.%j.out') + '\n')
        script.write('#SBATCH -e ' + os.path.join(resultantfolder_timei, 'job.%j.error.txt') + '\n\n')
        script.write(system_cmd)
        script.close()
        os.system('chmod +x ' + script_path)
        os.system('sbatch ' + script_path)


def PLSr1_KFold_RandomCV_MultiTimes_Sub(
    Subjects_Data_Mat_Path_List,
    Subjects_Score,
    Covariates,
    Fold_Quantity,
    ComponentNumber_Range,
    CVRepeatTimes,
    ResultantFolder,
    Parallel_Quantity,
    Permutation_Flag,
    RandIndex_File='',
    Random_Seed=-1,
):
    subjects_data_list = []
    for i in range(len(Subjects_Data_Mat_Path_List)):
        data = sio.loadmat(_unwrap_mat_value(Subjects_Data_Mat_Path_List[i]))
        subjects_data_list.append(data['Subjects_Data'])

    subjects_score = np.asarray(Subjects_Score).reshape(-1)
    covariates = np.asarray(Covariates)
    fold_quantity = int(np.asarray(Fold_Quantity).reshape(-1)[0])
    componentnumber_range = np.asarray(ComponentNumber_Range).reshape(-1).astype(int)
    cvrepeattimes = int(np.asarray(CVRepeatTimes).reshape(-1)[0])
    resultantfolder = _unwrap_mat_value(ResultantFolder)
    parallel_quantity = int(np.asarray(Parallel_Quantity).reshape(-1)[0])
    permutation_flag = int(np.asarray(Permutation_Flag).reshape(-1)[0])
    splitindex_file = _unwrap_mat_value(RandIndex_File)
    random_seed = int(np.asarray(Random_Seed).reshape(-1)[0])

    PLSr1_Holdout_RandomCV(
        subjects_data_list,
        subjects_score,
        covariates,
        fold_quantity,
        componentnumber_range,
        cvrepeattimes,
        resultantfolder,
        parallel_quantity,
        permutation_flag,
        splitindex_file,
        random_seed,
    )


def PLSr1_Holdout_RandomCV(
    Subjects_Data_List,
    Subjects_Score,
    Covariates,
    Fold_Quantity,
    ComponentNumber_Range,
    CVRepeatTimes_ForInner,
    ResultantFolder,
    Parallel_Quantity,
    Permutation_Flag,
    SplitIndex_File='',
    Random_Seed=-1,
):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    resultantfolder_list = []
    for conn_cate in conn_cate_list:
        resultantfolder_fc = os.path.join(ResultantFolder, conn_cate)
        if not os.path.exists(resultantfolder_fc):
            os.makedirs(resultantfolder_fc)
        resultantfolder_list.append(resultantfolder_fc)

    train_index, test_index = _load_half_split(SplitIndex_File, len(Subjects_Score))
    sio.savemat(
        os.path.join(ResultantFolder, 'SplitIndex.mat'),
        {
            'Train_Index': train_index,
            'Test_Index': test_index,
        },
    )

    subjects_score_train = Subjects_Score[train_index].copy()
    subjects_score_test = Subjects_Score[test_index].copy()
    covariates_train = Covariates[train_index, :]
    covariates_test = Covariates[test_index, :]

    subjects_data_train_raw = [data[train_index, :].copy() for data in Subjects_Data_List]
    subjects_data_test_raw = [data[test_index, :].copy() for data in Subjects_Data_List]
    rng_seed = None if Random_Seed is None or int(Random_Seed) < 0 else int(Random_Seed)
    rng = np.random.default_rng(rng_seed)

    permutation_index = {}
    if Permutation_Flag:
        train_permutation = rng.permutation(len(subjects_score_train))
        subjects_score_train = subjects_score_train[train_permutation]
        permutation_index['Train'] = train_permutation

    subjects_data_train_list, subjects_data_test_list = _regress_covariates_from_train(
        subjects_data_train_raw,
        subjects_data_test_raw,
        covariates_train,
        covariates_test,
        Covariates,
    )
    subjects_data_train_list, subjects_data_test_list = _scale_train_and_target(
        subjects_data_train_list,
        subjects_data_test_list,
    )

    optimal_componentnumber_list, inner_corr_list, inner_mae_inv_list = PLSr1_OptimalComponentNumber_KFold(
        subjects_data_train_list,
        subjects_score_train,
        Fold_Quantity,
        ComponentNumber_Range,
        CVRepeatTimes_ForInner,
        ResultantFolder,
        Parallel_Quantity,
        None if rng_seed is None else rng_seed + 1,
    )

    holdout_corr_list = []
    holdout_mae_list = []
    for conn_index in np.arange(len(Subjects_Data_List)):
        clf = cross_decomposition.PLSRegression(
            n_components=optimal_componentnumber_list[conn_index]
        )
        clf.fit(subjects_data_train_list[conn_index], subjects_score_train)
        holdout_score = clf.predict(subjects_data_test_list[conn_index]).reshape(-1)

        holdout_corr = np.corrcoef(holdout_score, subjects_score_test)[0, 1]
        holdout_mae = np.mean(np.abs(holdout_score - subjects_score_test))
        holdout_corr_list.append(holdout_corr)
        holdout_mae_list.append(holdout_mae)

        weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ ** 2))
        coef_vector = clf.coef_.flatten() if clf.coef_.ndim > 1 else clf.coef_
        weight_haufe = np.dot(np.cov(np.transpose(subjects_data_train_list[conn_index])), coef_vector)
        weight_haufe = weight_haufe / np.sqrt(np.sum(weight_haufe ** 2))

        holdout_result = {
            'Train_Index': train_index,
            'Test_Index': test_index,
            'Test_Score': subjects_score_test,
            'Predict_Score': holdout_score,
            'Corr': holdout_corr,
            'MAE': holdout_mae,
            'ComponentNumber': optimal_componentnumber_list[conn_index],
            'Inner_Corr': inner_corr_list[conn_index],
            'Inner_MAE_inv': inner_mae_inv_list[conn_index],
            'w_Brain': weight,
            'w_Brain_Haufe': weight_haufe,
        }
        sio.savemat(
            os.path.join(resultantfolder_list[conn_index], 'Holdout_Score.mat'),
            holdout_result,
        )
        sio.savemat(
            os.path.join(resultantfolder_list[conn_index], 'Res_NFold.mat'),
            {'Mean_Corr': holdout_corr, 'Mean_MAE': holdout_mae},
        )

    if Permutation_Flag:
        sio.savemat(os.path.join(ResultantFolder, 'PermutationIndex.mat'), permutation_index)

    return np.array(holdout_corr_list), np.array(holdout_mae_list)


def PLSr1_OptimalComponentNumber_KFold(
    Training_Data_List,
    Training_Score,
    Fold_Quantity,
    ComponentNumber_Range,
    CVRepeatTimes,
    ResultantFolder,
    Parallel_Quantity,
    Random_Seed=None,
):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)

    subjects_quantity = len(Training_Score)
    inner_eachfold_size = int(np.fix(np.divide(subjects_quantity, Fold_Quantity)))
    remain = np.mod(subjects_quantity, Fold_Quantity)

    inner_corr_list = []
    inner_mae_inv_list = []
    for _ in np.arange(len(Training_Data_List)):
        inner_corr = np.zeros((CVRepeatTimes, Fold_Quantity, len(ComponentNumber_Range)))
        inner_mae_inv = np.zeros((CVRepeatTimes, Fold_Quantity, len(ComponentNumber_Range)))
        inner_corr_list.append(inner_corr)
        inner_mae_inv_list.append(inner_mae_inv)
    rng = np.random.default_rng(Random_Seed)

    for i in np.arange(CVRepeatTimes):
        randindex = rng.permutation(subjects_quantity)

        for k in np.arange(Fold_Quantity):
            inner_fold_k_index = randindex[inner_eachfold_size * k + np.arange(inner_eachfold_size)]
            if remain > k:
                inner_fold_k_index = np.insert(
                    inner_fold_k_index,
                    len(inner_fold_k_index),
                    randindex[inner_eachfold_size * Fold_Quantity + k],
                )

            inner_fold_k_data_test_list = []
            inner_fold_k_score_test = Training_Score[inner_fold_k_index]
            inner_fold_k_data_train_list = []
            inner_fold_k_score_train = np.delete(Training_Score, inner_fold_k_index)

            for conn_index in np.arange(len(Training_Data_List)):
                inner_fold_k_data_test = Training_Data_List[conn_index][inner_fold_k_index, :]
                inner_fold_k_data_test_list.append(inner_fold_k_data_test)
                inner_fold_k_data_train = np.delete(
                    Training_Data_List[conn_index],
                    inner_fold_k_index,
                    axis=0,
                )
                inner_fold_k_data_train_list.append(inner_fold_k_data_train)

            for conn_index in np.arange(len(Training_Data_List)):
                scale = preprocessing.MinMaxScaler()
                inner_fold_k_data_train_list[conn_index] = scale.fit_transform(
                    inner_fold_k_data_train_list[conn_index]
                )
                inner_fold_k_data_test_list[conn_index] = scale.transform(
                    inner_fold_k_data_test_list[conn_index]
                )

            Parallel(n_jobs=Parallel_Quantity, backend='threading')(
                delayed(PLSr1_SubComponentNumber)(
                    inner_fold_k_data_train_list,
                    inner_fold_k_score_train,
                    inner_fold_k_data_test_list,
                    inner_fold_k_score_test,
                    ComponentNumber_Range[l],
                    l,
                    ResultantFolder,
                )
                for l in np.arange(len(ComponentNumber_Range))
            )

            for conn_index in np.arange(len(Training_Data_List)):
                for l in np.arange(len(ComponentNumber_Range)):
                    componentnumber_l_mat_path = os.path.join(
                        ResultantFolder,
                        conn_cate_list[conn_index],
                        'ComponentNumber_' + str(l) + '.mat',
                    )
                    componentnumber_l_mat = sio.loadmat(componentnumber_l_mat_path)
                    inner_corr_list[conn_index][i, k, l] = np.asarray(
                        componentnumber_l_mat['Corr']
                    ).reshape(-1)[0]
                    inner_mae_inv_list[conn_index][i, k, l] = np.asarray(
                        componentnumber_l_mat['MAE_inv']
                    ).reshape(-1)[0]
                    os.remove(componentnumber_l_mat_path)
                inner_corr_list[conn_index] = np.nan_to_num(inner_corr_list[conn_index])
                inner_mae_inv_list[conn_index] = np.nan_to_num(inner_mae_inv_list[conn_index])

    inner_corr_foldmean_list = []
    inner_mae_inv_foldmean_list = []
    optimal_componentnumber_list = []
    for conn_index in np.arange(len(Training_Data_List)):
        inner_corr = inner_corr_list[conn_index]
        inner_mae_inv = inner_mae_inv_list[conn_index]

        inner_corr_cvmean = np.mean(inner_corr, axis=0)
        inner_mae_inv_cvmean = np.mean(inner_mae_inv, axis=0)
        inner_corr_foldmean = np.mean(inner_corr_cvmean, axis=0)
        inner_mae_inv_foldmean = np.mean(inner_mae_inv_cvmean, axis=0)
        inner_corr_foldmean = _zscore_or_zero(inner_corr_foldmean)
        inner_mae_inv_foldmean = _zscore_or_zero(inner_mae_inv_foldmean)
        inner_evaluation = inner_corr_foldmean + inner_mae_inv_foldmean

        inner_corr_foldmean_list.append(inner_corr_foldmean)
        inner_mae_inv_foldmean_list.append(inner_mae_inv_foldmean)

        inner_evaluation_mat = {
            'Inner_Corr_CVMean': inner_corr_cvmean,
            'Inner_MAE_inv_CVMean': inner_mae_inv_cvmean,
            'Inner_Corr_FoldMean': inner_corr_foldmean,
            'Inner_MAE_inv_FoldMean': inner_mae_inv_foldmean,
            'Inner_Evaluation': inner_evaluation,
        }
        sio.savemat(
            os.path.join(ResultantFolder, conn_cate_list[conn_index], 'Inner_Evaluation.mat'),
            inner_evaluation_mat,
        )

        optimal_componentnumber_index = np.argmax(inner_evaluation)
        optimal_componentnumber = ComponentNumber_Range[optimal_componentnumber_index]
        optimal_componentnumber_list.append(optimal_componentnumber)

    return optimal_componentnumber_list, inner_corr_list, inner_mae_inv_list


def PLSr1_SubComponentNumber(
    Training_Data_List,
    Training_Score,
    Testing_Data_List,
    Testing_Score,
    ComponentNumber,
    ComponentNumber_ID,
    ResultantFolder,
):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    for conn_index in np.arange(len(Training_Data_List)):
        clf = cross_decomposition.PLSRegression(n_components=ComponentNumber)
        clf.fit(Training_Data_List[conn_index], Training_Score)
        predict_score = clf.predict(Testing_Data_List[conn_index]).reshape(-1)
        corr = np.corrcoef(predict_score, Testing_Score)[0, 1]
        mae_inv = np.divide(1, np.mean(np.abs(predict_score - Testing_Score)))
        sio.savemat(
            os.path.join(
                ResultantFolder,
                conn_cate_list[conn_index],
                'ComponentNumber_' + str(ComponentNumber_ID) + '.mat',
            ),
            {'Corr': corr, 'MAE_inv': mae_inv},
        )


def _unwrap_mat_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ''
        return str(value.reshape(-1)[0])
    return str(value)


def _load_half_split(splitindex_file, subjects_quantity):
    if not splitindex_file:
        raise ValueError('SplitIndex_File is required for V_holdout half-split evaluation.')

    split_mat = sio.loadmat(splitindex_file)
    train_index = np.asarray(split_mat['Train_Index']).reshape(-1).astype(int)
    test_index = np.asarray(split_mat['Test_Index']).reshape(-1).astype(int)

    if np.intersect1d(train_index, test_index).size > 0:
        raise ValueError('Train_Index and Test_Index overlap in split file.')
    if len(train_index) + len(test_index) != subjects_quantity:
        raise ValueError('Train_Index and Test_Index do not cover all subjects.')

    return train_index, test_index


def _build_group_stratification_labels(subjects_score, n_splits):
    subjects_score = np.asarray(subjects_score).reshape(-1)
    max_bins = min(10, len(np.unique(subjects_score)))
    for bin_count in range(max_bins, 1, -1):
        labels = pd.qcut(subjects_score, q=bin_count, labels=False, duplicates='drop')
        labels = np.asarray(labels, dtype=int)
        bincount = np.bincount(labels)
        if bincount.size >= 2 and np.min(bincount) >= n_splits:
            return labels
    raise ValueError('Unable to create stratification labels for grouped half-split.')


def save_grouped_half_split(subjects_score, family_ids, output_file, random_state=1234):
    subjects_score = np.asarray(subjects_score).reshape(-1)
    family_ids = np.asarray(family_ids).reshape(-1)
    if len(subjects_score) != len(family_ids):
        raise ValueError('subjects_score and family_ids must have the same length.')

    stratification_labels = _build_group_stratification_labels(subjects_score, n_splits=2)
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    train_index, test_index = next(
        splitter.split(
            np.zeros((len(subjects_score), 1)),
            stratification_labels,
            groups=family_ids,
        )
    )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sio.savemat(
        output_file,
        {
            'Train_Index': train_index.astype(int),
            'Test_Index': test_index.astype(int),
            'Family_ID': family_ids.astype(int),
        },
    )
    return train_index.astype(int), test_index.astype(int)


def _build_covariate_formula(covariates):
    covariates_quantity = covariates.shape[1]
    formula = 'Data'
    for k in np.arange(covariates_quantity):
        if k == 0 or k == 2:
            all_levels = sorted(set(covariates[:, k]))
            term = f'C(Covariate_{k}, levels={all_levels})'
        else:
            term = f'Covariate_{k}'

        if k == 0:
            formula += f' ~ {term}'
        else:
            formula += f' + {term}'
    return formula


def _build_covariate_dataframe(covariates):
    return pd.DataFrame(
        {'Covariate_' + str(k): covariates[:, k] for k in np.arange(covariates.shape[1])}
    )


def _regress_covariates_from_train(
    train_data_list,
    target_data_list,
    covariates_train,
    covariates_target,
    covariates_all,
):
    train_regressed_list = []
    target_regressed_list = []
    formula = _build_covariate_formula(covariates_all)
    base_train_df = _build_covariate_dataframe(covariates_train)
    base_target_df = _build_covariate_dataframe(covariates_target)

    for conn_index in np.arange(len(train_data_list)):
        train_data = train_data_list[conn_index].copy()
        target_data = target_data_list[conn_index].copy()
        features_quantity = train_data.shape[1]

        for feature_index in np.arange(features_quantity):
            train_df = base_train_df.copy()
            train_df['Data'] = train_data[:, feature_index]
            target_df = base_target_df.copy()
            target_df['Data'] = target_data[:, feature_index]

            linmodel_res = sm.ols(formula=formula, data=train_df).fit()
            train_data[:, feature_index] = linmodel_res.resid
            target_data[:, feature_index] = (
                target_data[:, feature_index] - linmodel_res.predict(target_df)
            )

        train_regressed_list.append(train_data)
        target_regressed_list.append(target_data)

    return train_regressed_list, target_regressed_list


def _scale_train_and_target(train_data_list, target_data_list):
    scaled_train_list = []
    scaled_target_list = []
    for conn_index in np.arange(len(train_data_list)):
        scale = preprocessing.MinMaxScaler()
        scaled_train_list.append(scale.fit_transform(train_data_list[conn_index]))
        scaled_target_list.append(scale.transform(target_data_list[conn_index]))
    return scaled_train_list, scaled_target_list


def _zscore_or_zero(values):
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std
