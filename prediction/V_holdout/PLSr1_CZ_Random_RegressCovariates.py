# -*- coding: utf-8 -*-
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_decomposition
from joblib import Parallel, delayed
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
):
    """
    保持原有入口签名不变，但在 V_holdout 中改为单次 holdout。
    当 Fold_Quantity=10 时，对应 train/validation/test = 8:1:1。
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

    for i in np.arange(CVRepeatTimes):
        ResultantFolder_TimeI = ResultantFolder + '/Time_' + str(i)
        if not os.path.exists(ResultantFolder_TimeI):
            os.makedirs(ResultantFolder_TimeI)

        if RandIndex_File_List != '':
            RandIndex_File = RandIndex_File_List[i]
        else:
            RandIndex_File = ''

        Configuration_Mat = {
            'Subjects_Data_Mat_Path_List': Subjects_Data_Mat_Path_List,
            'Subjects_Score': Subjects_Score,
            'Covariates': Covariates,
            'Fold_Quantity': Fold_Quantity,
            'ComponentNumber_Range': ComponentNumber_Range,
            'CVRepeatTimes': CVRepeatTimes,
            'ResultantFolder_TimeI': ResultantFolder_TimeI,
            'Parallel_Quantity': Parallel_Quantity,
            'Permutation_Flag': Permutation_Flag,
            'RandIndex_File': RandIndex_File,
        }

        sio.savemat(ResultantFolder_TimeI + '/Configuration.mat', Configuration_Mat)

        system_cmd = 'python3 -c ' + '\'import sys;\
            sys.path.append("' + CodesPath + '");\
            from PLSr1_CZ_Random_RegressCovariates import PLSr1_KFold_RandomCV_MultiTimes_Sub; \
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
            PLSr1_KFold_RandomCV_MultiTimes_Sub(Subjects_Data_Mat_Path_List, Subjects_Score[0], Covariates, Fold_Quantity[0][0], ComponentNumber_Range[0], 1, ResultantFolder_TimeI[0], Parallel_Quantity[0][0], Permutation_Flag[0][0], RandIndex_File)\' '
        system_cmd = system_cmd + ' > "' + ResultantFolder_TimeI + '/Time_' + str(i) + '.log" 2>&1\n'

        script = open(ResultantFolder_TimeI + '/script.sh', 'w')
        script.write('#!/bin/bash\n')
        script.write('#SBATCH --job-name=prediction' + str(i) + '\n')
        script.write('#SBATCH --cpus-per-task=3\n')
        script.write('#SBATCH -p q_cn\n')
        script.write('#SBATCH -o ' + ResultantFolder_TimeI + '/job.%j.out\n')
        script.write('#SBATCH -e ' + ResultantFolder_TimeI + '/job.%j.error.txt\n\n')
        script.write(system_cmd)
        script.close()
        os.system('chmod +x ' + ResultantFolder_TimeI + '/script.sh')
        os.system('sbatch ' + ResultantFolder_TimeI + '/script.sh')


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
):
    Subjects_Data_List = []
    for i in range(len(Subjects_Data_Mat_Path_List)):
        data = sio.loadmat(_unwrap_mat_value(Subjects_Data_Mat_Path_List[i]))
        Subjects_Data_List.append(data['Subjects_Data'])

    Subjects_Score = np.asarray(Subjects_Score).reshape(-1)
    Covariates = np.asarray(Covariates)
    Fold_Quantity = int(np.asarray(Fold_Quantity).reshape(-1)[0])
    ComponentNumber_Range = np.asarray(ComponentNumber_Range).reshape(-1).astype(int)
    ResultantFolder = _unwrap_mat_value(ResultantFolder)
    Parallel_Quantity = int(np.asarray(Parallel_Quantity).reshape(-1)[0])
    Permutation_Flag = int(np.asarray(Permutation_Flag).reshape(-1)[0])
    RandIndex_File = _unwrap_mat_value(RandIndex_File)

    PLSr1_Holdout_RandomCV(
        Subjects_Data_List,
        Subjects_Score,
        Covariates,
        Fold_Quantity,
        ComponentNumber_Range,
        ResultantFolder,
        Parallel_Quantity,
        Permutation_Flag,
        RandIndex_File,
    )


def PLSr1_Holdout_RandomCV(
    Subjects_Data_List,
    Subjects_Score,
    Covariates,
    Fold_Quantity,
    ComponentNumber_Range,
    ResultantFolder,
    Parallel_Quantity,
    Permutation_Flag,
    RandIndex_File='',
):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    ResultantFolder_List = []
    for conn_cate in conn_cate_list:
        ResultantFolder_FC = os.path.join(ResultantFolder, conn_cate)
        if not os.path.exists(ResultantFolder_FC):
            os.makedirs(ResultantFolder_FC)
        ResultantFolder_List.append(ResultantFolder_FC)

    if Fold_Quantity < 3:
        raise ValueError('Fold_Quantity must be at least 3 for train/validation/test holdout.')

    Subjects_Quantity = len(Subjects_Score)
    if len(RandIndex_File) == 0:
        RandIndex = _generate_stratified_randindex(Subjects_Score, Fold_Quantity)
    else:
        tmpData = sio.loadmat(RandIndex_File)
        RandIndex = np.asarray(tmpData['RandIndex']).reshape(-1).astype(int)

    sio.savemat(ResultantFolder + '/RandIndex.mat', {'RandIndex': RandIndex})

    split_indices = _build_split_indices(RandIndex, Subjects_Quantity, Fold_Quantity)
    train_index = np.concatenate(split_indices[:-2]).astype(int)
    validation_index = split_indices[-2].astype(int)
    test_index = split_indices[-1].astype(int)

    sio.savemat(
        ResultantFolder + '/SplitIndex.mat',
        {
            'Train_Index': train_index,
            'Validation_Index': validation_index,
            'Test_Index': test_index,
        },
    )

    subjects_score_train = Subjects_Score[train_index]
    subjects_score_validation = Subjects_Score[validation_index]
    subjects_score_test = Subjects_Score[test_index]

    covariates_train = Covariates[train_index, :]
    covariates_validation = Covariates[validation_index, :]
    covariates_test = Covariates[test_index, :]

    subjects_data_train_raw = [data[train_index, :].copy() for data in Subjects_Data_List]
    subjects_data_validation_raw = [data[validation_index, :].copy() for data in Subjects_Data_List]
    subjects_data_test_raw = [data[test_index, :].copy() for data in Subjects_Data_List]

    subjects_data_train_list, subjects_data_validation_list = _regress_covariates_from_train(
        subjects_data_train_raw,
        subjects_data_validation_raw,
        covariates_train,
        covariates_validation,
        Covariates,
    )

    permutation_index = {}
    if Permutation_Flag:
        train_permutation = np.arange(len(subjects_score_train))
        np.random.shuffle(train_permutation)
        subjects_score_train_for_tuning = subjects_score_train[train_permutation]
        permutation_index['Train'] = train_permutation
    else:
        subjects_score_train_for_tuning = subjects_score_train

    subjects_data_train_list, subjects_data_validation_list = _scale_train_and_target(
        subjects_data_train_list,
        subjects_data_validation_list,
    )

    optimal_componentnumber_list, validation_corr_list, validation_mae_inv_list = PLSr1_OptimalComponentNumber_Validation(
        subjects_data_train_list,
        subjects_score_train_for_tuning,
        subjects_data_validation_list,
        subjects_score_validation,
        ComponentNumber_Range,
        ResultantFolder,
        Parallel_Quantity,
    )

    development_index = np.concatenate((train_index, validation_index)).astype(int)
    subjects_score_development = Subjects_Score[development_index]
    covariates_development = Covariates[development_index, :]
    subjects_data_development_raw = [data[development_index, :].copy() for data in Subjects_Data_List]

    if Permutation_Flag:
        development_permutation = np.arange(len(subjects_score_development))
        np.random.shuffle(development_permutation)
        subjects_score_development = subjects_score_development[development_permutation]
        permutation_index['Development'] = development_permutation

    subjects_data_development_list, subjects_data_test_list = _regress_covariates_from_train(
        subjects_data_development_raw,
        subjects_data_test_raw,
        covariates_development,
        covariates_test,
        Covariates,
    )
    subjects_data_development_list, subjects_data_test_list = _scale_train_and_target(
        subjects_data_development_list,
        subjects_data_test_list,
    )

    holdout_corr_list = []
    holdout_mae_list = []
    for conn_index in np.arange(len(Subjects_Data_List)):
        clf = cross_decomposition.PLSRegression(n_components=optimal_componentnumber_list[conn_index])
        clf.fit(subjects_data_development_list[conn_index], subjects_score_development)
        holdout_score = clf.predict(subjects_data_test_list[conn_index]).reshape(-1)

        holdout_corr = np.corrcoef(holdout_score, subjects_score_test)[0, 1]
        holdout_mae = np.mean(np.abs(holdout_score - subjects_score_test))
        holdout_corr_list.append(holdout_corr)
        holdout_mae_list.append(holdout_mae)

        weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ ** 2))
        coef_vector = clf.coef_.flatten() if clf.coef_.ndim > 1 else clf.coef_
        weight_haufe = np.dot(np.cov(np.transpose(subjects_data_development_list[conn_index])), coef_vector)
        weight_haufe = weight_haufe / np.sqrt(np.sum(weight_haufe ** 2))

        holdout_result = {
            'Train_Index': train_index,
            'Validation_Index': validation_index,
            'Test_Index': test_index,
            'Validation_Score': subjects_score_validation,
            'Test_Score': subjects_score_test,
            'Predict_Score': holdout_score,
            'Corr': holdout_corr,
            'MAE': holdout_mae,
            'ComponentNumber': optimal_componentnumber_list[conn_index],
            'Validation_Corr': validation_corr_list[conn_index],
            'Validation_MAE_inv': validation_mae_inv_list[conn_index],
            'w_Brain': weight,
            'w_Brain_Haufe': weight_haufe,
        }
        sio.savemat(
            os.path.join(ResultantFolder, conn_cate_list[conn_index], 'Holdout_Score.mat'),
            holdout_result,
        )

        sio.savemat(
            os.path.join(ResultantFolder_List[conn_index], 'Res_NFold.mat'),
            {'Mean_Corr': holdout_corr, 'Mean_MAE': holdout_mae},
        )

    if Permutation_Flag:
        sio.savemat(ResultantFolder + '/PermutationIndex.mat', permutation_index)

    return (np.array(holdout_corr_list), np.array(holdout_mae_list))


def PLSr1_OptimalComponentNumber_Validation(
    Training_Data_List,
    Training_Score,
    Validation_Data_List,
    Validation_Score,
    ComponentNumber_Range,
    ResultantFolder,
    Parallel_Quantity,
):
    conn_cate_list = ['GGFC', 'GWFC', 'WWFC']
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)

    Parallel(n_jobs=Parallel_Quantity, backend="threading")(
        delayed(PLSr1_SubComponentNumber)(
            Training_Data_List,
            Training_Score,
            Validation_Data_List,
            Validation_Score,
            ComponentNumber_Range[l],
            l,
            ResultantFolder,
        )
        for l in np.arange(len(ComponentNumber_Range))
    )

    validation_corr_list = []
    validation_mae_inv_list = []
    optimal_componentnumber_list = []
    for conn_index in np.arange(len(Training_Data_List)):
        validation_corr = np.zeros(len(ComponentNumber_Range))
        validation_mae_inv = np.zeros(len(ComponentNumber_Range))
        for l in np.arange(len(ComponentNumber_Range)):
            component_path = os.path.join(
                ResultantFolder,
                conn_cate_list[conn_index],
                'ComponentNumber_' + str(l) + '.mat',
            )
            component_mat = sio.loadmat(component_path)
            validation_corr[l] = np.asarray(component_mat['Corr']).reshape(-1)[0]
            validation_mae_inv[l] = np.asarray(component_mat['MAE_inv']).reshape(-1)[0]
            os.remove(component_path)

        validation_corr = np.nan_to_num(validation_corr)
        validation_mae_inv = np.nan_to_num(validation_mae_inv)
        validation_corr_z = _zscore_or_zero(validation_corr)
        validation_mae_inv_z = _zscore_or_zero(validation_mae_inv)
        validation_evaluation = validation_corr_z + validation_mae_inv_z

        sio.savemat(
            ResultantFolder + '/' + conn_cate_list[conn_index] + '/Validation_Evaluation.mat',
            {
                'Validation_Corr': validation_corr,
                'Validation_MAE_inv': validation_mae_inv,
                'Validation_Corr_Z': validation_corr_z,
                'Validation_MAE_inv_Z': validation_mae_inv_z,
                'Validation_Evaluation': validation_evaluation,
            },
        )

        optimal_componentnumber_index = np.argmax(validation_evaluation)
        optimal_componentnumber = ComponentNumber_Range[optimal_componentnumber_index]

        validation_corr_list.append(validation_corr)
        validation_mae_inv_list.append(validation_mae_inv)
        optimal_componentnumber_list.append(optimal_componentnumber)

    return (optimal_componentnumber_list, validation_corr_list, validation_mae_inv_list)


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
        Predict_Score = clf.predict(Testing_Data_List[conn_index]).reshape(-1)
        Corr = np.corrcoef(Predict_Score, Testing_Score)[0, 1]
        MAE_inv = np.divide(1, np.mean(np.abs(Predict_Score - Testing_Score)))
        sio.savemat(
            os.path.join(
                ResultantFolder,
                conn_cate_list[conn_index],
                'ComponentNumber_' + str(ComponentNumber_ID) + '.mat',
            ),
            {'Corr': Corr, 'MAE_inv': MAE_inv},
        )


def _unwrap_mat_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ''
        return str(value.reshape(-1)[0])
    return str(value)


def _generate_stratified_randindex(subjects_score, split_quantity):
    sorted_indices_desc = np.argsort(subjects_score)[::-1]
    sample_quantity = len(sorted_indices_desc)
    remain = sample_quantity % split_quantity
    num_bins = (sample_quantity + split_quantity - 1) // split_quantity

    bins = [
        sorted_indices_desc[i * split_quantity: (i + 1) * split_quantity]
        for i in range(num_bins)
    ]
    shuffled_bins = [np.random.permutation(bin_i) for bin_i in bins]
    randindex = np.zeros(sample_quantity)
    if remain == 0:
        for j in range(num_bins):
            for i in range(split_quantity):
                randindex[i * num_bins + j] = shuffled_bins[j][i]
    else:
        for j in range(num_bins - 1):
            for i in range(split_quantity):
                if i * (num_bins - 1) + j >= sample_quantity:
                    continue
                randindex[i * (num_bins - 1) + j] = shuffled_bins[j][i]
        for k in range(len(shuffled_bins[-1])):
            randindex[(num_bins - 1) * split_quantity + k] = shuffled_bins[-1][k]

    return randindex.astype(int)


def save_stratified_randindex(subjects_score, split_quantity, output_file):
    randindex = _generate_stratified_randindex(subjects_score, split_quantity)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sio.savemat(output_file, {'RandIndex': randindex})
    return randindex


def _build_split_indices(randindex, subjects_quantity, split_quantity):
    each_split_size = int(np.fix(np.divide(subjects_quantity, split_quantity)))
    remain = np.mod(subjects_quantity, split_quantity)
    split_indices = []
    for split_index in np.arange(split_quantity):
        current_index = randindex[each_split_size * split_index + np.arange(each_split_size)]
        if remain > split_index:
            current_index = np.insert(
                current_index,
                len(current_index),
                randindex[each_split_size * split_quantity + split_index],
            )
        split_indices.append(current_index.astype(int))
    return split_indices


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
        {
            'Covariate_' + str(k): covariates[:, k]
            for k in np.arange(covariates.shape[1])
        }
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
            target_data[:, feature_index] = target_data[:, feature_index] - linmodel_res.predict(target_df)

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
