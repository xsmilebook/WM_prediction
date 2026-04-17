# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from sklearn import cross_decomposition
from sklearn import preprocessing
import statsmodels.formula.api as sm

CODE_PATH = os.path.dirname(os.path.abspath(__file__))
INNER_CV_REPEAT_TIMES = 20


def _save_configuration(
    config_path,
    subjects_data_mat_path_list,
    subjects_score,
    covariates,
    fold_quantity,
    component_number_range,
    resultant_folder_time_i,
    parallel_quantity,
    permutation_flag,
    feature_name_list,
    randindex_file,
):
    np.savez(
        config_path,
        Subjects_Data_Mat_Path_List=np.array(subjects_data_mat_path_list, dtype=object),
        Subjects_Score=np.asarray(subjects_score),
        Covariates=np.asarray(covariates),
        Fold_Quantity=np.array([fold_quantity], dtype=int),
        ComponentNumber_Range=np.asarray(component_number_range),
        ResultantFolder_TimeI=np.array([resultant_folder_time_i], dtype=object),
        Parallel_Quantity=np.array([parallel_quantity], dtype=int),
        Permutation_Flag=np.array([permutation_flag], dtype=int),
        Feature_Name_List=np.array(feature_name_list, dtype=object),
        RandIndex_File=np.array([randindex_file], dtype=object),
    )


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
    Feature_Name_List,
    RandIndex_File_List='',
):
    """
    PLSr1_KFold_RandomCV_MultiTimes: use PLS regression with K-fold
    cross-validation and optional permutation for multiple feature sets.
    """
    if len(Subjects_Data_List) != len(Feature_Name_List):
        raise ValueError('Subjects_Data_List and Feature_Name_List must have the same length.')

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)

    Subjects_Data_Mat_Path_List = []
    for feature_name, subjects_data in zip(Feature_Name_List, Subjects_Data_List):
        subjects_data_mat_path = os.path.join(ResultantFolder, f'{feature_name}_Subjects_Data.mat')
        sio.savemat(subjects_data_mat_path, {'Subjects_Data': subjects_data})
        Subjects_Data_Mat_Path_List.append(subjects_data_mat_path)

    for i in np.arange(CVRepeatTimes):
        resultant_folder_time_i = os.path.join(ResultantFolder, f'Time_{i}')
        if not os.path.exists(resultant_folder_time_i):
            os.makedirs(resultant_folder_time_i)

        randindex_file = RandIndex_File_List[i] if RandIndex_File_List else ''
        config_path = os.path.join(resultant_folder_time_i, 'configuration.npz')
        _save_configuration(
            config_path,
            Subjects_Data_Mat_Path_List,
            Subjects_Score,
            Covariates,
            Fold_Quantity,
            ComponentNumber_Range,
            resultant_folder_time_i,
            Parallel_Quantity,
            Permutation_Flag,
            Feature_Name_List,
            randindex_file,
        )

        system_cmd = (
            "python3 -c "
            "'import sys; "
            f"sys.path.insert(0, {CODE_PATH!r}); "
            "import numpy as np; "
            "from PLSr1_CZ_Random_RegressCovariates import PLSr1_KFold_RandomCV_MultiTimes_Sub; "
            f"config = np.load({config_path!r}, allow_pickle=True); "
            "PLSr1_KFold_RandomCV_MultiTimes_Sub("
            "config[\"Subjects_Data_Mat_Path_List\"].tolist(), "
            "config[\"Subjects_Score\"], "
            "config[\"Covariates\"], "
            "int(config[\"Fold_Quantity\"][0]), "
            "config[\"ComponentNumber_Range\"], "
            f"{INNER_CV_REPEAT_TIMES}, "
            "config[\"ResultantFolder_TimeI\"].tolist()[0], "
            "int(config[\"Parallel_Quantity\"][0]), "
            "int(config[\"Permutation_Flag\"][0]), "
            "config[\"Feature_Name_List\"].tolist(), "
            "config[\"RandIndex_File\"].tolist()[0])'"
        )
        system_cmd = f'{system_cmd} > "{os.path.join(resultant_folder_time_i, f"Time_{i}.log")}" 2>&1\n'

        script_path = os.path.join(resultant_folder_time_i, 'script.sh')
        with open(script_path, 'w') as script:
            script.write('#!/bin/bash\n')
            script.write(f'#SBATCH --job-name=prediction{i}\n')
            script.write('#SBATCH --nodes=1\n')
            script.write('#SBATCH --ntasks=1\n')
            script.write('#SBATCH --cpus-per-task=1\n')
            script.write('#SBATCH -p q_fat_c\n')
            script.write(f'#SBATCH -o {os.path.join(resultant_folder_time_i, "job.%j.out")}\n')
            script.write(f'#SBATCH -e {os.path.join(resultant_folder_time_i, "job.%j.error.txt")}\n\n')
            script.write(system_cmd)

        os.system(f'chmod +x {script_path}')
        os.system(f'sbatch {script_path}')


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
    Feature_Name_List,
    RandIndex_File='',
):
    subjects_data_list = []
    for data_path in Subjects_Data_Mat_Path_List:
        data = sio.loadmat(data_path)
        subjects_data_list.append(data['Subjects_Data'])

    PLSr1_KFold_RandomCV(
        subjects_data_list,
        Subjects_Score,
        Covariates,
        Fold_Quantity,
        ComponentNumber_Range,
        CVRepeatTimes,
        ResultantFolder,
        Parallel_Quantity,
        Permutation_Flag,
        Feature_Name_List,
        RandIndex_File,
    )


def PLSr1_KFold_RandomCV(
    Subjects_Data_List,
    Subjects_Score,
    Covariates,
    Fold_Quantity,
    ComponentNumber_Range,
    CVRepeatTimes_ForInner,
    ResultantFolder,
    Parallel_Quantity,
    Permutation_Flag,
    Feature_Name_List,
    RandIndex_File='',
):
    resultant_folder_list = []
    for feature_name in Feature_Name_List:
        resultant_folder_fc = os.path.join(ResultantFolder, feature_name)
        if not os.path.exists(resultant_folder_fc):
            os.makedirs(resultant_folder_fc)
        resultant_folder_list.append(resultant_folder_fc)

    subjects_quantity = len(Subjects_Score)
    each_fold_size = int(np.fix(np.divide(subjects_quantity, Fold_Quantity)))
    remain = np.mod(subjects_quantity, Fold_Quantity)
    if not RandIndex_File:
        sorted_indices_desc = np.argsort(Subjects_Score)[::-1]

        subject_count = len(sorted_indices_desc)
        remainder = subject_count % Fold_Quantity
        num_bins = (subject_count + Fold_Quantity - 1) // Fold_Quantity
        bins = [
            sorted_indices_desc[i * Fold_Quantity : (i + 1) * Fold_Quantity]
            for i in range(num_bins)
        ]
        shuffled_bins = [np.random.permutation(bin_i) for bin_i in bins]
        rand_index = np.zeros(subject_count)
        print('num_bins:', num_bins)
        if remainder == 0:
            for bin_index in range(num_bins):
                for fold_index in range(Fold_Quantity):
                    rand_index[fold_index * num_bins + bin_index] = shuffled_bins[bin_index][fold_index]
        else:
            for bin_index in range(num_bins - 1):
                for fold_index in range(Fold_Quantity):
                    if fold_index * (num_bins - 1) + bin_index >= subject_count:
                        continue
                    rand_index[fold_index * (num_bins - 1) + bin_index] = shuffled_bins[bin_index][fold_index]
            for last_index in range(len(shuffled_bins[-1])):
                rand_index[(num_bins - 1) * Fold_Quantity + last_index] = shuffled_bins[-1][last_index]

        rand_index = rand_index.astype(int)
    else:
        tmp_data = sio.loadmat(RandIndex_File)
        rand_index = tmp_data['RandIndex'].ravel()

    sio.savemat(os.path.join(ResultantFolder, 'RandIndex.mat'), {'RandIndex': rand_index})

    fold_corr = []
    fold_mae = []
    features_quantity_list = [subjects_data.shape[1] for subjects_data in Subjects_Data_List]

    for j in np.arange(Fold_Quantity):
        fold_j_index = rand_index[each_fold_size * j + np.arange(each_fold_size)]
        if remain > j:
            fold_j_index = np.insert(fold_j_index, len(fold_j_index), rand_index[each_fold_size * Fold_Quantity + j])

        subjects_data_test_list = []
        subjects_score_test = Subjects_Score[fold_j_index]
        subjects_data_train_list = []
        subjects_score_train = np.delete(Subjects_Score, fold_j_index)

        for conn_index in np.arange(len(Subjects_Data_List)):
            subjects_data_test = Subjects_Data_List[conn_index][fold_j_index, :]
            subjects_data_train = np.delete(Subjects_Data_List[conn_index], fold_j_index, axis=0)
            subjects_data_test_list.append(subjects_data_test)
            subjects_data_train_list.append(subjects_data_train)

            covariates_test = Covariates[fold_j_index, :]
            covariates_train = np.delete(Covariates, fold_j_index, axis=0)
            covariates_quantity = np.shape(Covariates)[1]
            df = {}
            df_test = {}
            for cov_index in np.arange(covariates_quantity):
                df[f'Covariate_{cov_index}'] = covariates_train[:, cov_index]
                df_test[f'Covariate_{cov_index}'] = covariates_test[:, cov_index]

            formula = 'Data'
            for cov_index in np.arange(covariates_quantity):
                if cov_index == 0 or cov_index == 2:
                    all_levels = sorted(set(Covariates[:, cov_index]))
                    term = f'C(Covariate_{cov_index}, levels={all_levels})'
                else:
                    term = f'Covariate_{cov_index}'
                formula += f' ~ {term}' if cov_index == 0 else f' + {term}'

            for feature_index in np.arange(features_quantity_list[conn_index]):
                df['Data'] = subjects_data_train_list[conn_index][:, feature_index]
                df = pd.DataFrame(df)
                df_test['Data'] = subjects_data_test_list[conn_index][:, feature_index]
                linmodel_res = sm.ols(formula=formula, data=df).fit()
                subjects_data_train_list[conn_index][:, feature_index] = linmodel_res.resid
                y_test_pred = linmodel_res.predict(df_test)
                subjects_data_test_list[conn_index][:, feature_index] = (
                    subjects_data_test_list[conn_index][:, feature_index] - y_test_pred
                )

        if Permutation_Flag:
            subjects_index_random = np.arange(len(subjects_score_train))
            np.random.shuffle(subjects_index_random)
            subjects_score_train = subjects_score_train[subjects_index_random]
            if j == 0:
                permutation_index = {'Fold_0': subjects_index_random}
            else:
                permutation_index[f'Fold_{j}'] = subjects_index_random

        for conn_index in np.arange(len(Subjects_Data_List)):
            normalize = preprocessing.MinMaxScaler()
            subjects_data_train_list[conn_index] = normalize.fit_transform(subjects_data_train_list[conn_index])
            subjects_data_test_list[conn_index] = normalize.transform(subjects_data_test_list[conn_index])

        optimal_component_number_list, inner_corr_list, inner_mae_inv_list = PLSr1_OptimalComponentNumber_KFold(
            subjects_data_train_list,
            subjects_score_train,
            Fold_Quantity,
            ComponentNumber_Range,
            CVRepeatTimes_ForInner,
            ResultantFolder,
            Parallel_Quantity,
            Feature_Name_List,
        )

        fold_j_corr_list = []
        fold_j_mae_list = []
        for conn_index in np.arange(len(Subjects_Data_List)):
            clf = cross_decomposition.PLSRegression(n_components=optimal_component_number_list[conn_index])
            clf.fit(subjects_data_train_list[conn_index], subjects_score_train)
            fold_j_score = clf.predict(subjects_data_test_list[conn_index]).T

            fold_j_corr = np.corrcoef(fold_j_score, subjects_score_test)[0, 1]
            fold_j_corr_list.append(fold_j_corr)
            fold_j_mae = np.mean(np.abs(np.subtract(fold_j_score, subjects_score_test)))
            fold_j_mae_list.append(fold_j_mae)

            weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ ** 2))
            coef_vector = clf.coef_.flatten() if clf.coef_.ndim > 1 else clf.coef_
            weight_haufe = np.dot(np.cov(np.transpose(subjects_data_train_list[conn_index])), coef_vector)
            weight_haufe = weight_haufe / np.sqrt(np.sum(weight_haufe ** 2))
            fold_j_result = {
                'Index': fold_j_index,
                'Test_Score': subjects_score_test,
                'Predict_Score': fold_j_score,
                'Corr': fold_j_corr,
                'MAE': fold_j_mae,
                'ComponentNumber': optimal_component_number_list[conn_index],
                'Inner_Corr': inner_corr_list[conn_index],
                'Inner_MAE_inv': inner_mae_inv_list[conn_index],
                'w_Brain': weight,
                'w_Brain_Haufe': weight_haufe,
            }
            fold_j_filename = f'Fold_{j}_Score.mat'
            resultant_file = os.path.join(ResultantFolder, Feature_Name_List[conn_index], fold_j_filename)
            sio.savemat(resultant_file, fold_j_result)

        fold_corr.append(fold_j_corr_list)
        fold_mae.append(fold_j_mae_list)

    mean_corr = np.array(fold_corr).mean(axis=0)
    mean_mae = np.array(fold_mae).mean(axis=0)
    for conn_index in np.arange(len(Subjects_Data_List)):
        res_nfold = {'Mean_Corr': mean_corr[conn_index], 'Mean_MAE': mean_mae[conn_index]}
        resultant_file = os.path.join(resultant_folder_list[conn_index], 'Res_NFold.mat')
        sio.savemat(resultant_file, res_nfold)

    if Permutation_Flag:
        sio.savemat(os.path.join(ResultantFolder, 'PermutationIndex.mat'), permutation_index)

    return (mean_corr, mean_mae)


def PLSr1_OptimalComponentNumber_KFold(
    Training_Data_List,
    Training_Score,
    Fold_Quantity,
    ComponentNumber_Range,
    CVRepeatTimes,
    ResultantFolder,
    Parallel_Quantity,
    Feature_Name_List,
):
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)

    subjects_quantity = len(Training_Score)
    inner_each_fold_size = int(np.fix(np.divide(subjects_quantity, Fold_Quantity)))
    remain = np.mod(subjects_quantity, Fold_Quantity)

    inner_corr_list = []
    inner_mae_inv_list = []
    for _ in np.arange(len(Training_Data_List)):
        inner_corr_list.append(np.zeros((CVRepeatTimes, Fold_Quantity, len(ComponentNumber_Range))))
        inner_mae_inv_list.append(np.zeros((CVRepeatTimes, Fold_Quantity, len(ComponentNumber_Range))))

    component_number_quantity = len(ComponentNumber_Range)
    for i in np.arange(CVRepeatTimes):
        rand_index = np.arange(subjects_quantity)
        np.random.shuffle(rand_index)

        for k in np.arange(Fold_Quantity):
            inner_fold_k_index = rand_index[inner_each_fold_size * k + np.arange(inner_each_fold_size)]
            if remain > k:
                inner_fold_k_index = np.insert(
                    inner_fold_k_index,
                    len(inner_fold_k_index),
                    rand_index[inner_each_fold_size * Fold_Quantity + k],
                )

            inner_fold_k_data_test_list = []
            inner_fold_k_score_test = Training_Score[inner_fold_k_index]
            inner_fold_k_data_train_list = []
            inner_fold_k_score_train = np.delete(Training_Score, inner_fold_k_index)

            for conn_index in np.arange(len(Training_Data_List)):
                inner_fold_k_data_test = Training_Data_List[conn_index][inner_fold_k_index, :]
                inner_fold_k_data_train = np.delete(Training_Data_List[conn_index], inner_fold_k_index, axis=0)
                inner_fold_k_data_test_list.append(inner_fold_k_data_test)
                inner_fold_k_data_train_list.append(inner_fold_k_data_train)

            for conn_index in np.arange(len(Training_Data_List)):
                scale = preprocessing.MinMaxScaler()
                inner_fold_k_data_train_list[conn_index] = scale.fit_transform(inner_fold_k_data_train_list[conn_index])
                inner_fold_k_data_test_list[conn_index] = scale.transform(inner_fold_k_data_test_list[conn_index])

            Parallel(n_jobs=Parallel_Quantity, backend='threading')(
                delayed(PLSr1_SubComponentNumber)(
                    inner_fold_k_data_train_list,
                    inner_fold_k_score_train,
                    inner_fold_k_data_test_list,
                    inner_fold_k_score_test,
                    ComponentNumber_Range[component_index],
                    component_index,
                    ResultantFolder,
                    Feature_Name_List,
                )
                for component_index in np.arange(component_number_quantity)
            )

            for conn_index in np.arange(len(Training_Data_List)):
                for component_index in np.arange(component_number_quantity):
                    component_mat_path = os.path.join(
                        ResultantFolder,
                        Feature_Name_List[conn_index],
                        f'ComponentNumber_{component_index}.mat',
                    )
                    component_mat = sio.loadmat(component_mat_path)
                    inner_corr_list[conn_index][i, k, component_index] = component_mat['Corr']
                    inner_mae_inv_list[conn_index][i, k, component_index] = component_mat['MAE_inv']
                    os.remove(component_mat_path)
                inner_corr_list[conn_index] = np.nan_to_num(inner_corr_list[conn_index])

    optimal_component_number_list = []
    for conn_index in np.arange(len(Training_Data_List)):
        inner_corr = inner_corr_list[conn_index]
        inner_mae_inv = inner_mae_inv_list[conn_index]

        inner_corr_cvmean = np.mean(inner_corr, axis=0)
        inner_mae_inv_cvmean = np.mean(inner_mae_inv, axis=0)

        inner_corr_foldmean = np.mean(inner_corr_cvmean, axis=0)
        inner_mae_inv_foldmean = np.mean(inner_mae_inv_cvmean, axis=0)
        inner_corr_foldmean = (inner_corr_foldmean - np.mean(inner_corr_foldmean)) / np.std(inner_corr_foldmean)
        inner_mae_inv_foldmean = (inner_mae_inv_foldmean - np.mean(inner_mae_inv_foldmean)) / np.std(inner_mae_inv_foldmean)
        inner_evaluation = inner_corr_foldmean + inner_mae_inv_foldmean

        inner_evaluation_mat = {
            'Inner_Corr_CVMean': inner_corr_cvmean,
            'Inner_MAE_inv_CVMean': inner_mae_inv_cvmean,
            'Inner_Corr_FoldMean': inner_corr_foldmean,
            'Inner_MAE_inv_FoldMean': inner_mae_inv_foldmean,
            'Inner_Evaluation': inner_evaluation,
        }
        sio.savemat(
            os.path.join(ResultantFolder, Feature_Name_List[conn_index], 'Inner_Evaluation.mat'),
            inner_evaluation_mat,
        )

        optimal_component_number_index = np.argmax(inner_evaluation)
        optimal_component_number_list.append(ComponentNumber_Range[optimal_component_number_index])

    return (optimal_component_number_list, inner_corr_list, inner_mae_inv_list)


def PLSr1_SubComponentNumber(
    Training_Data_List,
    Training_Score,
    Testing_Data_List,
    Testing_Score,
    ComponentNumber,
    ComponentNumber_ID,
    ResultantFolder,
    Feature_Name_List,
):
    for conn_index in np.arange(len(Training_Data_List)):
        clf = cross_decomposition.PLSRegression(n_components=ComponentNumber)
        clf.fit(Training_Data_List[conn_index], Training_Score)
        predict_score = clf.predict(Testing_Data_List[conn_index]).T
        corr = np.corrcoef(predict_score, Testing_Score)[0, 1]
        mae_inv = np.divide(1, np.mean(np.abs(predict_score - Testing_Score)))
        resultant_file = os.path.join(
            ResultantFolder,
            Feature_Name_List[conn_index],
            f'ComponentNumber_{ComponentNumber_ID}.mat',
        )
        sio.savemat(resultant_file, {'Corr': corr, 'MAE_inv': mae_inv})
