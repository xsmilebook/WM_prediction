import os
import numpy as np

FEATURE_NAME_LIST = [
    'GG_GW_MergedFC',
    'GG_WW_MergedFC',
    'GW_WW_MergedFC',
    'GG_GW_WW_MergedFC',
]


def build_merged_feature_sets(gg_data, gw_data, ww_data):
    sample_sizes = {gg_data.shape[0], gw_data.shape[0], ww_data.shape[0]}
    if len(sample_sizes) != 1:
        raise ValueError('GG, GW, and WW feature matrices must have the same number of subjects.')

    subjects_data = [
        np.concatenate([gg_data, gw_data], axis=1),
        np.concatenate([gg_data, ww_data], axis=1),
        np.concatenate([gw_data, ww_data], axis=1),
        np.concatenate([gg_data, gw_data, ww_data], axis=1),
    ]
    return FEATURE_NAME_LIST.copy(), subjects_data


def build_randindex_file_list(base_prediction_folder, cv_times):
    randindex_file_list = [
        os.path.join(base_prediction_folder, 'RegressCovariates_RandomCV', f'Time_{i}', 'RandIndex.mat')
        for i in range(cv_times)
    ]
    missing_files = [path for path in randindex_file_list if not os.path.exists(path)]
    if missing_files:
        missing_preview = '\n'.join(missing_files[:5])
        raise FileNotFoundError(f'Baseline RandIndex files are missing:\n{missing_preview}')
    return randindex_file_list
