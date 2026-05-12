import os
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats


def partial_corr(x, y, z):
    """
    Calculate partial correlation between x and y, controlling for z.
    Formula equivalent to MATLAB's partialcorr.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    if np.sum(mask) < 2:
        return np.nan

    x = x[mask]
    y = y[mask]
    z = z[mask]

    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)

    denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denominator == 0:
        return np.nan

    return (r_xy - (r_xz * r_yz)) / denominator


def compute_empirical_right_tail_p(observed_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    valid_null = null_values[~np.isnan(null_values)]
    if valid_null.size == 0 or np.isnan(observed_value):
        return np.nan
    return float((np.sum(valid_null >= observed_value) + 1.0) / (valid_null.size + 1.0))


def get_significance_label(p_value):
    if np.isnan(p_value):
        return np.nan
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def list_time_ids(base_folder):
    return sorted(
        int(item.split('_')[1])
        for item in os.listdir(base_folder)
        if item.startswith('Time_') and os.path.isdir(os.path.join(base_folder, item))
    )


def list_target_ids(project_folder):
    target_ids = []
    for item in sorted(os.listdir(project_folder)):
        holdout_folder = os.path.join(
            project_folder,
            item,
            'V_holdout',
            'RegressCovariates_Holdout',
        )
        if os.path.isdir(holdout_folder):
            target_ids.append(item)
    return target_ids


def load_holdout_score(base_folder, run_id, fc_type):
    score_file = os.path.join(base_folder, f'Time_{run_id}', fc_type, 'Holdout_Score.mat')
    if not os.path.isfile(score_file):
        return None

    try:
        mat = sio.loadmat(score_file)
        return {
            'index': mat['Test_Index'].flatten().astype(int),
            'predict': mat['Predict_Score'].flatten().astype(float),
            'test': mat['Test_Score'].flatten().astype(float),
            'corr': float(mat['Corr'].item()),
            'mae': float(mat['MAE'].item()),
        }
    except Exception:
        return None


def load_corr_mae_arrays(base_folder, fc_type, time_ids):
    corr_values = np.full(len(time_ids), np.nan)
    mae_values = np.full(len(time_ids), np.nan)

    for idx, run_id in enumerate(time_ids):
        score = load_holdout_score(base_folder, run_id, fc_type)
        if score is None:
            continue
        corr_values[idx] = score['corr']
        mae_values[idx] = score['mae']

    return corr_values, mae_values


def compute_partial_series(base_folder, time_ids):
    partial_r_gw_total = np.full(len(time_ids), np.nan)
    partial_r_ww_total = np.full(len(time_ids), np.nan)

    for run_idx, run_id in enumerate(time_ids):
        gg_score = load_holdout_score(base_folder, run_id, 'GGFC')
        gw_score = load_holdout_score(base_folder, run_id, 'GWFC')
        ww_score = load_holdout_score(base_folder, run_id, 'WWFC')

        if gg_score is None or gw_score is None or ww_score is None:
            continue

        sort_gg = np.argsort(gg_score['index'])
        sort_gw = np.argsort(gw_score['index'])
        sort_ww = np.argsort(ww_score['index'])

        if not (
            np.array_equal(gg_score['index'][sort_gg], gw_score['index'][sort_gw])
            and np.array_equal(gg_score['index'][sort_gg], ww_score['index'][sort_ww])
        ):
            warnings.warn(f'Test index mismatch at Time_{run_id}. Skipping.')
            continue

        try:
            partial_r_gw_total[run_idx] = partial_corr(
                gw_score['predict'][sort_gw],
                gw_score['test'][sort_gw],
                gg_score['predict'][sort_gg],
            )
            partial_r_ww_total[run_idx] = partial_corr(
                ww_score['predict'][sort_ww],
                ww_score['test'][sort_ww],
                gg_score['predict'][sort_gg],
            )
        except Exception:
            pass

    return partial_r_gw_total, partial_r_ww_total


def summarize_metric(observed_values, permutation_values):
    observed_values = np.asarray(observed_values, dtype=float)
    permutation_values = np.asarray(permutation_values, dtype=float)

    valid_observed = observed_values[~np.isnan(observed_values)]
    valid_permutation = permutation_values[~np.isnan(permutation_values)]

    observed_value = float(valid_observed[0]) if valid_observed.size == 1 else np.nan
    observed_median = float(np.nanmedian(valid_observed)) if valid_observed.size else np.nan
    permutation_mean = float(np.mean(valid_permutation)) if valid_permutation.size else np.nan
    permutation_median = float(np.median(valid_permutation)) if valid_permutation.size else np.nan
    permutation_std = float(np.std(valid_permutation, ddof=1)) if valid_permutation.size > 1 else np.nan
    p_value = compute_empirical_right_tail_p(observed_value, valid_permutation)

    return {
        'observed_value': observed_value,
        'observed_median': observed_median,
        'n_observed': int(valid_observed.size),
        'n_permutation': int(valid_permutation.size),
        'permutation_mean': permutation_mean,
        'permutation_median': permutation_median,
        'permutation_std': permutation_std,
        'empirical_p_right_tail': p_value,
        'significance_label': get_significance_label(p_value),
    }


ProjectFolder = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/ABCD/prediction'
targetStr_total = list_target_ids(ProjectFolder)
num_targets = len(targetStr_total)

results_summary = []
all_data = {
    'R_gg': {},
    'R_gw': {},
    'R_ww': {},
    'partialR_gw': {},
    'partialR_ww': {},
    'perm_R_gg': {},
    'perm_R_gw': {},
    'perm_R_ww': {},
    'perm_partialR_gw': {},
    'perm_partialR_ww': {},
}

for i_str, target_str in enumerate(targetStr_total):
    print(f'\n[{i_str + 1}/{num_targets}] Processing target: {target_str}')

    base_folder = os.path.join(ProjectFolder, target_str, 'V_holdout', 'RegressCovariates_Holdout')
    permutation_folder = os.path.join(
        ProjectFolder,
        target_str,
        'V_holdout',
        'RegressCovariates_Holdout_Permutation',
    )

    actual_time_ids = list_time_ids(base_folder)
    if not actual_time_ids:
        warnings.warn(f'No Time_* folders found for {target_str}: {base_folder}. Skipping.')
        continue

    print(f'  Found {len(actual_time_ids)} observed holdout runs.')

    corr_actual_gg, mae_actual_gg = load_corr_mae_arrays(base_folder, 'GGFC', actual_time_ids)
    corr_actual_gw, mae_actual_gw = load_corr_mae_arrays(base_folder, 'GWFC', actual_time_ids)
    corr_actual_ww, mae_actual_ww = load_corr_mae_arrays(base_folder, 'WWFC', actual_time_ids)

    all_data['R_gg'][target_str] = {'Corr': corr_actual_gg, 'MAE': mae_actual_gg}
    all_data['R_gw'][target_str] = {'Corr': corr_actual_gw, 'MAE': mae_actual_gw}
    all_data['R_ww'][target_str] = {'Corr': corr_actual_ww, 'MAE': mae_actual_ww}

    print('  Calculating observed partial correlations...')
    partial_r_gw_total, partial_r_ww_total = compute_partial_series(base_folder, actual_time_ids)
    all_data['partialR_gw'][target_str] = partial_r_gw_total
    all_data['partialR_ww'][target_str] = partial_r_ww_total

    corr_perm_gg = np.array([])
    mae_perm_gg = np.array([])
    corr_perm_gw = np.array([])
    mae_perm_gw = np.array([])
    corr_perm_ww = np.array([])
    mae_perm_ww = np.array([])
    partial_perm_gw = np.array([])
    partial_perm_ww = np.array([])

    if os.path.exists(permutation_folder):
        permutation_time_ids = list_time_ids(permutation_folder)
        print(f'  Found {len(permutation_time_ids)} permutation holdout runs.')
        if permutation_time_ids:
            corr_perm_gg, mae_perm_gg = load_corr_mae_arrays(permutation_folder, 'GGFC', permutation_time_ids)
            corr_perm_gw, mae_perm_gw = load_corr_mae_arrays(permutation_folder, 'GWFC', permutation_time_ids)
            corr_perm_ww, mae_perm_ww = load_corr_mae_arrays(permutation_folder, 'WWFC', permutation_time_ids)
            print('  Calculating permutation partial correlations...')
            partial_perm_gw, partial_perm_ww = compute_partial_series(permutation_folder, permutation_time_ids)
        else:
            warnings.warn(f'Permutation folder has no Time_* runs: {permutation_folder}')
    else:
        warnings.warn(f'Permutation folder not found for {target_str}: {permutation_folder}')

    all_data['perm_R_gg'][target_str] = {'Corr': corr_perm_gg, 'MAE': mae_perm_gg}
    all_data['perm_R_gw'][target_str] = {'Corr': corr_perm_gw, 'MAE': mae_perm_gw}
    all_data['perm_R_ww'][target_str] = {'Corr': corr_perm_ww, 'MAE': mae_perm_ww}
    all_data['perm_partialR_gw'][target_str] = partial_perm_gw
    all_data['perm_partialR_ww'][target_str] = partial_perm_ww

    gg_summary = summarize_metric(corr_actual_gg, corr_perm_gg)
    gw_summary = summarize_metric(corr_actual_gw, corr_perm_gw)
    ww_summary = summarize_metric(corr_actual_ww, corr_perm_ww)
    partial_gw_summary = summarize_metric(partial_r_gw_total, partial_perm_gw)
    partial_ww_summary = summarize_metric(partial_r_ww_total, partial_perm_ww)

    print(f"    GG corr: {gg_summary['observed_value']:.4f}, p={gg_summary['empirical_p_right_tail']}")
    print(f"    GW corr: {gw_summary['observed_value']:.4f}, p={gw_summary['empirical_p_right_tail']}")
    print(f"    WW corr: {ww_summary['observed_value']:.4f}, p={ww_summary['empirical_p_right_tail']}")
    print(f"    GW partial corr: {partial_gw_summary['observed_value']:.4f}, p={partial_gw_summary['empirical_p_right_tail']}")
    print(f"    WW partial corr: {partial_ww_summary['observed_value']:.4f}, p={partial_ww_summary['empirical_p_right_tail']}")

    results_summary.append(
        {
            'targetStr': target_str,
            'n_observed_runs': gg_summary['n_observed'],
            'n_permutation_runs': gg_summary['n_permutation'],
            'GG_corr': gg_summary['observed_value'],
            'GW_corr': gw_summary['observed_value'],
            'WW_corr': ww_summary['observed_value'],
            'GW_partial_corr': partial_gw_summary['observed_value'],
            'WW_partial_corr': partial_ww_summary['observed_value'],
            'GG_perm_mean': gg_summary['permutation_mean'],
            'GW_perm_mean': gw_summary['permutation_mean'],
            'WW_perm_mean': ww_summary['permutation_mean'],
            'GW_partial_perm_mean': partial_gw_summary['permutation_mean'],
            'WW_partial_perm_mean': partial_ww_summary['permutation_mean'],
            'GG_empirical_p': gg_summary['empirical_p_right_tail'],
            'GW_empirical_p': gw_summary['empirical_p_right_tail'],
            'WW_empirical_p': ww_summary['empirical_p_right_tail'],
            'GW_partial_empirical_p': partial_gw_summary['empirical_p_right_tail'],
            'WW_partial_empirical_p': partial_ww_summary['empirical_p_right_tail'],
            'GG_significance': gg_summary['significance_label'],
            'GW_significance': gw_summary['significance_label'],
            'WW_significance': ww_summary['significance_label'],
            'GW_partial_significance': partial_gw_summary['significance_label'],
            'WW_partial_significance': partial_ww_summary['significance_label'],
        }
    )

print('\nFormatting data for MATLAB compatibility...')

valid_targets = [res['targetStr'] for res in results_summary]
n_valid = len(valid_targets)

if n_valid == 0:
    print('No valid targets processed. Exiting.')
    raise SystemExit(0)

cell_R_gg = np.zeros((n_valid, 3), dtype=object)
cell_R_gw = np.zeros((n_valid, 3), dtype=object)
cell_R_ww = np.zeros((n_valid, 3), dtype=object)
cell_partial_gw = np.zeros((n_valid, 2), dtype=object)
cell_partial_ww = np.zeros((n_valid, 2), dtype=object)
cell_observed_results = np.zeros((n_valid, 6), dtype=object)
cell_significance_results = np.zeros((n_valid, 11), dtype=object)

for idx, res in enumerate(results_summary):
    t_str = res['targetStr']

    cell_R_gg[idx, 0] = t_str
    cell_R_gg[idx, 1] = all_data['R_gg'][t_str]['Corr']
    cell_R_gg[idx, 2] = all_data['R_gg'][t_str]['MAE']

    cell_R_gw[idx, 0] = t_str
    cell_R_gw[idx, 1] = all_data['R_gw'][t_str]['Corr']
    cell_R_gw[idx, 2] = all_data['R_gw'][t_str]['MAE']

    cell_R_ww[idx, 0] = t_str
    cell_R_ww[idx, 1] = all_data['R_ww'][t_str]['Corr']
    cell_R_ww[idx, 2] = all_data['R_ww'][t_str]['MAE']

    cell_partial_gw[idx, 0] = t_str
    cell_partial_gw[idx, 1] = all_data['partialR_gw'][t_str]

    cell_partial_ww[idx, 0] = t_str
    cell_partial_ww[idx, 1] = all_data['partialR_ww'][t_str]

    cell_observed_results[idx, 0] = t_str
    cell_observed_results[idx, 1] = res['GG_corr']
    cell_observed_results[idx, 2] = res['GW_corr']
    cell_observed_results[idx, 3] = res['WW_corr']
    cell_observed_results[idx, 4] = res['GW_partial_corr']
    cell_observed_results[idx, 5] = res['WW_partial_corr']

    cell_significance_results[idx, 0] = t_str
    cell_significance_results[idx, 1] = res['GG_empirical_p']
    cell_significance_results[idx, 2] = res['GW_empirical_p']
    cell_significance_results[idx, 3] = res['WW_empirical_p']
    cell_significance_results[idx, 4] = res['GW_partial_empirical_p']
    cell_significance_results[idx, 5] = res['WW_partial_empirical_p']
    cell_significance_results[idx, 6] = res['GG_significance']
    cell_significance_results[idx, 7] = res['GW_significance']
    cell_significance_results[idx, 8] = res['WW_significance']
    cell_significance_results[idx, 9] = res['GW_partial_significance']
    cell_significance_results[idx, 10] = res['WW_partial_significance']

df_results = pd.DataFrame(results_summary)
print('\nFinal Results Table:')
print(df_results)
df_results.to_csv(os.path.join(ProjectFolder, 'V_holdout_partial_results_total_multi_targets.csv'), index=False)

output_file = os.path.join(ProjectFolder, 'V_holdout_partial_results_total_multi_targets.mat')
boxplot_file = os.path.join(ProjectFolder, 'V_holdout_partial_results_forBoxplot_multi_targets.mat')

mat_dict = {
    'R_gg_totalStr': cell_R_gg,
    'R_gw_totalStr': cell_R_gw,
    'R_ww_totalStr': cell_R_ww,
    'partialR_gw_totalStr': cell_partial_gw,
    'partialR_ww_totalStr': cell_partial_ww,
    'observedResults_totalStr': cell_observed_results,
    'permutationSignificance_totalStr': cell_significance_results,
}

sio.savemat(output_file, mat_dict)

header = np.array(
    ['targetStr', 'R_gg_total', 'R_gw_total', 'partialR_gw_total', 'R_ww_total', 'partialR_ww_total'],
    dtype=object,
)
data_rows = np.zeros((n_valid, 6), dtype=object)
data_rows[:, 0] = valid_targets
data_rows[:, 1] = cell_R_gg[:, 1]
data_rows[:, 2] = cell_R_gw[:, 1]
data_rows[:, 3] = cell_partial_gw[:, 1]
data_rows[:, 4] = cell_R_ww[:, 1]
data_rows[:, 5] = cell_partial_ww[:, 1]
dataCell = np.vstack((header, data_rows))

sio.savemat(boxplot_file, {'dataCell': dataCell})

print(f'\nSaved successfully to:\n  {output_file}\n  {boxplot_file}')
