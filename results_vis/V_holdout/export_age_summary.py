import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats


DEFAULT_DATA_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data'
DEFAULT_DATASETS = ['EFNY', 'HCPD', 'CCNP', 'PNC']
P_METRIC_KEYS = [
    'GG_empirical_p',
    'GW_empirical_p',
    'WW_empirical_p',
    'GW_partial_empirical_p',
    'WW_partial_empirical_p',
]
Q_METRIC_KEYS = [
    'GG_fdr_q',
    'GW_fdr_q',
    'WW_fdr_q',
    'GW_partial_fdr_q',
    'WW_partial_fdr_q',
]
Q_SIGNIFICANCE_KEYS = [
    'GG_fdr_significance',
    'GW_fdr_significance',
    'WW_fdr_significance',
    'GW_partial_fdr_significance',
    'WW_partial_fdr_significance',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export age holdout correlations and permutation significance results.'
    )
    parser.add_argument(
        '--data_root',
        default=DEFAULT_DATA_ROOT,
        help='Root directory containing dataset folders such as EFNY/HCPD/CCNP/PNC.',
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=DEFAULT_DATASETS,
        help='Age datasets to summarize. Default: EFNY HCPD CCNP PNC.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional holdout seed. If provided, use V_holdout_<seed>.',
    )
    parser.add_argument(
        '--holdout_dir_name',
        default=None,
        help='Explicit holdout directory name such as V_holdout_42. Overrides --seed.',
    )
    parser.add_argument(
        '--output_root',
        default=None,
        help='Directory for exported CSV and MAT summaries. Default: --data_root.',
    )
    parser.add_argument(
        '--output_prefix',
        default=None,
        help='Optional output file prefix. Default: <holdout_dir_name>_age.',
    )
    parser.add_argument(
        '--skip_permutation',
        action='store_true',
        help='Skip permutation loading and export observed holdout metrics only.',
    )
    return parser.parse_args()


def partial_corr(x, y, z):
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
    return float(np.sum(valid_null >= observed_value) / float(valid_null.size))


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


def compute_bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    valid_mask = ~np.isnan(p_values)
    if not np.any(valid_mask):
        return q_values

    valid_p = p_values[valid_mask]
    order = np.argsort(valid_p, kind='mergesort')
    ranked_p = valid_p[order]
    n_tests = float(valid_p.size)
    ranked_q = ranked_p * n_tests / np.arange(1.0, n_tests + 1.0)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)

    valid_q = np.empty(valid_p.shape, dtype=float)
    valid_q[order] = ranked_q
    q_values[valid_mask] = valid_q
    return q_values


def apply_fdr_to_results(results_summary):
    if not results_summary:
        return

    flat_p_values = []
    for result in results_summary:
        for p_key in P_METRIC_KEYS:
            flat_p_values.append(result[p_key])

    flat_q_values = compute_bh_fdr(flat_p_values)
    q_index = 0
    for result in results_summary:
        for q_key, q_sig_key in zip(Q_METRIC_KEYS, Q_SIGNIFICANCE_KEYS):
            q_value = flat_q_values[q_index]
            result[q_key] = q_value
            result[q_sig_key] = get_significance_label(q_value)
            q_index += 1


def list_time_ids(base_folder):
    return sorted(
        int(item.split('_')[1])
        for item in os.listdir(base_folder)
        if item.startswith('Time_') and os.path.isdir(os.path.join(base_folder, item))
    )


def load_holdout_score(base_folder, run_id, fc_type):
    score_file = os.path.join(base_folder, 'Time_{}'.format(run_id), fc_type, 'Holdout_Score.mat')
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


def get_rank_order(corr_values):
    corr_values = np.asarray(corr_values, dtype=float)
    temp_values = corr_values.copy()
    temp_values[np.isnan(temp_values)] = -np.inf
    return np.argsort(-temp_values)


def compute_partial_series(base_folder, time_ids, corr_actual_gg, corr_actual_gw, corr_actual_ww):
    partial_r_gw_total = np.full(len(time_ids), np.nan)
    partial_r_ww_total = np.full(len(time_ids), np.nan)

    for idx, run_id in enumerate(time_ids):
        if (
            np.isnan(corr_actual_gg[idx])
            or np.isnan(corr_actual_gw[idx])
            or np.isnan(corr_actual_ww[idx])
        ):
            continue

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
            warnings.warn(
                'Test index mismatch at Time_{} for GG/GW/WW. Skipping.'.format(
                    run_id,
                )
            )
            continue

        partial_r_gw_total[idx] = partial_corr(
            gw_score['predict'][sort_gw],
            gw_score['test'][sort_gw],
            gg_score['predict'][sort_gg],
        )
        partial_r_ww_total[idx] = partial_corr(
            ww_score['predict'][sort_ww],
            ww_score['test'][sort_ww],
            gg_score['predict'][sort_gg],
        )

    return partial_r_gw_total, partial_r_ww_total


def summarize_holdout_metric(observed_values, permutation_values, metric_name, dataset):
    observed_values = np.asarray(observed_values, dtype=float)
    permutation_values = np.asarray(permutation_values, dtype=float)

    valid_observed = observed_values[~np.isnan(observed_values)]
    valid_permutation = permutation_values[~np.isnan(permutation_values)]

    if valid_observed.size == 0:
        observed_value = np.nan
    else:
        observed_value = float(valid_observed[0])
        if valid_observed.size > 1:
            warnings.warn(
                '{} has {} observed holdout values for {}; using the first valid half test-set result.'.format(
                    dataset,
                    valid_observed.size,
                    metric_name,
                )
            )

    permutation_mean = float(np.mean(valid_permutation)) if valid_permutation.size else np.nan
    permutation_median = float(np.median(valid_permutation)) if valid_permutation.size else np.nan
    permutation_std = float(np.std(valid_permutation, ddof=1)) if valid_permutation.size > 1 else np.nan
    p_value = compute_empirical_right_tail_p(observed_value, valid_permutation)

    return {
        'observed_value': observed_value,
        'n_observed': int(valid_observed.size),
        'n_permutation': int(valid_permutation.size),
        'permutation_mean': permutation_mean,
        'permutation_median': permutation_median,
        'permutation_std': permutation_std,
        'empirical_p_right_tail': p_value,
        'significance_label': get_significance_label(p_value),
    }


def resolve_holdout_dir_name(args):
    if args.holdout_dir_name:
        return args.holdout_dir_name
    if args.seed is not None:
        return 'V_holdout_{}'.format(args.seed)
    return 'V_holdout'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_mat_cells(results_summary, all_data):
    n_valid = len(results_summary)
    cell_r_gg = np.zeros((n_valid, 3), dtype=object)
    cell_r_gw = np.zeros((n_valid, 3), dtype=object)
    cell_r_ww = np.zeros((n_valid, 3), dtype=object)
    cell_partial_gw = np.zeros((n_valid, 2), dtype=object)
    cell_partial_ww = np.zeros((n_valid, 2), dtype=object)
    cell_observed_results = np.zeros((n_valid, 6), dtype=object)
    cell_significance_results = np.zeros((n_valid, 21), dtype=object)

    for idx, res in enumerate(results_summary):
        dataset = res['dataset']

        cell_r_gg[idx, 0] = dataset
        cell_r_gg[idx, 1] = all_data['R_gg'][dataset]['Corr']
        cell_r_gg[idx, 2] = all_data['R_gg'][dataset]['MAE']

        cell_r_gw[idx, 0] = dataset
        cell_r_gw[idx, 1] = all_data['R_gw'][dataset]['Corr']
        cell_r_gw[idx, 2] = all_data['R_gw'][dataset]['MAE']

        cell_r_ww[idx, 0] = dataset
        cell_r_ww[idx, 1] = all_data['R_ww'][dataset]['Corr']
        cell_r_ww[idx, 2] = all_data['R_ww'][dataset]['MAE']

        cell_partial_gw[idx, 0] = dataset
        cell_partial_gw[idx, 1] = all_data['partialR_gw'][dataset]

        cell_partial_ww[idx, 0] = dataset
        cell_partial_ww[idx, 1] = all_data['partialR_ww'][dataset]

        cell_observed_results[idx, 0] = dataset
        cell_observed_results[idx, 1] = res['GG_corr']
        cell_observed_results[idx, 2] = res['GW_corr']
        cell_observed_results[idx, 3] = res['WW_corr']
        cell_observed_results[idx, 4] = res['GW_partial_corr']
        cell_observed_results[idx, 5] = res['WW_partial_corr']

        cell_significance_results[idx, 0] = dataset
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
        cell_significance_results[idx, 11] = res['GG_fdr_q']
        cell_significance_results[idx, 12] = res['GW_fdr_q']
        cell_significance_results[idx, 13] = res['WW_fdr_q']
        cell_significance_results[idx, 14] = res['GW_partial_fdr_q']
        cell_significance_results[idx, 15] = res['WW_partial_fdr_q']
        cell_significance_results[idx, 16] = res['GG_fdr_significance']
        cell_significance_results[idx, 17] = res['GW_fdr_significance']
        cell_significance_results[idx, 18] = res['WW_fdr_significance']
        cell_significance_results[idx, 19] = res['GW_partial_fdr_significance']
        cell_significance_results[idx, 20] = res['WW_partial_fdr_significance']

    return {
        'R_gg_age': cell_r_gg,
        'R_gw_age': cell_r_gw,
        'R_ww_age': cell_r_ww,
        'partialR_gw_age': cell_partial_gw,
        'partialR_ww_age': cell_partial_ww,
        'observedResults_age': cell_observed_results,
        'permutationSignificance_age': cell_significance_results,
    }


def main():
    args = parse_args()
    holdout_dir_name = resolve_holdout_dir_name(args)
    output_root = ensure_dir(args.output_root or args.data_root)
    output_prefix = args.output_prefix or '{}_age'.format(holdout_dir_name)

    results_summary = []
    all_data = {
        'R_gg': {},
        'R_gw': {},
        'R_ww': {},
        'partialR_gw': {},
        'partialR_ww': {},
    }

    for dataset in args.datasets:
        prediction_root = os.path.join(args.data_root, dataset, 'prediction', 'age')
        base_folder = os.path.join(
            prediction_root,
            holdout_dir_name,
            'RegressCovariates_Holdout',
        )
        permutation_folder = os.path.join(
            prediction_root,
            holdout_dir_name,
            'RegressCovariates_Holdout_Permutation',
        )

        if not os.path.isdir(base_folder):
            warnings.warn(
                'Observed holdout folder not found for {}: {}. Skipping.'.format(
                    dataset,
                    base_folder,
                )
            )
            continue

        actual_time_ids = list_time_ids(base_folder)
        if not actual_time_ids:
            warnings.warn(
                'No Time_* folders found for {}: {}. Skipping.'.format(
                    dataset,
                    base_folder,
                )
            )
            continue

        corr_actual_gg, mae_actual_gg = load_corr_mae_arrays(base_folder, 'GGFC', actual_time_ids)
        corr_actual_gw, mae_actual_gw = load_corr_mae_arrays(base_folder, 'GWFC', actual_time_ids)
        corr_actual_ww, mae_actual_ww = load_corr_mae_arrays(base_folder, 'WWFC', actual_time_ids)
        partial_r_gw_total, partial_r_ww_total = compute_partial_series(
            base_folder,
            actual_time_ids,
            corr_actual_gg,
            corr_actual_gw,
            corr_actual_ww,
        )

        all_data['R_gg'][dataset] = {'Corr': corr_actual_gg, 'MAE': mae_actual_gg}
        all_data['R_gw'][dataset] = {'Corr': corr_actual_gw, 'MAE': mae_actual_gw}
        all_data['R_ww'][dataset] = {'Corr': corr_actual_ww, 'MAE': mae_actual_ww}
        all_data['partialR_gw'][dataset] = partial_r_gw_total
        all_data['partialR_ww'][dataset] = partial_r_ww_total

        corr_perm_gg = np.array([])
        corr_perm_gw = np.array([])
        corr_perm_ww = np.array([])
        partial_perm_gw = np.array([])
        partial_perm_ww = np.array([])

        if args.skip_permutation:
            pass
        elif os.path.isdir(permutation_folder):
            permutation_time_ids = list_time_ids(permutation_folder)
            if permutation_time_ids:
                corr_perm_gg, _ = load_corr_mae_arrays(permutation_folder, 'GGFC', permutation_time_ids)
                corr_perm_gw, _ = load_corr_mae_arrays(permutation_folder, 'GWFC', permutation_time_ids)
                corr_perm_ww, _ = load_corr_mae_arrays(permutation_folder, 'WWFC', permutation_time_ids)
                partial_perm_gw, partial_perm_ww = compute_partial_series(
                    permutation_folder,
                    permutation_time_ids,
                    corr_perm_gg,
                    corr_perm_gw,
                    corr_perm_ww,
                )
            else:
                warnings.warn('Permutation folder has no Time_* runs: {}'.format(permutation_folder))
        else:
            warnings.warn(
                'Permutation folder not found for {}: {}'.format(
                    dataset,
                    permutation_folder,
                )
            )

        gg_summary = summarize_holdout_metric(corr_actual_gg, corr_perm_gg, 'GG_corr', dataset)
        gw_summary = summarize_holdout_metric(corr_actual_gw, corr_perm_gw, 'GW_corr', dataset)
        ww_summary = summarize_holdout_metric(corr_actual_ww, corr_perm_ww, 'WW_corr', dataset)
        partial_gw_summary = summarize_holdout_metric(
            partial_r_gw_total,
            partial_perm_gw,
            'GW_partial_corr',
            dataset,
        )
        partial_ww_summary = summarize_holdout_metric(
            partial_r_ww_total,
            partial_perm_ww,
            'WW_partial_corr',
            dataset,
        )

        results_summary.append(
            {
                'dataset': dataset,
                'targetStr': 'age',
                'holdout_dir_name': holdout_dir_name,
                'skip_permutation': args.skip_permutation,
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
                'GG_fdr_q': np.nan,
                'GW_fdr_q': np.nan,
                'WW_fdr_q': np.nan,
                'GW_partial_fdr_q': np.nan,
                'WW_partial_fdr_q': np.nan,
                'GG_fdr_significance': np.nan,
                'GW_fdr_significance': np.nan,
                'WW_fdr_significance': np.nan,
                'GW_partial_fdr_significance': np.nan,
                'WW_partial_fdr_significance': np.nan,
            }
        )

    if not results_summary:
        raise SystemExit('No valid age datasets were processed.')

    apply_fdr_to_results(results_summary)
    result_df = pd.DataFrame(results_summary)
    output_csv = os.path.join(output_root, '{}_summary.csv'.format(output_prefix))
    output_mat = os.path.join(output_root, '{}_summary.mat'.format(output_prefix))

    result_df.to_csv(output_csv, index=False)
    mat_dict = build_mat_cells(results_summary, all_data)
    mat_dict['holdoutDirName'] = np.array([holdout_dir_name], dtype=object)
    sio.savemat(output_mat, mat_dict)

    print(result_df.to_string(index=False))
    print('\nSaved age holdout summary to:\n  {}\n  {}'.format(output_csv, output_mat))


if __name__ == '__main__':
    main()
