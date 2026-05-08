import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

DEFAULT_PROJECT_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data'
DEFAULT_OUTPUT_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/results/V_siblings'

TASK_TARGETS = {
    'cognition': [
        'nihtbx_cryst_uncorrected',
        'nihtbx_fluidcomp_uncorrected',
        'nihtbx_totalcomp_uncorrected',
    ],
    'pfactor': ['General', 'Ext', 'ADHD'],
}

BASELINE_FEATURES = ['GGFC', 'GWFC', 'WWFC']
METRIC_ORDER = ['GG', 'GW', 'WW', 'GW/GG', 'WW/GG']
FEATURE_LABELS = {
    'GGFC': 'GG',
    'GWFC': 'GW',
    'WWFC': 'WW',
}
NUM_FOLDS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute empirical permutation p-values for ABCD V_siblings prediction metrics.'
    )
    parser.add_argument(
        '--dataset',
        default='ABCD',
        help='Dataset name. Default: ABCD.',
    )
    parser.add_argument(
        '--task',
        required=True,
        choices=sorted(TASK_TARGETS.keys()),
        help='Target family to summarize.',
    )
    parser.add_argument(
        '--targets',
        nargs='*',
        default=None,
        help='Optional explicit target list. Defaults depend on --task.',
    )
    parser.add_argument(
        '--project_root',
        default=DEFAULT_PROJECT_ROOT,
        help='Root directory containing per-dataset prediction folders.',
    )
    parser.add_argument(
        '--output_root',
        default=DEFAULT_OUTPUT_ROOT,
        help='Directory for exported CSV results.',
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_significance_label(p_value):
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def compute_empirical_right_tail_p(observed_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    valid_null = null_values[~np.isnan(null_values)]
    if valid_null.size == 0:
        return np.nan
    return float((np.sum(valid_null >= observed_value) + 1.0) / (valid_null.size + 1.0))


def load_corr_series(result_dir, feature_name):
    time_dirs = sorted(
        item for item in os.listdir(result_dir) if item.startswith('Time_')
    )
    rows = []
    for time_dir in time_dirs:
        time_id = int(time_dir.split('_')[1])
        result_path = os.path.join(result_dir, time_dir, feature_name, 'Res_NFold.mat')
        if not os.path.exists(result_path):
            raise FileNotFoundError(f'Missing result file: {result_path}')
        result_mat = sio.loadmat(result_path)
        rows.append({'time_id': time_id, 'corr': float(result_mat['Mean_Corr'].item())})
    return pd.DataFrame(rows).sort_values('time_id').reset_index(drop=True)


def partial_corr(x, y, z):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    if np.sum(valid) < 3:
        return np.nan

    x = x[valid]
    y = y[valid]
    z = z[valid]

    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)

    denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denominator == 0:
        return np.nan
    return float((r_xy - r_xz * r_yz) / denominator)


def load_fold_arrays(result_dir, time_id, feature_name):
    indices = []
    predicted = []
    test_scores = []
    for fold_id in range(NUM_FOLDS):
        score_path = os.path.join(
            result_dir,
            f'Time_{time_id}',
            feature_name,
            f'Fold_{fold_id}_Score.mat',
        )
        if not os.path.exists(score_path):
            raise FileNotFoundError(f'Missing fold score file: {score_path}')
        score_mat = sio.loadmat(score_path)
        indices.extend(score_mat['Index'].flatten())
        predicted.extend(score_mat['Predict_Score'].flatten())
        test_scores.extend(score_mat['Test_Score'].flatten())
    return (
        np.asarray(indices),
        np.asarray(predicted, dtype=float),
        np.asarray(test_scores, dtype=float),
    )


def compute_partial_series(result_dir):
    time_dirs = sorted(
        int(item.split('_')[1]) for item in os.listdir(result_dir) if item.startswith('Time_')
    )
    rows = []
    for time_id in time_dirs:
        gg_idx, gg_pred, _ = load_fold_arrays(result_dir, time_id, 'GGFC')
        gw_idx, gw_pred, gw_test = load_fold_arrays(result_dir, time_id, 'GWFC')
        ww_idx, ww_pred, ww_test = load_fold_arrays(result_dir, time_id, 'WWFC')

        gg_sort = np.argsort(gg_idx)
        gw_sort = np.argsort(gw_idx)
        ww_sort = np.argsort(ww_idx)

        if not np.array_equal(gg_idx[gg_sort], gw_idx[gw_sort]):
            raise ValueError(f'GG and GW fold indices are misaligned at Time_{time_id}.')
        if not np.array_equal(gg_idx[gg_sort], ww_idx[ww_sort]):
            raise ValueError(f'GG and WW fold indices are misaligned at Time_{time_id}.')

        rows.append(
            {
                'time_id': time_id,
                'GW/GG': partial_corr(gw_pred[gw_sort], gw_test[gw_sort], gg_pred[gg_sort]),
                'WW/GG': partial_corr(ww_pred[ww_sort], ww_test[ww_sort], gg_pred[gg_sort]),
            }
        )
    return pd.DataFrame(rows).sort_values('time_id').reset_index(drop=True)


def summarize_series(dataset, task, target, metric_name, observed_values, null_values):
    observed_values = np.asarray(observed_values, dtype=float)
    null_values = np.asarray(null_values, dtype=float)
    valid_observed = observed_values[~np.isnan(observed_values)]
    valid_null = null_values[~np.isnan(null_values)]

    observed_median = float(np.median(valid_observed)) if valid_observed.size else np.nan
    observed_mean = float(np.mean(valid_observed)) if valid_observed.size else np.nan
    null_mean = float(np.mean(valid_null)) if valid_null.size else np.nan
    null_median = float(np.median(valid_null)) if valid_null.size else np.nan
    null_std = float(np.std(valid_null, ddof=1)) if valid_null.size > 1 else np.nan
    p_value = compute_empirical_right_tail_p(observed_median, valid_null)

    return {
        'dataset': dataset,
        'task': task,
        'target': target,
        'metric_name': metric_name,
        'n_actual_runs': int(valid_observed.size),
        'observed_mean': observed_mean,
        'observed_median': observed_median,
        'n_permutation_runs': int(valid_null.size),
        'permutation_mean': null_mean,
        'permutation_median': null_median,
        'permutation_std': null_std,
        'z_score_vs_permutation': (
            (observed_median - null_mean) / null_std
            if valid_null.size > 1 and null_std > 0
            else np.nan
        ),
        'empirical_p_right_tail': p_value,
        'significance_label': get_significance_label(p_value) if not np.isnan(p_value) else np.nan,
    }


def summarize_target(project_root, dataset, task, target):
    base_dir = os.path.join(project_root, dataset, 'prediction', target, 'V_siblilngs')
    observed_dir = os.path.join(base_dir, 'RegressCovariates_RandomCV')
    permutation_dir = os.path.join(base_dir, 'RegressCovariates_RandomCV_Permutation')

    if not os.path.exists(observed_dir):
        raise FileNotFoundError(f'Observed result directory does not exist: {observed_dir}')
    if not os.path.exists(permutation_dir):
        raise FileNotFoundError(f'Permutation result directory does not exist: {permutation_dir}')

    rows = []
    for feature_name in BASELINE_FEATURES:
        observed_df = load_corr_series(observed_dir, feature_name)
        permutation_df = load_corr_series(permutation_dir, feature_name)
        rows.append(
            summarize_series(
                dataset=dataset,
                task=task,
                target=target,
                metric_name=FEATURE_LABELS[feature_name],
                observed_values=observed_df['corr'].to_numpy(dtype=float),
                null_values=permutation_df['corr'].to_numpy(dtype=float),
            )
        )

    observed_partial_df = compute_partial_series(observed_dir)
    permutation_partial_df = compute_partial_series(permutation_dir)
    for metric_name in ['GW/GG', 'WW/GG']:
        rows.append(
            summarize_series(
                dataset=dataset,
                task=task,
                target=target,
                metric_name=metric_name,
                observed_values=observed_partial_df[metric_name].to_numpy(dtype=float),
                null_values=permutation_partial_df[metric_name].to_numpy(dtype=float),
            )
        )

    target_df = pd.DataFrame(rows)
    target_df['metric_name'] = pd.Categorical(
        target_df['metric_name'],
        categories=METRIC_ORDER,
        ordered=True,
    )
    return target_df.sort_values('metric_name').reset_index(drop=True)


def main():
    args = parse_args()
    targets = args.targets if args.targets else TASK_TARGETS[args.task]
    output_dir = ensure_dir(args.output_root)

    all_results = []
    for target in targets:
        target_df = summarize_target(
            project_root=args.project_root,
            dataset=args.dataset,
            task=args.task,
            target=target,
        )
        all_results.append(target_df)
        print(target_df.to_string(index=False))
        print()

    result_df = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(output_dir, f'{args.dataset}_{args.task}_permutation_significance.csv')
    result_df.to_csv(output_path, index=False)
    print(f'Saved permutation significance summary to {output_path}')


if __name__ == '__main__':
    main()
