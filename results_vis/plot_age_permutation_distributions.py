import argparse
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats


DEFAULT_DATA_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data'
DEFAULT_DATASETS = ['EFNY', 'HCPD', 'CCNP', 'PNC']
DEFAULT_RESULTS_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/results/age'
DEFAULT_OBSERVED_DIR = 'RegressCovariates_RandomCV'
DEFAULT_PERMUTATION_DIR = 'RegressCovariates_RandomCV_Permutation'
DEFAULT_NUM_FOLDS = 5

METRIC_CONFIGS = [
    ('GG_corr', 'GG corr', '#2F5D8A', '#BFD6EA'),
    ('GW_corr', 'GW corr', '#4E8B5B', '#C9E2CD'),
    ('WW_corr', 'WW corr', '#A86D1F', '#E9D0AA'),
    ('GW_partial_corr', 'GW partial corr', '#8A3B5D', '#E5C0CF'),
    ('WW_partial_corr', 'WW partial corr', '#5D4A8A', '#CDC3E8'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot age RandomCV permutation distributions for GG/GW/WW and partial correlations.'
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
        help='Age datasets to plot. Default: EFNY HCPD CCNP PNC.',
    )
    parser.add_argument(
        '--observed_dir_name',
        default=DEFAULT_OBSERVED_DIR,
        help='Observed result directory name. Default: RegressCovariates_RandomCV.',
    )
    parser.add_argument(
        '--permutation_dir_name',
        default=DEFAULT_PERMUTATION_DIR,
        help='Permutation result directory name. Default: RegressCovariates_RandomCV_Permutation.',
    )
    parser.add_argument(
        '--pairing_mode',
        choices=['sorted', 'same_time'],
        default='sorted',
        help='Partial-correlation pairing mode. Default: sorted.',
    )
    parser.add_argument(
        '--num_folds',
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help='Number of outer folds stored in each Time_i folder. Default: 5.',
    )
    parser.add_argument(
        '--output_root',
        default=DEFAULT_RESULTS_ROOT,
        help='Directory for output figures and summary CSV. Default: results/age.',
    )
    parser.add_argument(
        '--output_prefix',
        default=None,
        help='Optional output prefix. Default: age_randomcv_permutation_<pairing_mode>.',
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=30,
        help='Histogram bin count. Default: 30.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI. Default: 300.',
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


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


def list_time_ids(base_folder):
    return sorted(
        int(item.split('_')[1])
        for item in os.listdir(base_folder)
        if item.startswith('Time_') and os.path.isdir(os.path.join(base_folder, item))
    )


def load_corr_mae_arrays(base_folder, fc_type, time_ids):
    corr_values = np.full(len(time_ids), np.nan)
    mae_values = np.full(len(time_ids), np.nan)
    for idx, run_id in enumerate(time_ids):
        res_file = os.path.join(base_folder, f'Time_{run_id}', fc_type, 'Res_NFold.mat')
        if not os.path.isfile(res_file):
            continue
        try:
            mat = sio.loadmat(res_file)
            corr_values[idx] = float(mat['Mean_Corr'].item())
            mae_values[idx] = float(mat['Mean_MAE'].item())
        except Exception:
            continue
    return corr_values, mae_values


def load_all_folds(base_folder, run_id, fc_type, num_folds):
    idx, pred, test = [], [], []
    for fold_id in range(num_folds):
        score_file = os.path.join(
            base_folder,
            f'Time_{run_id}',
            fc_type,
            f'Fold_{fold_id}_Score.mat',
        )
        if not os.path.isfile(score_file):
            return None, None, None
        try:
            mat = sio.loadmat(score_file)
            idx.extend(mat['Index'].flatten())
            pred.extend(mat['Predict_Score'].flatten())
            test.extend(mat['Test_Score'].flatten())
        except Exception:
            return None, None, None
    return np.asarray(idx), np.asarray(pred), np.asarray(test)


def get_rank_order(corr_values):
    corr_values = np.asarray(corr_values, dtype=float)
    temp_values = corr_values.copy()
    temp_values[np.isnan(temp_values)] = -np.inf
    return np.argsort(-temp_values)


def compute_partial_series_same_time(base_folder, time_ids, corr_gg, corr_gw, corr_ww, num_folds):
    partial_r_gw_total = np.full(len(time_ids), np.nan)
    partial_r_ww_total = np.full(len(time_ids), np.nan)

    for idx, run_id in enumerate(time_ids):
        if np.isnan(corr_gg[idx]) or np.isnan(corr_gw[idx]) or np.isnan(corr_ww[idx]):
            continue

        gg_idx, gg_pred, _ = load_all_folds(base_folder, run_id, 'GGFC', num_folds)
        gw_idx, gw_pred, gw_test = load_all_folds(base_folder, run_id, 'GWFC', num_folds)
        ww_idx, ww_pred, ww_test = load_all_folds(base_folder, run_id, 'WWFC', num_folds)

        if gg_idx is None or gw_idx is None or ww_idx is None:
            continue

        sort_gg = np.argsort(gg_idx)
        sort_gw = np.argsort(gw_idx)
        sort_ww = np.argsort(ww_idx)

        if not (
            np.array_equal(gg_idx[sort_gg], gw_idx[sort_gw])
            and np.array_equal(gg_idx[sort_gg], ww_idx[sort_ww])
        ):
            warnings.warn(f'Index mismatch at Time_{run_id}. Skipping.')
            continue

        partial_r_gw_total[idx] = partial_corr(
            gw_pred[sort_gw],
            gw_test[sort_gw].astype(float),
            gg_pred[sort_gg],
        )
        partial_r_ww_total[idx] = partial_corr(
            ww_pred[sort_ww],
            ww_test[sort_ww].astype(float),
            gg_pred[sort_gg],
        )

    return partial_r_gw_total, partial_r_ww_total


def compute_partial_series_sorted(base_folder, time_ids, corr_gg, corr_gw, corr_ww, num_folds):
    partial_r_gw_total = np.full(len(time_ids), np.nan)
    partial_r_ww_total = np.full(len(time_ids), np.nan)
    rank_gg = get_rank_order(corr_gg)
    rank_gw = get_rank_order(corr_gw)
    rank_ww = get_rank_order(corr_ww)

    for rank_idx in range(len(time_ids)):
        gg_run_idx = rank_gg[rank_idx]
        gw_run_idx = rank_gw[rank_idx]
        ww_run_idx = rank_ww[rank_idx]

        if (
            np.isnan(corr_gg[gg_run_idx])
            or np.isnan(corr_gw[gw_run_idx])
            or np.isnan(corr_ww[ww_run_idx])
        ):
            continue

        gg_idx, gg_pred, _ = load_all_folds(base_folder, time_ids[gg_run_idx], 'GGFC', num_folds)
        gw_idx, gw_pred, gw_test = load_all_folds(base_folder, time_ids[gw_run_idx], 'GWFC', num_folds)
        ww_idx, ww_pred, ww_test = load_all_folds(base_folder, time_ids[ww_run_idx], 'WWFC', num_folds)

        if gg_idx is None or gw_idx is None or ww_idx is None:
            continue

        sort_gg = np.argsort(gg_idx)
        sort_gw = np.argsort(gw_idx)
        sort_ww = np.argsort(ww_idx)

        if not (
            np.array_equal(gg_idx[sort_gg], gw_idx[sort_gw])
            and np.array_equal(gg_idx[sort_gg], ww_idx[sort_ww])
        ):
            warnings.warn(
                'Index mismatch at rank {} (GG=Time_{}, GW=Time_{}, WW=Time_{}). Skipping.'.format(
                    rank_idx,
                    time_ids[gg_run_idx],
                    time_ids[gw_run_idx],
                    time_ids[ww_run_idx],
                )
            )
            continue

        partial_r_gw_total[rank_idx] = partial_corr(
            gw_pred[sort_gw],
            gw_test[sort_gw].astype(float),
            gg_pred[sort_gg],
        )
        partial_r_ww_total[rank_idx] = partial_corr(
            ww_pred[sort_ww],
            ww_test[sort_ww].astype(float),
            gg_pred[sort_gg],
        )

    return partial_r_gw_total, partial_r_ww_total


def first_valid_value(values):
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    return float(valid[0]) if valid.size else np.nan


def median_valid_value(values):
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    return float(np.median(valid)) if valid.size else np.nan


def valid_array(values):
    values = np.asarray(values, dtype=float)
    return values[~np.isnan(values)]


def load_dataset_metrics(base_folder, permutation_folder, pairing_mode, num_folds):
    observed_time_ids = list_time_ids(base_folder)
    permutation_time_ids = list_time_ids(permutation_folder)

    corr_actual_gg, _ = load_corr_mae_arrays(base_folder, 'GGFC', observed_time_ids)
    corr_actual_gw, _ = load_corr_mae_arrays(base_folder, 'GWFC', observed_time_ids)
    corr_actual_ww, _ = load_corr_mae_arrays(base_folder, 'WWFC', observed_time_ids)
    corr_perm_gg, _ = load_corr_mae_arrays(permutation_folder, 'GGFC', permutation_time_ids)
    corr_perm_gw, _ = load_corr_mae_arrays(permutation_folder, 'GWFC', permutation_time_ids)
    corr_perm_ww, _ = load_corr_mae_arrays(permutation_folder, 'WWFC', permutation_time_ids)

    if pairing_mode == 'same_time':
        partial_obs_gw, partial_obs_ww = compute_partial_series_same_time(
            base_folder,
            observed_time_ids,
            corr_actual_gg,
            corr_actual_gw,
            corr_actual_ww,
            num_folds,
        )
        partial_perm_gw, partial_perm_ww = compute_partial_series_same_time(
            permutation_folder,
            permutation_time_ids,
            corr_perm_gg,
            corr_perm_gw,
            corr_perm_ww,
            num_folds,
        )
    else:
        partial_obs_gw, partial_obs_ww = compute_partial_series_sorted(
            base_folder,
            observed_time_ids,
            corr_actual_gg,
            corr_actual_gw,
            corr_actual_ww,
            num_folds,
        )
        partial_perm_gw, partial_perm_ww = compute_partial_series_sorted(
            permutation_folder,
            permutation_time_ids,
            corr_perm_gg,
            corr_perm_gw,
            corr_perm_ww,
            num_folds,
        )

    observed = {
        'GG_corr': median_valid_value(corr_actual_gg),
        'GW_corr': median_valid_value(corr_actual_gw),
        'WW_corr': median_valid_value(corr_actual_ww),
        'GW_partial_corr': median_valid_value(partial_obs_gw),
        'WW_partial_corr': median_valid_value(partial_obs_ww),
    }
    permutation = {
        'GG_corr': valid_array(corr_perm_gg),
        'GW_corr': valid_array(corr_perm_gw),
        'WW_corr': valid_array(corr_perm_ww),
        'GW_partial_corr': valid_array(partial_perm_gw),
        'WW_partial_corr': valid_array(partial_perm_ww),
    }
    observed_distribution = {
        'GG_corr': valid_array(corr_actual_gg),
        'GW_corr': valid_array(corr_actual_gw),
        'WW_corr': valid_array(corr_actual_ww),
        'GW_partial_corr': valid_array(partial_obs_gw),
        'WW_partial_corr': valid_array(partial_obs_ww),
    }
    return observed, observed_distribution, permutation


def summarize_distribution(dataset, metric_key, observed_value, observed_values, permutation_values):
    return {
        'dataset': dataset,
        'metric': metric_key,
        'observed_median': observed_value,
        'n_observed': int(observed_values.size),
        'observed_mean': float(np.mean(observed_values)) if observed_values.size else np.nan,
        'observed_std': float(np.std(observed_values, ddof=1)) if observed_values.size > 1 else np.nan,
        'n_permutation': int(permutation_values.size),
        'perm_mean': float(np.mean(permutation_values)) if permutation_values.size else np.nan,
        'perm_std': float(np.std(permutation_values, ddof=1)) if permutation_values.size > 1 else np.nan,
        'perm_q05': float(np.quantile(permutation_values, 0.05)) if permutation_values.size else np.nan,
        'perm_q25': float(np.quantile(permutation_values, 0.25)) if permutation_values.size else np.nan,
        'perm_median': float(np.quantile(permutation_values, 0.50)) if permutation_values.size else np.nan,
        'perm_q75': float(np.quantile(permutation_values, 0.75)) if permutation_values.size else np.nan,
        'perm_q95': float(np.quantile(permutation_values, 0.95)) if permutation_values.size else np.nan,
        'perm_max': float(np.max(permutation_values)) if permutation_values.size else np.nan,
        'perm_gt_observed_ratio': float(np.mean(permutation_values >= observed_value))
        if permutation_values.size and not np.isnan(observed_value)
        else np.nan,
    }


def add_hist_panel(ax, permutation_values, observed_value, title, line_color, fill_color, bins):
    if permutation_values.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No valid permutation values', ha='center', va='center')
        ax.set_axis_off()
        return

    x_candidates = [float(np.min(permutation_values)), float(np.max(permutation_values))]
    if not np.isnan(observed_value):
        x_candidates.extend([float(observed_value), float(observed_value)])
    x_min = min(x_candidates)
    x_max = max(x_candidates)
    if x_min == x_max:
        padding = max(abs(x_min), 1.0) * 0.05
        x_min -= padding
        x_max += padding

    bin_edges = np.linspace(x_min, x_max, bins + 1)
    ax.hist(
        permutation_values,
        bins=bin_edges,
        density=True,
        alpha=0.75,
        color=fill_color,
        edgecolor='white',
        linewidth=0.6,
    )
    perm_median = float(np.median(permutation_values))
    ax.axvline(
        perm_median,
        color=line_color,
        linestyle='--',
        linewidth=1.8,
        alpha=0.9,
    )
    if not np.isnan(observed_value):
        ax.axvline(
            observed_value,
            color='#B22222',
            linestyle='-',
            linewidth=2.0,
            alpha=0.95,
        )
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.25)


def plot_distributions(plot_data, output_path, bins, dpi, pairing_mode):
    datasets = [item['dataset'] for item in plot_data]
    fig, axes = plt.subplots(
        len(datasets),
        len(METRIC_CONFIGS),
        figsize=(18, 3.3 * len(datasets)),
        constrained_layout=True,
    )

    if len(datasets) == 1:
        axes = np.array([axes])

    for row_idx, dataset_data in enumerate(plot_data):
        for col_idx, (metric_key, title, line_color, fill_color) in enumerate(METRIC_CONFIGS):
            ax = axes[row_idx, col_idx]
            observed_value = dataset_data['observed'][metric_key]
            permutation_values = dataset_data['permutation'][metric_key]
            full_title = f"{dataset_data['dataset']} {title}"
            add_hist_panel(
                ax,
                permutation_values,
                observed_value,
                full_title,
                line_color,
                fill_color,
                bins,
            )
            if permutation_values.size and not np.isnan(observed_value):
                ratio = np.mean(permutation_values >= observed_value)
                ax.text(
                    0.98,
                    0.95,
                    f'obs med={observed_value:.3f}\np={ratio:.3f}',
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
                )
            if row_idx == len(datasets) - 1:
                ax.set_xlabel('Correlation')
            if col_idx == 0:
                ax.set_ylabel('Density')

    fig.suptitle(
        f'Age RandomCV permutation distributions ({pairing_mode})',
        fontsize=14,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    output_prefix = (
        args.output_prefix
        if args.output_prefix
        else f'age_randomcv_permutation_{args.pairing_mode}'
    )

    plot_data = []
    summary_rows = []
    for dataset in args.datasets:
        prediction_root = os.path.join(args.data_root, dataset, 'prediction', 'age')
        base_folder = os.path.join(prediction_root, args.observed_dir_name)
        permutation_folder = os.path.join(prediction_root, args.permutation_dir_name)

        if not os.path.isdir(base_folder):
            raise FileNotFoundError(f'Observed folder not found: {base_folder}')
        if not os.path.isdir(permutation_folder):
            raise FileNotFoundError(f'Permutation folder not found: {permutation_folder}')

        observed, observed_distribution, permutation = load_dataset_metrics(
            base_folder,
            permutation_folder,
            args.pairing_mode,
            args.num_folds,
        )
        plot_data.append(
            {
                'dataset': dataset,
                'observed': observed,
                'permutation': permutation,
            }
        )
        for metric_key, _, _, _ in METRIC_CONFIGS:
            summary_rows.append(
                summarize_distribution(
                    dataset,
                    metric_key,
                    observed[metric_key],
                    observed_distribution[metric_key],
                    permutation[metric_key],
                )
            )

    figure_path = os.path.join(output_root, f'{output_prefix}.tiff')
    csv_path = os.path.join(output_root, f'{output_prefix}_summary.csv')

    plot_distributions(
        plot_data,
        figure_path,
        args.bins,
        args.dpi,
        args.pairing_mode,
    )
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print(f'Saved figure to {figure_path}')
    print(f'Saved summary CSV to {csv_path}')


if __name__ == '__main__':
    main()
