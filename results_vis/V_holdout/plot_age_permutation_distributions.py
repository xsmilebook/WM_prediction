import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_age_summary import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASETS,
    compute_partial_series,
    get_rank_order,
    list_time_ids,
    load_corr_mae_arrays,
    load_holdout_score,
    resolve_holdout_dir_name,
)


DEFAULT_RESULTS_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/results/V_holdout/age'
METRIC_CONFIGS = [
    ('GG_corr', 'GG corr', '#2F5D8A', '#BFD6EA'),
    ('GW_corr', 'GW corr', '#4E8B5B', '#C9E2CD'),
    ('WW_corr', 'WW corr', '#A86D1F', '#E9D0AA'),
    ('GW_partial_corr', 'GW partial corr', '#8A3B5D', '#E5C0CF'),
    ('WW_partial_corr', 'WW partial corr', '#5D4A8A', '#CDC3E8'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot age holdout permutation distributions for GG/GW/WW and partial correlations.'
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
        '--pairing_mode',
        choices=['sorted', 'same_time'],
        default='sorted',
        help='Partial-correlation pairing mode. Default: sorted.',
    )
    parser.add_argument(
        '--output_root',
        default=DEFAULT_RESULTS_ROOT,
        help='Directory for output figures and summary CSV. Default: results/V_holdout/age.',
    )
    parser.add_argument(
        '--output_prefix',
        default=None,
        help='Optional output prefix. Default: <holdout_dir_name>_age_permutation_<pairing_mode>.',
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


def compute_partial_series_same_time(base_folder, time_ids):
    partial_r_gw_total = np.full(len(time_ids), np.nan)
    partial_r_ww_total = np.full(len(time_ids), np.nan)

    for idx, run_id in enumerate(time_ids):
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
            continue

        from export_age_summary import partial_corr

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


def load_dataset_metrics(base_folder, permutation_folder, pairing_mode):
    observed_time_ids = list_time_ids(base_folder)
    permutation_time_ids = list_time_ids(permutation_folder)

    corr_actual_gg, _ = load_corr_mae_arrays(base_folder, 'GGFC', observed_time_ids)
    corr_actual_gw, _ = load_corr_mae_arrays(base_folder, 'GWFC', observed_time_ids)
    corr_actual_ww, _ = load_corr_mae_arrays(base_folder, 'WWFC', observed_time_ids)
    corr_perm_gg, _ = load_corr_mae_arrays(permutation_folder, 'GGFC', permutation_time_ids)
    corr_perm_gw, _ = load_corr_mae_arrays(permutation_folder, 'GWFC', permutation_time_ids)
    corr_perm_ww, _ = load_corr_mae_arrays(permutation_folder, 'WWFC', permutation_time_ids)

    if pairing_mode == 'sorted':
        partial_obs_gw, partial_obs_ww = compute_partial_series(
            base_folder,
            observed_time_ids,
            corr_actual_gg,
            corr_actual_gw,
            corr_actual_ww,
        )
        partial_perm_gw, partial_perm_ww = compute_partial_series(
            permutation_folder,
            permutation_time_ids,
            corr_perm_gg,
            corr_perm_gw,
            corr_perm_ww,
        )
    else:
        partial_obs_gw, partial_obs_ww = compute_partial_series_same_time(
            base_folder,
            observed_time_ids,
        )
        partial_perm_gw, partial_perm_ww = compute_partial_series_same_time(
            permutation_folder,
            permutation_time_ids,
        )

    observed = {
        'GG_corr': first_valid_value(corr_actual_gg),
        'GW_corr': first_valid_value(corr_actual_gw),
        'WW_corr': first_valid_value(corr_actual_ww),
        'GW_partial_corr': first_valid_value(partial_obs_gw),
        'WW_partial_corr': first_valid_value(partial_obs_ww),
    }
    permutation = {
        'GG_corr': valid_array(corr_perm_gg),
        'GW_corr': valid_array(corr_perm_gw),
        'WW_corr': valid_array(corr_perm_ww),
        'GW_partial_corr': valid_array(partial_perm_gw),
        'WW_partial_corr': valid_array(partial_perm_ww),
    }
    return observed, permutation


def first_valid_value(values):
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    return float(valid[0]) if valid.size else np.nan


def valid_array(values):
    values = np.asarray(values, dtype=float)
    return values[~np.isnan(values)]


def summarize_distribution(dataset, metric_key, observed_value, permutation_values):
    return {
        'dataset': dataset,
        'metric': metric_key,
        'observed': observed_value,
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

    x_min = min(float(np.min(permutation_values)), float(observed_value))
    x_max = max(float(np.max(permutation_values)), float(observed_value))
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
    ax.axvline(
        observed_value,
        color='#B22222',
        linestyle='-',
        linewidth=2.0,
        alpha=0.95,
    )
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.25)


def plot_distributions(plot_data, output_path, bins, dpi, holdout_dir_name, pairing_mode):
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
            if permutation_values.size:
                ratio = np.mean(permutation_values >= observed_value)
                ax.text(
                    0.98,
                    0.95,
                    f'obs={observed_value:.3f}\np={ratio:.3f}',
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
        f'Age holdout permutation distributions ({holdout_dir_name}, {pairing_mode})',
        fontsize=14,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    holdout_dir_name = resolve_holdout_dir_name(args)
    output_root = ensure_dir(args.output_root)
    output_prefix = (
        args.output_prefix
        if args.output_prefix
        else f'{holdout_dir_name}_age_permutation_{args.pairing_mode}'
    )

    plot_data = []
    summary_rows = []
    for dataset in args.datasets:
        prediction_root = os.path.join(args.data_root, dataset, 'prediction', 'age')
        base_folder = os.path.join(prediction_root, holdout_dir_name, 'RegressCovariates_Holdout')
        permutation_folder = os.path.join(
            prediction_root,
            holdout_dir_name,
            'RegressCovariates_Holdout_Permutation',
        )

        if not os.path.isdir(base_folder):
            raise FileNotFoundError(f'Observed holdout folder not found: {base_folder}')
        if not os.path.isdir(permutation_folder):
            raise FileNotFoundError(f'Permutation folder not found: {permutation_folder}')

        observed, permutation = load_dataset_metrics(
            base_folder,
            permutation_folder,
            args.pairing_mode,
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
        holdout_dir_name,
        args.pairing_mode,
    )
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print(f'Saved figure to {figure_path}')
    print(f'Saved summary CSV to {csv_path}')


if __name__ == '__main__':
    main()
