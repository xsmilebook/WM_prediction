import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


DEFAULT_DATA_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data'
DEFAULT_DATASETS = ['EFNY', 'HCPD', 'CCNP', 'PNC']
DEFAULT_RESULTS_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/results/age'
DEFAULT_OBSERVED_DIR = 'RegressCovariates_RandomCV'
DEFAULT_PERMUTATION_DIR = 'RegressCovariates_RandomCV_Permutation'
DEFAULT_NUM_FOLDS = 5
FC_CONFIGS = [
    ('GGFC', 'GG corr', '#2F5D8A', '#BFD6EA'),
    ('GWFC', 'GW corr', '#4E8B5B', '#C9E2CD'),
    ('WWFC', 'WW corr', '#A86D1F', '#E9D0AA'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot permutation distributions of fold-level correlations for age RandomCV results.'
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
        default='age_randomcv_fold_permutation',
        help='Output prefix. One figure per dataset plus one summary CSV will be written.',
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


def list_time_ids(base_folder):
    return sorted(
        int(item.split('_')[1])
        for item in os.listdir(base_folder)
        if item.startswith('Time_') and os.path.isdir(os.path.join(base_folder, item))
    )


def load_fold_corr_matrix(base_folder, fc_type, time_ids, num_folds):
    fold_corr = np.full((len(time_ids), num_folds), np.nan)
    for time_idx, run_id in enumerate(time_ids):
        for fold_idx in range(num_folds):
            score_file = os.path.join(
                base_folder,
                f'Time_{run_id}',
                fc_type,
                f'Fold_{fold_idx}_Score.mat',
            )
            if not os.path.isfile(score_file):
                continue
            try:
                mat = sio.loadmat(score_file)
                fold_corr[time_idx, fold_idx] = float(mat['Corr'].item())
            except Exception:
                continue
    return fold_corr


def valid_array(values):
    values = np.asarray(values, dtype=float)
    return values[~np.isnan(values)]


def summarize_fold_distribution(dataset, fc_type, fold_idx, observed_values, permutation_values):
    observed_values = valid_array(observed_values)
    permutation_values = valid_array(permutation_values)
    observed_median = float(np.median(observed_values)) if observed_values.size else np.nan
    return {
        'dataset': dataset,
        'fc_type': fc_type,
        'fold': int(fold_idx),
        'n_observed': int(observed_values.size),
        'observed_mean': float(np.mean(observed_values)) if observed_values.size else np.nan,
        'observed_median': observed_median,
        'observed_std': float(np.std(observed_values, ddof=1)) if observed_values.size > 1 else np.nan,
        'n_permutation': int(permutation_values.size),
        'perm_mean': float(np.mean(permutation_values)) if permutation_values.size else np.nan,
        'perm_median': float(np.median(permutation_values)) if permutation_values.size else np.nan,
        'perm_std': float(np.std(permutation_values, ddof=1)) if permutation_values.size > 1 else np.nan,
        'perm_q05': float(np.quantile(permutation_values, 0.05)) if permutation_values.size else np.nan,
        'perm_q25': float(np.quantile(permutation_values, 0.25)) if permutation_values.size else np.nan,
        'perm_q75': float(np.quantile(permutation_values, 0.75)) if permutation_values.size else np.nan,
        'perm_q95': float(np.quantile(permutation_values, 0.95)) if permutation_values.size else np.nan,
        'perm_gt_observed_ratio': float(np.mean(permutation_values >= observed_median))
        if permutation_values.size and not np.isnan(observed_median)
        else np.nan,
    }


def add_hist_panel(ax, permutation_values, observed_value, title, line_color, fill_color, bins):
    permutation_values = valid_array(permutation_values)
    if permutation_values.size == 0:
        ax.set_title(title, fontsize=10)
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
    ax.axvline(
        float(np.median(permutation_values)),
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


def plot_dataset_fold_distributions(dataset, observed_mats, permutation_mats, output_path, bins, dpi):
    num_rows = len(FC_CONFIGS)
    num_cols = observed_mats[FC_CONFIGS[0][0]].shape[1]
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.0 * num_cols, 2.8 * num_rows),
        constrained_layout=True,
    )

    for row_idx, (fc_type, row_label, line_color, fill_color) in enumerate(FC_CONFIGS):
        observed_mat = observed_mats[fc_type]
        permutation_mat = permutation_mats[fc_type]
        for fold_idx in range(num_cols):
            ax = axes[row_idx, fold_idx]
            observed_values = valid_array(observed_mat[:, fold_idx])
            permutation_values = valid_array(permutation_mat[:, fold_idx])
            observed_median = float(np.median(observed_values)) if observed_values.size else np.nan
            add_hist_panel(
                ax,
                permutation_values,
                observed_median,
                f'Fold {fold_idx}',
                line_color,
                fill_color,
                bins,
            )
            if permutation_values.size and not np.isnan(observed_median):
                ratio = np.mean(permutation_values >= observed_median)
                ax.text(
                    0.98,
                    0.95,
                    f'obs med={observed_median:.3f}\np={ratio:.3f}',
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
                )
            if fold_idx == 0:
                ax.set_ylabel(f'{row_label}\nDensity')
            if row_idx == num_rows - 1:
                ax.set_xlabel('Correlation')

    fig.suptitle(f'{dataset} age permutation fold distributions', fontsize=14)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    summary_rows = []

    for dataset in args.datasets:
        prediction_root = os.path.join(args.data_root, dataset, 'prediction', 'age')
        observed_folder = os.path.join(prediction_root, args.observed_dir_name)
        permutation_folder = os.path.join(prediction_root, args.permutation_dir_name)

        if not os.path.isdir(observed_folder):
            raise FileNotFoundError(f'Observed folder not found: {observed_folder}')
        if not os.path.isdir(permutation_folder):
            raise FileNotFoundError(f'Permutation folder not found: {permutation_folder}')

        observed_time_ids = list_time_ids(observed_folder)
        permutation_time_ids = list_time_ids(permutation_folder)

        observed_mats = {}
        permutation_mats = {}
        for fc_type, _, _, _ in FC_CONFIGS:
            observed_mats[fc_type] = load_fold_corr_matrix(
                observed_folder,
                fc_type,
                observed_time_ids,
                args.num_folds,
            )
            permutation_mats[fc_type] = load_fold_corr_matrix(
                permutation_folder,
                fc_type,
                permutation_time_ids,
                args.num_folds,
            )
            for fold_idx in range(args.num_folds):
                summary_rows.append(
                    summarize_fold_distribution(
                        dataset,
                        fc_type,
                        fold_idx,
                        observed_mats[fc_type][:, fold_idx],
                        permutation_mats[fc_type][:, fold_idx],
                    )
                )

        figure_path = os.path.join(output_root, f'{args.output_prefix}_{dataset}.tiff')
        plot_dataset_fold_distributions(
            dataset,
            observed_mats,
            permutation_mats,
            figure_path,
            args.bins,
            args.dpi,
        )
        print(f'Saved figure to {figure_path}')

    csv_path = os.path.join(output_root, f'{args.output_prefix}_summary.csv')
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f'Saved summary CSV to {csv_path}')


if __name__ == '__main__':
    main()
