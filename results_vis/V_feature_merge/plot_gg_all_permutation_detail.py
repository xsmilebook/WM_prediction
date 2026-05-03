import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DEFAULT_RESULTS_ROOT, ensure_dir

OBSERVED_SOURCE = 'observed_random_cv'
PERMUTATION_SOURCE = 'shared_target_permutation'
PANEL_CONFIGS = [
    ('gg_corr', 'GG corr', '#4C78A8', '#9EC1E6'),
    ('all_corr', 'GG+GW+WW corr', '#F58518', '#F8CFA2'),
    ('delta_corr', 'All-GG corr', '#54A24B', '#B9E3B2'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot observed and shared-target-permutation distributions from a GG/All detail CSV.',
    )
    parser.add_argument(
        '--input_csv',
        required=True,
        help='Path to <dataset>_<task>_gg_all_permutation_detail.csv',
    )
    parser.add_argument(
        '--output_path',
        default=None,
        help='Optional figure output path. Default: alongside the input CSV with suffix _distribution.tiff',
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


def load_input_df(input_csv):
    df = pd.read_csv(input_csv)
    required_columns = {
        'dataset',
        'target',
        'source',
        'gg_corr',
        'all_corr',
        'delta_corr',
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ', '.join(sorted(missing_columns))
        raise ValueError(f'Missing required columns in input CSV: {missing_str}')

    observed_df = df.loc[df['source'] == OBSERVED_SOURCE].copy()
    permutation_df = df.loc[df['source'] == PERMUTATION_SOURCE].copy()
    if observed_df.empty:
        raise ValueError('No observed_random_cv rows found in input CSV.')
    if permutation_df.empty:
        raise ValueError('No shared_target_permutation rows found in input CSV.')

    dataset_values = observed_df['dataset'].dropna().unique().tolist()
    target_values = observed_df['target'].dropna().unique().tolist()
    dataset_label = dataset_values[0] if dataset_values else 'unknown_dataset'
    target_label = target_values[0] if target_values else 'unknown_target'
    return observed_df, permutation_df, dataset_label, target_label


def add_distribution_panel(ax, observed_values, permutation_values, value_col, title, line_color, fill_color, bins):
    combined_min = min(float(np.min(observed_values)), float(np.min(permutation_values)))
    combined_max = max(float(np.max(observed_values)), float(np.max(permutation_values)))
    if combined_min == combined_max:
        padding = max(abs(combined_min), 1.0) * 0.05
        combined_min -= padding
        combined_max += padding

    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    ax.hist(
        permutation_values,
        bins=bin_edges,
        density=True,
        alpha=0.55,
        color=fill_color,
        edgecolor='white',
        linewidth=0.6,
        label=f'Permutation (n={len(permutation_values)})',
    )
    ax.hist(
        observed_values,
        bins=bin_edges,
        density=True,
        alpha=0.45,
        color=line_color,
        edgecolor='white',
        linewidth=0.6,
        label=f'Observed (n={len(observed_values)})',
    )

    observed_median = float(np.median(observed_values))
    permutation_median = float(np.median(permutation_values))
    ax.axvline(
        observed_median,
        color=line_color,
        linestyle='-',
        linewidth=2.0,
        label=f'Observed median = {observed_median:.4f}',
    )
    ax.axvline(
        permutation_median,
        color=fill_color,
        linestyle='--',
        linewidth=2.0,
        label=f'Permutation median = {permutation_median:.4f}',
    )

    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel('Density')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(frameon=False, fontsize=8)


def main():
    args = parse_args()
    observed_df, permutation_df, dataset_label, target_label = load_input_df(args.input_csv)

    if args.output_path:
        output_path = args.output_path
    else:
        input_dir = os.path.dirname(args.input_csv)
        input_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        output_path = os.path.join(input_dir, f'{input_name}_distribution.tiff')
    ensure_dir(os.path.dirname(output_path) if os.path.dirname(output_path) else DEFAULT_RESULTS_ROOT)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    for ax, (column_name, title, line_color, fill_color) in zip(axes, PANEL_CONFIGS):
        observed_values = observed_df[column_name].dropna().to_numpy(dtype=float)
        permutation_values = permutation_df[column_name].dropna().to_numpy(dtype=float)
        add_distribution_panel(
            ax,
            observed_values,
            permutation_values,
            column_name,
            title,
            line_color,
            fill_color,
            args.bins,
        )

    fig.suptitle(f'{dataset_label} {target_label}: observed vs permutation distributions', fontsize=13)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved distribution figure to {output_path}')


if __name__ == '__main__':
    main()
