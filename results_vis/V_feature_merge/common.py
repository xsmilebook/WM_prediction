import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

BASELINE_FEATURES = ['GGFC', 'GWFC', 'WWFC']
MERGED_FEATURES = [
    'GG_GW_MergedFC',
    'GG_WW_MergedFC',
    'GW_WW_MergedFC',
    'GG_GW_WW_MergedFC',
]
MERGED_TO_CHILDREN = {
    'GG_GW_MergedFC': ['GGFC', 'GWFC'],
    'GG_WW_MergedFC': ['GGFC', 'WWFC'],
    'GW_WW_MergedFC': ['GWFC', 'WWFC'],
    'GG_GW_WW_MergedFC': ['GGFC', 'GWFC', 'WWFC'],
}
TASK_TARGETS = {
    'age': ['age'],
    'cognition': [
        'nihtbx_cryst_uncorrected',
        'nihtbx_fluidcomp_uncorrected',
        'nihtbx_totalcomp_uncorrected',
    ],
    'pfactor': ['General', 'Ext', 'ADHD', 'Int'],
}
DEFAULT_PROJECT_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data'
DEFAULT_REPO_ROOT = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction'
DEFAULT_RESULTS_ROOT = os.path.join(DEFAULT_REPO_ROOT, 'results', 'V_feature_merge')
FEATURE_DISPLAY_MAP = {
    'GGFC': 'GG',
    'GWFC': 'GW',
    'WWFC': 'WW',
    'GG_GW_MergedFC': 'GG+GW',
    'GW_WW_MergedFC': 'GW+WW',
    'GG_WW_MergedFC': 'GG+WW',
    'GG_GW_WW_MergedFC': 'GG+GW+WW',
}
FEATURE_PLOT_ORDER = [
    'GGFC',
    'GWFC',
    'WWFC',
    'GG_GW_MergedFC',
    'GW_WW_MergedFC',
    'GG_WW_MergedFC',
    'GG_GW_WW_MergedFC',
]
DEFAULT_PFACTOR_SIGNIFICANCE_TARGETS = ['General', 'Ext', 'ADHD']


def parse_common_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', required=True, help='Dataset name, e.g. HCPD or ABCD.')
    parser.add_argument(
        '--task',
        required=True,
        choices=sorted(TASK_TARGETS.keys()),
        help='Prediction task family.',
    )
    parser.add_argument(
        '--project_root',
        default=DEFAULT_PROJECT_ROOT,
        help='Root directory containing per-dataset prediction folders.',
    )
    parser.add_argument(
        '--targets',
        nargs='*',
        default=None,
        help='Optional explicit target list. Defaults depend on --task.',
    )
    parser.add_argument(
        '--output_root',
        default=None,
        help='Optional custom output root. Defaults to each target V_feature_merge/statistics folder.',
    )
    return parser.parse_args()


def extract_time_index(path):
    feature_dir = os.path.dirname(path)
    time_dir = os.path.dirname(feature_dir)
    time_token = os.path.basename(time_dir)
    return int(time_token.split('_')[1])


def load_corr_series(result_dir, feature_name):
    pattern = os.path.join(result_dir, 'Time_*', feature_name, 'Res_NFold.mat')
    result_paths = sorted(glob.glob(pattern), key=extract_time_index)
    if not result_paths:
        raise FileNotFoundError(f'No result files found for {feature_name}: {pattern}')

    rows = []
    for result_path in result_paths:
        result_mat = sio.loadmat(result_path)
        rows.append(
            {
                'time_id': extract_time_index(result_path),
                'corr': float(result_mat['Mean_Corr'].item()),
            }
        )
    return pd.DataFrame(rows).sort_values('time_id').reset_index(drop=True)


def resolve_targets(task, targets):
    return targets if targets else TASK_TARGETS[task]


def get_target_paths(project_root, dataset, target):
    target_dir = os.path.join(project_root, dataset, 'prediction', target)
    baseline_dir = os.path.join(target_dir, 'RegressCovariates_RandomCV')
    merged_dir = os.path.join(target_dir, 'V_feature_merge', 'RegressCovariates_RandomCV')
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f'Target directory does not exist: {target_dir}')
    if not os.path.exists(baseline_dir):
        raise FileNotFoundError(f'Baseline directory does not exist: {baseline_dir}')
    if not os.path.exists(merged_dir):
        raise FileNotFoundError(f'Merged directory does not exist: {merged_dir}')
    return target_dir, baseline_dir, merged_dir


def get_output_dir(target_dir, output_root):
    if output_root:
        return os.path.join(output_root, os.path.basename(target_dir))
    return os.path.join(target_dir, 'V_feature_merge', 'statistics')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_target_feature_data(project_root, dataset, target):
    target_dir, baseline_dir, merged_dir = get_target_paths(project_root, dataset, target)
    baseline_data = {
        feature_name: load_corr_series(baseline_dir, feature_name)
        for feature_name in BASELINE_FEATURES
    }
    merged_data = {
        feature_name: load_corr_series(merged_dir, feature_name)
        for feature_name in MERGED_FEATURES
    }
    return target_dir, baseline_data, merged_data


def get_significance_label(p_value):
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def add_significance_bar(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h, label, ha='center', va='bottom', fontsize=11, fontweight='bold')


def plot_boxplot_with_points(
    series_list,
    labels,
    colors,
    title,
    ylabel,
    output_path,
    significance=None,
    connect_pairs=False,
):
    fig, ax = plt.subplots(figsize=(7, 6))
    positions = np.arange(1, len(series_list) + 1)
    bp = ax.boxplot(
        series_list,
        positions=positions,
        patch_artist=True,
        widths=0.55,
        medianprops={'color': 'black', 'linewidth': 1.4},
        whiskerprops={'color': '#4d4d4d'},
        capprops={'color': '#4d4d4d'},
        boxprops={'edgecolor': '#4d4d4d'},
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(series_list, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=18,
            alpha=0.55,
            color='#2f2f2f',
            linewidths=0,
            zorder=3,
        )

    if connect_pairs and len(series_list) == 2 and len(series_list[0]) == len(series_list[1]):
        for left_value, right_value in zip(series_list[0], series_list[1]):
            ax.plot(
                [positions[0], positions[1]],
                [left_value, right_value],
                color='#b8b8b8',
                alpha=0.45,
                linewidth=0.8,
                zorder=1,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    all_values = np.concatenate([np.asarray(values) for values in series_list])
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_range = y_max - y_min
    if y_range == 0:
        y_range = max(abs(y_max), 1.0) * 0.1 if y_max != 0 else 0.1
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.22 * y_range)

    if significance is not None:
        add_significance_bar(
            ax,
            1,
            2,
            y_max + 0.04 * y_range,
            0.03 * y_range,
            significance,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def format_float(value):
    if pd.isna(value):
        return 'nan'
    return f'{value:.6g}'
