import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    FEATURE_DISPLAY_MAP,
    FEATURE_PLOT_ORDER,
    ensure_dir,
    load_target_feature_data,
)

AGE_DATASETS = ['HCPD', 'CCNP', 'EFNY', 'PNC']
ABCD_DATASET = 'ABCD'
TARGET_TITLE_MAP = {
    'age': 'Age',
    'nihtbx_cryst_uncorrected': 'Crystal',
    'nihtbx_fluidcomp_uncorrected': 'Fluid',
    'nihtbx_totalcomp_uncorrected': 'Total',
    'General': 'General',
    'Ext': 'External',
    'ADHD': 'ADHD',
    'Int': 'Int',
}
FEATURE_COLOR_MAP = {
    'GGFC': '#4C78A8',
    'GWFC': '#72B7B2',
    'WWFC': '#54A24B',
    'GG_GW_MergedFC': '#F58518',
    'GW_WW_MergedFC': '#E45756',
    'GG_WW_MergedFC': '#B279A2',
    'GG_GW_WW_MergedFC': '#9D755D',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot half-violin and boxplots for feature-merge median correlations.',
    )
    parser.add_argument(
        '--project_root',
        default=DEFAULT_PROJECT_ROOT,
        help='Root directory containing per-dataset prediction folders.',
    )
    parser.add_argument(
        '--output_root',
        default=DEFAULT_RESULTS_ROOT,
        help='Directory for exported figures and summary CSV.',
    )
    parser.add_argument(
        '--age_datasets',
        nargs='*',
        default=AGE_DATASETS,
        help='Datasets to plot for the age target.',
    )
    parser.add_argument(
        '--abcd_dataset',
        default=ABCD_DATASET,
        help='Dataset name used for cognition and pfactor plots.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI.',
    )
    parser.add_argument(
        '--skip_age',
        action='store_true',
        help='Skip age plots.',
    )
    parser.add_argument(
        '--skip_abcd',
        action='store_true',
        help='Skip ABCD cognition and pfactor plots.',
    )
    return parser.parse_args()


def get_plot_specifications(args):
    specs = []
    if not args.skip_age:
        for dataset in args.age_datasets:
            specs.append(
                {
                    'dataset': dataset,
                    'task': 'age',
                    'target': 'age',
                    'output_dir': os.path.join(args.output_root, 'age'),
                    'filename': f'{dataset}_age_median_corr_half_violin_box.png',
                }
            )

    if not args.skip_abcd:
        cognition_targets = [
            'nihtbx_cryst_uncorrected',
            'nihtbx_fluidcomp_uncorrected',
            'nihtbx_totalcomp_uncorrected',
        ]
        pfactor_targets = ['General', 'Ext', 'ADHD', 'Int']

        for target in cognition_targets:
            specs.append(
                {
                    'dataset': args.abcd_dataset,
                    'task': 'cognition',
                    'target': target,
                    'output_dir': os.path.join(args.output_root, args.abcd_dataset, 'cognition'),
                    'filename': f'{target}_median_corr_half_violin_box.png',
                }
            )

        for target in pfactor_targets:
            specs.append(
                {
                    'dataset': args.abcd_dataset,
                    'task': 'pfactor',
                    'target': target,
                    'output_dir': os.path.join(args.output_root, args.abcd_dataset, 'pfactor'),
                    'filename': f'{target}_median_corr_half_violin_box.png',
                }
            )

    return specs


def build_feature_dataframe(project_root, dataset, target, task):
    _, baseline_data, merged_data = load_target_feature_data(project_root, dataset, target)
    feature_sources = {}
    feature_sources.update(baseline_data)
    feature_sources.update(merged_data)

    rows = []
    for feature_name in FEATURE_PLOT_ORDER:
        feature_df = feature_sources[feature_name]
        for _, row in feature_df.iterrows():
            rows.append(
                {
                    'dataset': dataset,
                    'task': task,
                    'target': target,
                    'time_id': int(row['time_id']),
                    'feature_name': feature_name,
                    'feature_label': FEATURE_DISPLAY_MAP[feature_name],
                    'corr': float(row['corr']),
                }
            )
    return pd.DataFrame(rows)


def summarize_feature_dataframe(plot_df):
    summary_df = (
        plot_df.groupby(['dataset', 'task', 'target', 'feature_name', 'feature_label'], as_index=False)
        .agg(
            n_runs=('corr', 'size'),
            median_corr=('corr', 'median'),
            mean_corr=('corr', 'mean'),
            std_corr=('corr', 'std'),
            min_corr=('corr', 'min'),
            max_corr=('corr', 'max'),
        )
    )
    summary_df['feature_name'] = pd.Categorical(
        summary_df['feature_name'],
        categories=FEATURE_PLOT_ORDER,
        ordered=True,
    )
    return summary_df.sort_values('feature_name').reset_index(drop=True)


def add_half_violin(ax, values, position, color, width=0.7):
    violin = ax.violinplot(
        [values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin['bodies']:
        # Collapse the right half onto the center line so the exported shape is a true half violin.
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.minimum(vertices[:, 0], position)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.5)
        body.set_linewidth(1.2)


def add_shifted_boxplot(ax, values, position, color, width=0.16):
    box = ax.boxplot(
        [values],
        positions=[position + 0.2],
        widths=width,
        patch_artist=True,
        showfliers=True,
        medianprops={'color': 'black', 'linewidth': 1.4},
        whiskerprops={'color': '#4d4d4d', 'linewidth': 1.0},
        capprops={'color': '#4d4d4d', 'linewidth': 1.0},
        boxprops={'edgecolor': '#4d4d4d', 'linewidth': 1.0},
    )
    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)


def plot_half_violin_box(plot_df, title, output_path, dpi):
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    positions = np.arange(1, len(FEATURE_PLOT_ORDER) + 1)

    for position, feature_name in zip(positions, FEATURE_PLOT_ORDER):
        values = plot_df.loc[plot_df['feature_name'] == feature_name, 'corr'].dropna().to_numpy()
        add_half_violin(ax, values, position, FEATURE_COLOR_MAP[feature_name])
        add_shifted_boxplot(ax, values, position, FEATURE_COLOR_MAP[feature_name])

    all_values = plot_df['corr'].dropna().to_numpy()
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_range = y_max - y_min
    if y_range == 0:
        y_range = max(abs(y_max), 1.0) * 0.1 if y_max != 0 else 0.1

    ax.set_xlim(0.45, len(FEATURE_PLOT_ORDER) + 0.8)
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.16 * y_range)
    ax.set_xticks(positions)
    ax.set_xticklabels([FEATURE_DISPLAY_MAP[name] for name in FEATURE_PLOT_ORDER], fontsize=14)
    ax.set_ylabel('Prediction Accuracy', fontsize=14)
    # ax.set_title(title, fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.axvline(3.5, color='#bdbdbd', linestyle='--', linewidth=1.0)
    ax.text(2.0, y_max + 0.08 * y_range, 'Single feature', ha='center', va='bottom', fontsize=11)
    ax.text(5.5, y_max + 0.08 * y_range, 'Merged feature', ha='center', va='bottom', fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def build_plot_title(dataset, task, target):
    target_label = TARGET_TITLE_MAP.get(target, target)
    if task == 'age':
        return f'{dataset} age: median corr across 101 random CV runs'
    return f'{dataset} {task} ({target_label}): median corr across 101 random CV runs'


def run_plot(spec, project_root, dpi):
    plot_df = build_feature_dataframe(
        project_root=project_root,
        dataset=spec['dataset'],
        target=spec['target'],
        task=spec['task'],
    )
    output_dir = ensure_dir(spec['output_dir'])
    output_path = os.path.join(output_dir, spec['filename'])
    plot_half_violin_box(
        plot_df=plot_df,
        title=build_plot_title(spec['dataset'], spec['task'], spec['target']),
        output_path=output_path,
        dpi=dpi,
    )
    summary_df = summarize_feature_dataframe(plot_df)
    return output_path, summary_df


def main():
    args = parse_args()
    ensure_dir(args.output_root)

    all_summaries = []
    for spec in get_plot_specifications(args):
        output_path, summary_df = run_plot(
            spec=spec,
            project_root=args.project_root,
            dpi=args.dpi,
        )
        all_summaries.append(summary_df)
        print(f'Saved figure: {output_path}')

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = os.path.join(args.output_root, 'feature_merge_distribution_summary.csv')
    combined_summary.to_csv(summary_path, index=False)
    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    main()
