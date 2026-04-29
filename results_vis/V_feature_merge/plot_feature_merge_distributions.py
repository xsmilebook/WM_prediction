import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from common import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    FEATURE_DISPLAY_MAP,
    ensure_dir,
    load_target_feature_data,
)

AVAILABLE_FONT_NAMES = {font.name for font in font_manager.fontManager.ttflist}
PREFERRED_FONT_FAMILY = 'Arial' if 'Arial' in AVAILABLE_FONT_NAMES else 'DejaVu Sans'

plt.rcParams.update({
    'font.family': PREFERRED_FONT_FAMILY
})

AGE_DATASETS = ['HCPD', 'CCNP', 'EFNY', 'PNC']
ABCD_DATASET = 'ABCD'
PLOTTED_FEATURES = ['GGFC', 'GG_GW_WW_MergedFC']
REFERENCE_GROUP_COUNT = 4
REFERENCE_VIOLIN_WIDTH = 0.2
REFERENCE_BOX_WIDTH = 0.1
REFERENCE_FEATURE_OFFSETS = {
    'GGFC': -0.18,
    'GG_GW_WW_MergedFC': 0.18,
}
REFERENCE_BOX_SHIFTS = {
    'GGFC': 0.08,
    'GG_GW_WW_MergedFC': -0.08,
}
FEATURE_SELF_SIGNIFICANCE_LABEL = '**'
SIGNIFICANCE_FONT_FAMILY = 'DejaVu Sans'
FIGURE_HEIGHT = 3.2
LEGEND_FONT_SIZE = 8
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
    'GGFC': '#A3C1E4',
    'GG_GW_WW_MergedFC': '#EB9C9F',
}


def get_figure_width(group_count, plot_name=None):
    if plot_name == 'pfactor':
        return 2.2
    return max(4.5, 1.2 * group_count + 1.0)


def get_horizontal_geometry(group_count, plot_name=None):
    reference_figure_width = get_figure_width(REFERENCE_GROUP_COUNT, plot_name='age')
    current_figure_width = get_figure_width(group_count, plot_name=plot_name)
    reference_scale = reference_figure_width / REFERENCE_GROUP_COUNT
    current_scale = current_figure_width / group_count
    
    if plot_name == 'pfactor':
        width_scale = 1.0
    else:
        width_scale = reference_scale / current_scale


    return {
        'violin_width': REFERENCE_VIOLIN_WIDTH * width_scale,
        'box_width': REFERENCE_BOX_WIDTH * width_scale,
        'feature_offsets': {
            feature_name: offset * width_scale
            for feature_name, offset in REFERENCE_FEATURE_OFFSETS.items()
        },
        'box_shifts': {
            feature_name: shift * width_scale
            for feature_name, shift in REFERENCE_BOX_SHIFTS.items()
        },
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
        specs.append(
            {
                'plot_name': 'age',
                'task': 'age',
                'group_items': [
                    {
                        'dataset': dataset,
                        'target': 'age',
                        'group_label': group_label,
                    }
                    for dataset, group_label in [
                        ('EFNY', 'EFNY'),
                        ('CCNP', 'devCCNP'),
                        ('HCPD', 'HCP-D'),
                        ('PNC', 'PNC'),
                    ]
                    if dataset in args.age_datasets
                ],
                'output_dir': os.path.join(args.output_root, 'age'),
                'filename': 'age_all_datasets_GG_vs_GG_GW_WW_half_violin_box.tiff',
            }
        )

    if not args.skip_abcd:
        cognition_targets = [
            'nihtbx_totalcomp_uncorrected',
            'nihtbx_cryst_uncorrected',
            'nihtbx_fluidcomp_uncorrected',
        ]
        pfactor_targets = ['ADHD']

        specs.append(
            {
                'plot_name': 'cognition',
                'task': 'cognition',
                'group_items': [
                    {
                        'dataset': args.abcd_dataset,
                        'target': target,
                        'group_label': TARGET_TITLE_MAP.get(target, target),
                    }
                    for target in cognition_targets
                ],
                'output_dir': os.path.join(args.output_root, args.abcd_dataset, 'cognition'),
                'filename': 'cognition_all_targets_GG_vs_GG_GW_WW_half_violin_box.tiff',
            }
        )
        specs.append(
            {
                'plot_name': 'pfactor',
                'task': 'pfactor',
                'group_items': [
                    {
                        'dataset': args.abcd_dataset,
                        'target': target,
                        'group_label': TARGET_TITLE_MAP.get(target, target),
                    }
                    for target in pfactor_targets
                ],
                'output_dir': os.path.join(args.output_root, args.abcd_dataset, 'pfactor'),
                'filename': 'pfactor_all_targets_GG_vs_GG_GW_WW_half_violin_box.tiff',
            }
        )

    return specs


def build_feature_dataframe(project_root, dataset, target, task, plot_name, group_label):
    _, baseline_data, merged_data = load_target_feature_data(project_root, dataset, target)
    feature_sources = {}
    feature_sources.update(baseline_data)
    feature_sources.update(merged_data)

    rows = []
    for feature_name in PLOTTED_FEATURES:
        feature_df = feature_sources[feature_name]
        for _, row in feature_df.iterrows():
            rows.append(
                {
                    'plot_name': plot_name,
                    'group_label': group_label,
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
        plot_df.groupby(
            ['plot_name', 'group_label', 'dataset', 'task', 'target', 'feature_name', 'feature_label'],
            as_index=False,
        )
        .agg(
            n_runs=('corr', 'size'),
            median_corr=('corr', 'median'),
            mean_corr=('corr', 'mean'),
            std_corr=('corr', 'std'),
            min_corr=('corr', 'min'),
            max_corr=('corr', 'max'),
        )
    )
    summary_df['feature_name'] = pd.Categorical(summary_df['feature_name'], categories=PLOTTED_FEATURES, ordered=True)
    return summary_df.sort_values(['plot_name', 'group_label', 'feature_name']).reset_index(drop=True)


def add_half_violin(ax, values, position, color, side, width):
    violin = ax.violinplot(
        [values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin['bodies']:
        vertices = body.get_paths()[0].vertices
        if side == 'left':
            vertices[:, 0] = np.minimum(vertices[:, 0], position)
        elif side == 'right':
            vertices[:, 0] = np.maximum(vertices[:, 0], position)
        else:
            raise ValueError(f'Unsupported violin side: {side}')
        body.set_facecolor(color)
        body.set_edgecolor('black')
        # body.set_alpha(0.5)
        body.set_linewidth(0.5)


def add_shifted_boxplot(ax, values, position, color, shift, width):
    flier_style = dict(
        marker='o',
        markerfacecolor=color,
        markersize=3,
        markeredgecolor='black',
        markeredgewidth=0.5,
        alpha=1.0,
    )
    
    box = ax.boxplot(
        [values],
        positions=[position + shift],
        widths=width,
        patch_artist=True,
        showfliers=True,
        flierprops=flier_style,
        medianprops={'color': 'black', 'linewidth': 0.7},
        whiskerprops={'color': 'black', 'linewidth': 0.5},
        capprops={'color': 'black', 'linewidth': 0.5},
        boxprops={'edgecolor': 'black', 'linewidth': 0.5},
    )
    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    return position + shift


def get_feature_significance_x(box_center, shift, box_width):
    if shift > 0:
        return box_center - box_width / 2.0
    if shift < 0:
        return box_center + box_width / 2.0
    return box_center


def get_paired_corr_arrays(group_df):
    gg_df = (
        group_df.loc[group_df['feature_name'] == 'GGFC', ['time_id', 'corr']]
        .rename(columns={'corr': 'corr_gg'})
        .sort_values('time_id')
    )
    merged_df = (
        group_df.loc[group_df['feature_name'] == 'GG_GW_WW_MergedFC', ['time_id', 'corr']]
        .rename(columns={'corr': 'corr_merged'})
        .sort_values('time_id')
    )
    paired_df = gg_df.merge(merged_df, on='time_id', how='inner').dropna(subset=['corr_gg', 'corr_merged'])
    gg_values = paired_df['corr_gg'].to_numpy()
    merged_values = paired_df['corr_merged'].to_numpy()
    if len(gg_values) == 0:
        raise ValueError('No paired observations available for GG vs GG_GW_WW_MergedFC.')
    return gg_values, merged_values


def compute_significance(group_df):
    gg_values, merged_values = get_paired_corr_arrays(group_df)
    t_stat, p_value = stats.ttest_rel(merged_values, gg_values)
    delta = merged_values - gg_values
    return {
        'n_pairs': len(gg_values),
        'gg_median_corr': float(np.median(gg_values)),
        'gg_gw_ww_median_corr': float(np.median(merged_values)),
        'mean_delta_corr': delta.mean(),
        'median_delta_corr': float(np.median(delta)),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significance_label': get_plot_significance_label(p_value),
    }


def get_plot_significance_label(p_value):
    if p_value < 0.001:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def draw_significance_label(ax, x, y, label):
    ax.text(
        x,
        y,
        label,
        ha='center',
        va='bottom',
        fontsize=7,
        fontweight='bold',
        fontfamily=SIGNIFICANCE_FONT_FAMILY,
    )


def add_significance_bar(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', linewidth=0.5)
    draw_significance_label(ax, (x1 + x2) / 2, y + h, label)


def plot_half_violin_box(plot_df, group_labels, output_path, dpi):
    plot_name = plot_df['plot_name'].iloc[0]
    fig_width = get_figure_width(len(group_labels), plot_name=plot_name)
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT))
    centers = np.arange(1, len(group_labels) + 1)
    geometry = get_horizontal_geometry(len(group_labels), plot_name=plot_name)
    feature_offsets = geometry['feature_offsets']
    box_shifts = geometry['box_shifts']
    violin_sides = {
        'GGFC': 'left',
        'GG_GW_WW_MergedFC': 'right',
    }
    feature_marker_x = []

    for center, group_label in zip(centers, group_labels):
        group_df = plot_df.loc[plot_df['group_label'] == group_label]
        for feature_name in PLOTTED_FEATURES:
            values = group_df.loc[group_df['feature_name'] == feature_name, 'corr'].dropna().to_numpy()
            position = center + feature_offsets[feature_name]
            add_half_violin(
                ax,
                values,
                position,
                FEATURE_COLOR_MAP[feature_name],
                side=violin_sides[feature_name],
                width=geometry['violin_width'],
            )
            box_center = add_shifted_boxplot(
                ax,
                values,
                position,
                FEATURE_COLOR_MAP[feature_name],
                shift=box_shifts[feature_name],
                width=geometry['box_width'],
            )
            feature_marker_x.append(
                (
                    group_label,
                    feature_name,
                    get_feature_significance_x(
                        box_center=box_center,
                        shift=box_shifts[feature_name],
                        box_width=geometry['box_width'],
                    ),
                )
            )

    all_values = plot_df['corr'].dropna().to_numpy()
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_range = y_max - y_min
    if y_range == 0:
        y_range = max(abs(y_max), 1.0) * 0.1 if y_max != 0 else 0.1
    axis_y_min = y_min - 0.08 * y_range
    axis_y_max = None
    annotation_range = y_range
    if plot_name == 'pfactor':
        axis_y_min = 0.03
        axis_y_max = 0.145
        annotation_range = axis_y_max - axis_y_min

    feature_max_map = (
        plot_df.groupby(['group_label', 'feature_name'])['corr']
        .max()
        .to_dict()
    )
    for group_label, feature_name, marker_x in feature_marker_x:
        # marker_y = float(feature_max_map[(group_label, feature_name)]) - 0.005 * annotation_range
        marker_y = float(feature_max_map[(group_label, feature_name)])
        if axis_y_max is not None:
            marker_y = min(marker_y, axis_y_max - 0.02 * annotation_range)
        draw_significance_label(ax, marker_x, marker_y, FEATURE_SELF_SIGNIFICANCE_LABEL)

    significance_rows = []
    bar_tops = []
    for center, group_label in zip(centers, group_labels):
        group_df = plot_df.loc[plot_df['group_label'] == group_label]
        stats_row = compute_significance(group_df)
        stats_row['group_label'] = group_label
        significance_rows.append(stats_row)

        group_max = float(group_df['corr'].max())
        bar_y = group_max + 0.05 * annotation_range
        bar_height = 0.015 * annotation_range
        if axis_y_max is not None:
            bar_y = min(bar_y, axis_y_max - 0.04 * annotation_range)
        bar_tops.append(bar_y + bar_height)
        add_significance_bar(
            ax,
            center + feature_offsets['GGFC'] + box_shifts['GGFC'],
            center + feature_offsets['GG_GW_WW_MergedFC'] + box_shifts['GG_GW_WW_MergedFC'],
            bar_y,
            bar_height,
            stats_row['significance_label'],
        )

    upper_ylim = max(bar_tops) + 0.07 * y_range if bar_tops else y_max + 0.16 * y_range
    if axis_y_max is not None:
        upper_ylim = axis_y_max
    # if plot_name == 'pfactor':
    #     ax.set_xlim(0.72, 1.28)
    # else:
    #     ax.set_xlim(0.5, len(group_labels) + 0.5)
    ax.set_xlim(0.5, len(group_labels) + 0.5)
    
    ax.set_ylim(axis_y_min, upper_ylim)
    if plot_name == 'age':
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_ylabel('Prediction Accuracy of Brain Age', fontsize=10)
    elif plot_name == 'cognition':
        ax.set_ylabel('Prediction Accuracy of Cognition', fontsize=10)
    else:
        ax.set_ylabel('Prediction Accuracy of Psychopathology', fontsize=10)
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels, fontsize=10)
    # ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)   
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(
        title="Legend",
        alignment="left",
        handles=[
            Patch(facecolor=FEATURE_COLOR_MAP['GGFC'], edgecolor=FEATURE_COLOR_MAP['GGFC'], alpha=0.5, label='G-G'),
            Patch(
                facecolor=FEATURE_COLOR_MAP['GG_GW_WW_MergedFC'],
                edgecolor='black',
                alpha=0.5,
                label='G-G,G-W,W-W',
            ),
        ],
        loc='upper right',
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    output_png_path = output_path.replace("tiff", "png")
    fig.savefig(output_png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return pd.DataFrame(significance_rows)

def run_plot(spec, project_root, dpi):
    plot_frames = []
    for item in spec['group_items']:
        plot_frames.append(
            build_feature_dataframe(
                project_root=project_root,
                dataset=item['dataset'],
                target=item['target'],
                task=spec['task'],
                plot_name=spec['plot_name'],
                group_label=item['group_label'],
            )
        )
    plot_df = pd.concat(plot_frames, ignore_index=True)
    output_dir = ensure_dir(spec['output_dir'])
    output_path = os.path.join(output_dir, spec['filename'])
    significance_df = plot_half_violin_box(
        plot_df=plot_df,
        group_labels=[item['group_label'] for item in spec['group_items']],
        output_path=output_path,
        dpi=dpi,
    )
    summary_df = summarize_feature_dataframe(plot_df)
    significance_df.insert(0, 'plot_name', spec['plot_name'])
    significance_df.insert(1, 'task', spec['task'])
    return output_path, summary_df, significance_df


def main():
    args = parse_args()
    ensure_dir(args.output_root)

    all_summaries = []
    all_significance = []
    for spec in get_plot_specifications(args):
        output_path, summary_df, significance_df = run_plot(
            spec=spec,
            project_root=args.project_root,
            dpi=args.dpi,
        )
        all_summaries.append(summary_df)
        all_significance.append(significance_df)
        print(f'Saved figure: {output_path}')

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = os.path.join(args.output_root, 'feature_merge_distribution_summary.csv')
    combined_summary.to_csv(summary_path, index=False)
    print(f'Saved summary: {summary_path}')

    combined_significance = pd.concat(all_significance, ignore_index=True)
    significance_path = os.path.join(args.output_root, 'feature_merge_distribution_significance.csv')
    combined_significance.to_csv(significance_path, index=False)
    print(f'Saved significance summary: {significance_path}')


if __name__ == '__main__':
    main()
