import os

import pandas as pd
from scipy import stats

from common import (
    MERGED_FEATURES,
    MERGED_TO_CHILDREN,
    ensure_dir,
    format_float,
    get_output_dir,
    get_significance_label,
    load_target_feature_data,
    parse_common_args,
    plot_boxplot_with_points,
    resolve_targets,
)


def choose_best_child(child_data):
    child_medians = {
        feature_name: child_df['corr'].median(skipna=True)
        for feature_name, child_df in child_data.items()
    }
    return max(child_medians, key=child_medians.get)


def analyze_target(project_root, dataset, target, output_root=None):
    target_dir, baseline_data, merged_data = load_target_feature_data(project_root, dataset, target)
    output_dir = ensure_dir(get_output_dir(target_dir, output_root))
    figure_dir = ensure_dir(os.path.join(output_dir, 'figures', 'paired_ttest'))

    rows = []
    for merged_feature in MERGED_FEATURES:
        children = MERGED_TO_CHILDREN[merged_feature]
        child_data = {feature_name: baseline_data[feature_name] for feature_name in children}
        best_child = choose_best_child(child_data)

        paired_df = child_data[best_child].merge(
            merged_data[merged_feature],
            on='time_id',
            suffixes=('_child', '_merged'),
            how='inner',
        ).dropna(subset=['corr_child', 'corr_merged'])
        if paired_df.empty:
            raise ValueError(f'No paired observations available for {target} {merged_feature}.')

        t_stat, p_value = stats.ttest_rel(paired_df['corr_merged'], paired_df['corr_child'])
        delta = paired_df['corr_merged'] - paired_df['corr_child']

        rows.append(
            {
                'target': target,
                'merged_feature': merged_feature,
                'best_child_feature': best_child,
                'n_pairs': len(paired_df),
                'merged_median_corr': paired_df['corr_merged'].median(),
                'best_child_median_corr': paired_df['corr_child'].median(),
                'mean_delta_corr': delta.mean(),
                'median_delta_corr': delta.median(),
                't_stat': float(t_stat),
                'p_value': float(p_value),
            }
        )

        output_path = os.path.join(
            figure_dir,
            f'{merged_feature}_before_after_boxplot.png',
        )
        title = (
            f'{target}: {merged_feature} vs {best_child}\n'
            f'paired t-test p={format_float(p_value)}'
        )
        plot_boxplot_with_points(
            series_list=[
                paired_df['corr_child'].to_numpy(),
                paired_df['corr_merged'].to_numpy(),
            ],
            labels=[best_child, merged_feature],
            colors=['#b8c4d6', '#de8f6e'],
            title=title,
            ylabel='Median correlation across random CV runs',
            output_path=output_path,
            significance=get_significance_label(p_value),
            connect_pairs=True,
        )

    result_df = pd.DataFrame(rows)
    output_csv = os.path.join(output_dir, 'paired_ttest_best_child.csv')
    result_df.to_csv(output_csv, index=False)
    return result_df, output_csv


def main():
    args = parse_common_args(
        'Run paired t-tests between merged FC features and their best-performing child features.',
    )
    targets = resolve_targets(args.task, args.targets)

    all_results = []
    output_paths = []
    for target in targets:
        result_df, output_csv = analyze_target(
            project_root=args.project_root,
            dataset=args.dataset,
            target=target,
            output_root=args.output_root,
        )
        all_results.append(result_df)
        output_paths.append(output_csv)
        print(result_df.to_string(index=False))
        print(f'Saved paired t-test summary to {output_csv}\n')

    if len(all_results) > 1:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(combined_df.to_string(index=False))
        print('Per-target outputs:')
        for output_csv in output_paths:
            print(f'  {output_csv}')


if __name__ == '__main__':
    main()
