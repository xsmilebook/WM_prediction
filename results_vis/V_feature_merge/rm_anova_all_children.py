import os

import pandas as pd
from scipy import stats

from common import (
    MERGED_FEATURES,
    MERGED_TO_CHILDREN,
    ensure_dir,
    format_float,
    get_output_dir,
    load_target_feature_data,
    parse_common_args,
    plot_boxplot_with_points,
    resolve_targets,
)


def build_group_data(merged_feature, merged_df, baseline_data):
    feature_order = MERGED_TO_CHILDREN[merged_feature] + [merged_feature]
    group_data = {}
    for feature_name in feature_order:
        source_df = merged_df if feature_name == merged_feature else baseline_data[feature_name]
        values = source_df['corr'].dropna().to_numpy()
        if len(values) == 0:
            raise ValueError(f'No observations available for {merged_feature} {feature_name}.')
        group_data[feature_name] = values

    return group_data, feature_order


def analyze_target(project_root, dataset, target, output_root=None):
    target_dir, baseline_data, merged_data = load_target_feature_data(project_root, dataset, target)
    output_dir = ensure_dir(get_output_dir(target_dir, output_root))
    figure_dir = ensure_dir(os.path.join(output_dir, 'figures', 'rm_anova'))

    rows = []
    for merged_feature in MERGED_FEATURES:
        group_data, feature_order = build_group_data(
            merged_feature=merged_feature,
            merged_df=merged_data[merged_feature],
            baseline_data=baseline_data,
        )

        group_values = [group_data[feature_name] for feature_name in feature_order]
        f_value, p_value = stats.f_oneway(*group_values)
        df_num = float(len(feature_order) - 1)
        total_n = int(sum(len(values) for values in group_values))
        df_den = float(total_n - len(feature_order))
        partial_eta_squared = (f_value * df_num) / ((f_value * df_num) + df_den)

        rows.append(
            {
                'target': target,
                'merged_feature': merged_feature,
                'features_included': ';'.join(feature_order),
                'group_sizes': ';'.join(
                    f'{feature_name}:{len(group_data[feature_name])}' for feature_name in feature_order
                ),
                'total_n': total_n,
                'df_num': df_num,
                'df_den': df_den,
                'f_value': float(f_value),
                'p_value': float(p_value),
                'partial_eta_squared': partial_eta_squared,
            }
        )

        output_path = os.path.join(
            figure_dir,
            f'{merged_feature}_rm_anova_boxplot.png',
        )
        title = (
            f'{target}: {merged_feature} one-way ANOVA\n'
            f'F({format_float(df_num)}, {format_float(df_den)})={format_float(f_value)}, '
            f'p={format_float(p_value)}, partial eta^2={format_float(partial_eta_squared)}'
        )
        colors = ['#b8c4d6'] * (len(feature_order) - 1) + ['#de8f6e']
        plot_boxplot_with_points(
            series_list=group_values,
            labels=feature_order,
            colors=colors,
            title=title,
            ylabel='Median correlation across random CV runs',
            output_path=output_path,
        )

    result_df = pd.DataFrame(rows)
    output_csv = os.path.join(output_dir, 'rm_anova_all_children.csv')
    result_df.to_csv(output_csv, index=False)
    return result_df, output_csv


def main():
    args = parse_common_args(
        'Run one-way ANOVA for merged FC features against all child features.',
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
        print(f'Saved one-way ANOVA summary to {output_csv}\n')

    if len(all_results) > 1:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(combined_df.to_string(index=False))
        print('Per-target outputs:')
        for output_csv in output_paths:
            print(f'  {output_csv}')


if __name__ == '__main__':
    main()
