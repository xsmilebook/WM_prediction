import os

import pandas as pd
from statsmodels.stats.anova import AnovaRM

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


def build_long_form(merged_feature, merged_df, baseline_data):
    feature_order = MERGED_TO_CHILDREN[merged_feature] + [merged_feature]
    frames = []
    for feature_name in feature_order:
        source_df = merged_df if feature_name == merged_feature else baseline_data[feature_name]
        feature_df = source_df.copy()
        feature_df = feature_df.rename(columns={'corr': 'corr_value'})
        feature_df['feature'] = feature_name
        frames.append(feature_df[['time_id', 'feature', 'corr_value']])

    long_df = pd.concat(frames, ignore_index=True)
    pivot_df = long_df.pivot(index='time_id', columns='feature', values='corr_value')
    balanced_df = pivot_df.dropna(axis=0, how='any')
    if balanced_df.empty:
        raise ValueError(f'No complete repeated-measures rows for {merged_feature}.')

    long_balanced = (
        balanced_df.reset_index()
        .melt(id_vars='time_id', var_name='feature', value_name='corr_value')
        .rename(columns={'time_id': 'subject_id'})
    )
    return long_balanced, feature_order


def analyze_target(project_root, dataset, target, output_root=None):
    target_dir, baseline_data, merged_data = load_target_feature_data(project_root, dataset, target)
    output_dir = ensure_dir(get_output_dir(target_dir, output_root))
    figure_dir = ensure_dir(os.path.join(output_dir, 'figures', 'rm_anova'))

    rows = []
    for merged_feature in MERGED_FEATURES:
        long_df, feature_order = build_long_form(
            merged_feature=merged_feature,
            merged_df=merged_data[merged_feature],
            baseline_data=baseline_data,
        )

        anova = AnovaRM(
            data=long_df,
            depvar='corr_value',
            subject='subject_id',
            within=['feature'],
        ).fit()
        anova_table = anova.anova_table.reset_index(drop=True).iloc[0]
        f_value = float(anova_table['F Value'])
        df_num = float(anova_table['Num DF'])
        df_den = float(anova_table['Den DF'])
        p_value = float(anova_table['Pr > F'])
        partial_eta_squared = (f_value * df_num) / ((f_value * df_num) + df_den)

        rows.append(
            {
                'target': target,
                'merged_feature': merged_feature,
                'features_included': ';'.join(feature_order),
                'n_subjects': int(long_df['subject_id'].nunique()),
                'df_num': df_num,
                'df_den': df_den,
                'f_value': f_value,
                'p_value': p_value,
                'partial_eta_squared': partial_eta_squared,
            }
        )

        pivot_df = (
            long_df.pivot(index='subject_id', columns='feature', values='corr_value')[feature_order]
        )
        output_path = os.path.join(
            figure_dir,
            f'{merged_feature}_rm_anova_boxplot.png',
        )
        title = (
            f'{target}: {merged_feature} repeated-measures ANOVA\n'
            f'F({format_float(df_num)}, {format_float(df_den)})={format_float(f_value)}, '
            f'p={format_float(p_value)}, partial eta^2={format_float(partial_eta_squared)}'
        )
        colors = ['#b8c4d6'] * (len(feature_order) - 1) + ['#de8f6e']
        plot_boxplot_with_points(
            series_list=[pivot_df[feature].to_numpy() for feature in feature_order],
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
        'Run repeated-measures ANOVA for merged FC features against all child features.',
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
        print(f'Saved repeated-measures ANOVA summary to {output_csv}\n')

    if len(all_results) > 1:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(combined_df.to_string(index=False))
        print('Per-target outputs:')
        for output_csv in output_paths:
            print(f'  {output_csv}')


if __name__ == '__main__':
    main()
