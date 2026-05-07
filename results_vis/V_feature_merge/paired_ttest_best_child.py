import os

import pandas as pd
import scipy.io as sio
from scipy import stats

from common import (
    DEFAULT_RESULTS_ROOT,
    ensure_dir,
    format_float,
    get_output_dir,
    get_significance_label,
    load_target_feature_data,
    parse_common_args,
    plot_boxplot_with_points,
)

AGE_DATASETS = ['HCPD', 'CCNP', 'EFNY', 'PNC']
COGNITION_TARGETS = [
    'nihtbx_totalcomp_uncorrected',
    'nihtbx_cryst_uncorrected',
    'nihtbx_fluidcomp_uncorrected',
]
PFACTOR_TARGETS = ['ADHD']

BASELINE_FEATURE = 'GGFC'
MERGED_FEATURE = 'GG_GW_WW_MergedFC'

R_OUTER = 101
K_FOLD = 5
J_TOTAL = R_OUTER * K_FOLD
N2_OVER_N1 = 1 / 4


def get_summary_output_path(task):
    summary_dir = ensure_dir(os.path.join(DEFAULT_RESULTS_ROOT, 'paired_ttest_pvalues'))
    return os.path.join(summary_dir, f'{task}.csv')


def load_fold_corr_series(result_dir, feature_name):
    rows = []
    for time_id in range(R_OUTER):
        for fold_id in range(K_FOLD):
            result_path = os.path.join(
                result_dir,
                f'Time_{time_id}',
                feature_name,
                f'Fold_{fold_id}_Score.mat',
            )
            if not os.path.exists(result_path):
                raise FileNotFoundError(f'Missing fold result file: {result_path}')
            result_mat = sio.loadmat(result_path)
            rows.append(
                {
                    'time_id': time_id,
                    'fold_id': fold_id,
                    'corr': float(result_mat['Corr'].item()),
                }
            )
    return pd.DataFrame(rows).sort_values(['time_id', 'fold_id']).reset_index(drop=True)


def corrected_resampled_ttest(delta_series):
    mean_delta = float(delta_series.mean())
    sigma = float(delta_series.std(ddof=1))
    if sigma == 0:
        if mean_delta == 0:
            return 0.0, 1.0
        return float('inf'), 0.0

    scale = (1 / J_TOTAL + N2_OVER_N1) ** 0.5
    t_stat = mean_delta / (scale * sigma)
    p_value = float(2 * stats.t.sf(abs(t_stat), df=J_TOTAL - 1))
    return float(t_stat), p_value


def bh_fdr(p_values):
    series = pd.Series(p_values, dtype=float)
    valid = series.notna()
    adjusted = pd.Series(float('nan'), index=series.index, dtype=float)
    if not valid.any():
        return adjusted

    ranked = series[valid].sort_values()
    m = len(ranked)
    bh_values = ranked * m / pd.Series(range(1, m + 1), index=ranked.index, dtype=float)
    bh_values = bh_values[::-1].cummin()[::-1].clip(upper=1.0)
    adjusted.loc[bh_values.index] = bh_values
    return adjusted


def get_group_specs(task, dataset):
    if task == 'age':
        return [{'dataset': item, 'target': 'age'} for item in AGE_DATASETS]
    if task == 'cognition':
        return [{'dataset': dataset, 'target': item} for item in COGNITION_TARGETS]
    if task == 'pfactor':
        return [{'dataset': dataset, 'target': item} for item in PFACTOR_TARGETS]
    raise ValueError(f'Unsupported task: {task}')


def analyze_target(project_root, dataset, target, output_root=None):
    target_dir, baseline_data, merged_data = load_target_feature_data(project_root, dataset, target)
    output_dir = ensure_dir(get_output_dir(target_dir, output_root))
    figure_dir = ensure_dir(os.path.join(output_dir, 'figures', 'paired_ttest'))

    baseline_dir = os.path.join(target_dir, 'RegressCovariates_RandomCV')
    merged_dir = os.path.join(target_dir, 'V_feature_merge', 'RegressCovariates_RandomCV')

    paired_time_df = baseline_data[BASELINE_FEATURE].merge(
        merged_data[MERGED_FEATURE],
        on='time_id',
        suffixes=('_gg', '_merged'),
        how='inner',
    ).dropna(subset=['corr_gg', 'corr_merged'])
    if paired_time_df.empty:
        raise ValueError(f'No paired 101-run observations available for {dataset} {target}.')

    gg_fold_df = load_fold_corr_series(baseline_dir, BASELINE_FEATURE)
    merged_fold_df = load_fold_corr_series(merged_dir, MERGED_FEATURE)
    paired_fold_df = gg_fold_df.merge(
        merged_fold_df,
        on=['time_id', 'fold_id'],
        suffixes=('_gg', '_merged'),
        how='inner',
    ).dropna(subset=['corr_gg', 'corr_merged'])
    if paired_fold_df.empty:
        raise ValueError(f'No paired 505-fold observations available for {dataset} {target}.')

    delta_time = paired_time_df['corr_merged'] - paired_time_df['corr_gg']
    delta_fold = paired_fold_df['corr_merged'] - paired_fold_df['corr_gg']

    t_stat_101, p_value_101 = stats.ttest_rel(paired_time_df['corr_merged'], paired_time_df['corr_gg'])
    t_stat_505, p_value_505 = stats.ttest_rel(paired_fold_df['corr_merged'], paired_fold_df['corr_gg'])
    corrected_t_stat, corrected_p_value = corrected_resampled_ttest(delta_fold)

    row = {
        'dataset': dataset,
        'target': target,
        'baseline_feature': BASELINE_FEATURE,
        'merged_feature': MERGED_FEATURE,
        'n_pairs_101': len(paired_time_df),
        'n_pairs_505': len(paired_fold_df),
        'merged_median_corr': paired_time_df['corr_merged'].median(),
        'gg_median_corr': paired_time_df['corr_gg'].median(),
        'mean_delta_corr_101': delta_time.mean(),
        'median_delta_corr_101': delta_time.median(),
        'mean_delta_corr_505': delta_fold.mean(),
        'median_delta_corr_505': delta_fold.median(),
        't_stat_101': float(t_stat_101),
        'p_value_101': float(p_value_101),
        't_stat_505': float(t_stat_505),
        'p_value_505': float(p_value_505),
        'corrected_resampled_t_stat': corrected_t_stat,
        'corrected_resampled_p_value': corrected_p_value,
        'R': R_OUTER,
        'k': K_FOLD,
        'J': J_TOTAL,
        'n2_over_n1': N2_OVER_N1,
    }

    output_path = os.path.join(
        figure_dir,
        f'{MERGED_FEATURE}_vs_{BASELINE_FEATURE}_before_after_boxplot.png',
    )
    title = (
        f'{dataset} {target}: {MERGED_FEATURE} vs {BASELINE_FEATURE}\n'
        f'101-run paired t-test p={format_float(p_value_101)}'
    )
    plot_boxplot_with_points(
        series_list=[
            paired_time_df['corr_gg'].to_numpy(),
            paired_time_df['corr_merged'].to_numpy(),
        ],
        labels=['GG', 'GG+GW+WW'],
        colors=['#b8c4d6', '#de8f6e'],
        title=title,
        ylabel='Median correlation across random CV runs',
        output_path=output_path,
        significance=get_significance_label(p_value_101),
        connect_pairs=True,
    )

    return pd.DataFrame([row]), output_dir


def main():
    args = parse_common_args(
        'Run paired t-tests between GGFC and GG_GW_WW_MergedFC.',
    )

    group_specs = get_group_specs(args.task, args.dataset)
    all_results = []
    output_dirs = {}
    for spec in group_specs:
        result_df, output_dir = analyze_target(
            project_root=args.project_root,
            dataset=spec['dataset'],
            target=spec['target'],
            output_root=args.output_root,
        )
        all_results.append(result_df)
        output_dirs[(spec['dataset'], spec['target'])] = output_dir

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df['corrected_resampled_p_value_fdr'] = bh_fdr(combined_df['corrected_resampled_p_value'])
    combined_df['significance_label_101'] = combined_df['p_value_101'].map(get_significance_label)
    combined_df['significance_label_505'] = combined_df['p_value_505'].map(get_significance_label)
    combined_df['significance_label_corrected_resampled'] = combined_df['corrected_resampled_p_value'].map(
        get_significance_label
    )
    combined_df['significance_label_corrected_resampled_fdr'] = combined_df[
        'corrected_resampled_p_value_fdr'
    ].map(get_significance_label)

    summary_output_path = get_summary_output_path(args.task)
    combined_df.to_csv(summary_output_path, index=False)

    print(combined_df.to_string(index=False))
    print('Summary output:')
    print(f'  {summary_output_path}')
    print('Per-target figure directories:')
    for key in sorted(output_dirs):
        print(f'  {output_dirs[key]}')


if __name__ == '__main__':
    main()
