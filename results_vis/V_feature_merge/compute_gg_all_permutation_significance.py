import os

import numpy as np
import pandas as pd

from common import (
    DEFAULT_RESULTS_ROOT,
    FEATURE_DISPLAY_MAP,
    ensure_dir,
    get_significance_label,
    get_target_paths,
    load_corr_series,
    parse_common_args,
    resolve_targets,
)

GG_FEATURE_NAME = 'GGFC'
ALL_FEATURE_NAME = 'GG_GW_WW_MergedFC'


def compute_empirical_right_tail_p(observed_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return (np.sum(null_values >= observed_value) + 1.0) / (null_values.size + 1.0)


def build_observed_delta_df(baseline_dir, merged_dir):
    gg_df = load_corr_series(baseline_dir, GG_FEATURE_NAME).rename(columns={'corr': 'gg_corr'})
    all_df = load_corr_series(merged_dir, ALL_FEATURE_NAME).rename(columns={'corr': 'all_corr'})
    observed_df = gg_df.merge(all_df, on='time_id', how='inner', validate='one_to_one')
    if observed_df.empty:
        raise ValueError('No paired observed runs available for GGFC and GG_GW_WW_MergedFC.')
    observed_df['delta_corr'] = observed_df['all_corr'] - observed_df['gg_corr']
    return observed_df.sort_values('time_id').reset_index(drop=True)


def build_permutation_delta_df(permutation_dir):
    gg_df = load_corr_series(permutation_dir, GG_FEATURE_NAME).rename(columns={'corr': 'gg_corr'})
    all_df = load_corr_series(permutation_dir, ALL_FEATURE_NAME).rename(columns={'corr': 'all_corr'})
    permutation_df = gg_df.merge(all_df, on='time_id', how='inner', validate='one_to_one')
    if permutation_df.empty:
        raise ValueError('No paired permutation runs available for GGFC and GG_GW_WW_MergedFC.')
    permutation_df['delta_corr'] = permutation_df['all_corr'] - permutation_df['gg_corr']
    return permutation_df.sort_values('time_id').reset_index(drop=True)


def main():
    args = parse_common_args(
        'Summarize GG vs GG+GW+WW shared target-permutation results and empirical significance.'
    )
    targets = resolve_targets(args.task, args.targets)
    output_dir = ensure_dir(args.output_root if args.output_root else DEFAULT_RESULTS_ROOT)

    detail_rows = []
    summary_rows = []

    for target in targets:
        target_dir, baseline_dir, merged_dir = get_target_paths(args.project_root, args.dataset, target)
        observed_df = build_observed_delta_df(baseline_dir, merged_dir).copy()
        observed_df['dataset'] = args.dataset
        observed_df['target'] = target
        observed_df['gg_feature_name'] = GG_FEATURE_NAME
        observed_df['all_feature_name'] = ALL_FEATURE_NAME
        observed_df['source'] = 'observed_random_cv'
        detail_rows.append(observed_df)

        permutation_dir = os.path.join(
            target_dir,
            'V_feature_merge',
            'RegressCovariates_RandomCV_Permutation_GG_All',
        )

        if not os.path.exists(permutation_dir):
            observed_median_gg = float(np.median(observed_df['gg_corr'].to_numpy(dtype=float)))
            observed_median_all = float(np.median(observed_df['all_corr'].to_numpy(dtype=float)))
            summary_rows.append(
                pd.DataFrame(
                    [
                        {
                            'dataset': args.dataset,
                            'target': target,
                            'gg_feature_name': GG_FEATURE_NAME,
                            'gg_feature_label': FEATURE_DISPLAY_MAP[GG_FEATURE_NAME],
                            'all_feature_name': ALL_FEATURE_NAME,
                            'all_feature_label': FEATURE_DISPLAY_MAP[ALL_FEATURE_NAME],
                            'n_observed_runs': int(len(observed_df)),
                            'observed_median_gg_corr': observed_median_gg,
                            'observed_median_all_corr': observed_median_all,
                            'observed_mean_delta_corr': float(
                                np.mean(observed_df['delta_corr'].to_numpy(dtype=float))
                            ),
                            'observed_median_delta_corr': observed_median_all - observed_median_gg,
                            'n_permutation_runs': np.nan,
                            'permutation_mean_delta_corr': np.nan,
                            'permutation_median_delta_corr': np.nan,
                            'permutation_std_delta_corr': np.nan,
                            'permutation_min_delta_corr': np.nan,
                            'permutation_max_delta_corr': np.nan,
                            'z_score_vs_permutation': np.nan,
                            'empirical_p_right_tail': np.nan,
                            'significance_label': np.nan,
                            'permutation_dir': permutation_dir,
                            'status': 'missing_permutation_dir',
                        }
                    ]
                )
            )
            continue

        permutation_df = build_permutation_delta_df(permutation_dir).copy()
        observed_median_gg = float(np.median(observed_df['gg_corr'].to_numpy(dtype=float)))
        observed_median_all = float(np.median(observed_df['all_corr'].to_numpy(dtype=float)))
        observed_delta = observed_median_all - observed_median_gg

        permutation_df['dataset'] = args.dataset
        permutation_df['target'] = target
        permutation_df['gg_feature_name'] = GG_FEATURE_NAME
        permutation_df['all_feature_name'] = ALL_FEATURE_NAME
        permutation_df['observed_median_delta_corr'] = observed_delta
        permutation_df['delta_minus_observed'] = permutation_df['delta_corr'] - observed_delta
        permutation_df['source'] = 'shared_target_permutation'
        detail_rows.append(permutation_df)

        null_values = permutation_df['delta_corr'].to_numpy(dtype=float)
        permutation_mean = float(np.mean(null_values))
        permutation_std = float(np.std(null_values, ddof=1))
        p_value = compute_empirical_right_tail_p(observed_delta, null_values)

        summary_rows.append(
            pd.DataFrame(
                [
                    {
                        'dataset': args.dataset,
                        'target': target,
                        'gg_feature_name': GG_FEATURE_NAME,
                        'gg_feature_label': FEATURE_DISPLAY_MAP[GG_FEATURE_NAME],
                        'all_feature_name': ALL_FEATURE_NAME,
                        'all_feature_label': FEATURE_DISPLAY_MAP[ALL_FEATURE_NAME],
                        'n_observed_runs': int(len(observed_df)),
                        'observed_median_gg_corr': observed_median_gg,
                        'observed_median_all_corr': observed_median_all,
                        'observed_mean_delta_corr': float(
                            np.mean(observed_df['delta_corr'].to_numpy(dtype=float))
                        ),
                        'observed_median_delta_corr': observed_delta,
                        'n_permutation_runs': int(len(permutation_df)),
                        'permutation_mean_delta_corr': permutation_mean,
                        'permutation_median_delta_corr': float(np.median(null_values)),
                        'permutation_std_delta_corr': permutation_std,
                        'permutation_min_delta_corr': float(np.min(null_values)),
                        'permutation_max_delta_corr': float(np.max(null_values)),
                        'z_score_vs_permutation': (
                            (observed_delta - permutation_mean) / permutation_std
                            if permutation_std > 0
                            else np.nan
                        ),
                        'empirical_p_right_tail': p_value,
                        'significance_label': get_significance_label(p_value),
                        'permutation_dir': permutation_dir,
                        'status': 'ok',
                    }
                ]
            )
        )

    detail_df = pd.concat(detail_rows, ignore_index=True)
    summary_df = pd.concat(summary_rows, ignore_index=True)

    detail_path = os.path.join(
        output_dir,
        f'{args.dataset}_{args.task}_gg_all_permutation_detail.csv',
    )
    summary_path = os.path.join(
        output_dir,
        f'{args.dataset}_{args.task}_gg_all_permutation_significance.csv',
    )
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f'Saved permutation detail to {detail_path}')
    print(f'Saved permutation significance summary to {summary_path}')


if __name__ == '__main__':
    main()
