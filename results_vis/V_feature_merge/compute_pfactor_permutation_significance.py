import argparse
import os

import numpy as np
import pandas as pd

from common import (
    BASELINE_FEATURES,
    DEFAULT_PFACTOR_SIGNIFICANCE_TARGETS,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    FEATURE_DISPLAY_MAP,
    FEATURE_PLOT_ORDER,
    MERGED_FEATURES,
    ensure_dir,
    get_significance_label,
    load_corr_series,
    load_target_feature_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ABCD pfactor feature performance against permutation null correlations.',
    )
    parser.add_argument(
        '--dataset',
        default='ABCD',
        help='Dataset name. Default: ABCD.',
    )
    parser.add_argument(
        '--targets',
        nargs='*',
        default=DEFAULT_PFACTOR_SIGNIFICANCE_TARGETS,
        help='pfactor targets to evaluate. Default: General Ext ADHD.',
    )
    parser.add_argument(
        '--project_root',
        default=DEFAULT_PROJECT_ROOT,
        help='Root directory containing per-dataset prediction folders.',
    )
    parser.add_argument(
        '--output_root',
        default=DEFAULT_RESULTS_ROOT,
        help='Directory for exported CSV results.',
    )
    return parser.parse_args()


def get_permutation_dirs(project_root, dataset, target):
    target_dir = os.path.join(project_root, dataset, 'prediction', target)
    baseline_perm_dir = os.path.join(target_dir, 'RegressCovariates_RandomCV_Permutation')
    merged_perm_dir = os.path.join(target_dir, 'V_feature_merge', 'RegressCovariates_RandomCV_Permutation')
    return target_dir, baseline_perm_dir, merged_perm_dir


def compute_empirical_right_tail_p(observed_value, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return (np.sum(null_values >= observed_value) + 1.0) / (null_values.size + 1.0)


def summarize_target(project_root, dataset, target):
    _, baseline_data, merged_data = load_target_feature_data(project_root, dataset, target)
    _, baseline_perm_dir, merged_perm_dir = get_permutation_dirs(project_root, dataset, target)

    observed_sources = {}
    observed_sources.update(baseline_data)
    observed_sources.update(merged_data)

    rows = []
    for feature_name in FEATURE_PLOT_ORDER:
        observed_df = observed_sources[feature_name]
        observed_values = observed_df['corr'].dropna().to_numpy(dtype=float)
        observed_median = float(np.median(observed_values))
        observed_mean = float(np.mean(observed_values))

        row = {
            'dataset': dataset,
            'target': target,
            'feature_name': feature_name,
            'feature_label': FEATURE_DISPLAY_MAP[feature_name],
            'n_actual_runs': int(observed_values.size),
            'observed_median_corr': observed_median,
            'observed_mean_corr': observed_mean,
            'n_permutation_runs': np.nan,
            'permutation_mean_corr': np.nan,
            'permutation_median_corr': np.nan,
            'permutation_std_corr': np.nan,
            'z_score_vs_permutation': np.nan,
            'empirical_p_right_tail': np.nan,
            'significance_label': np.nan,
            'permutation_source': '',
            'status': '',
        }

        if feature_name in BASELINE_FEATURES:
            if not os.path.exists(baseline_perm_dir):
                row['status'] = 'missing_baseline_permutation_dir'
            else:
                perm_df = load_corr_series(baseline_perm_dir, feature_name)
                perm_values = perm_df['corr'].dropna().to_numpy(dtype=float)
                perm_std = float(np.std(perm_values, ddof=1))
                p_value = compute_empirical_right_tail_p(observed_median, perm_values)

                row.update(
                    {
                        'n_permutation_runs': int(perm_values.size),
                        'permutation_mean_corr': float(np.mean(perm_values)),
                        'permutation_median_corr': float(np.median(perm_values)),
                        'permutation_std_corr': perm_std,
                        'z_score_vs_permutation': (
                            (observed_median - float(np.mean(perm_values))) / perm_std
                            if perm_std > 0
                            else np.nan
                        ),
                        'empirical_p_right_tail': p_value,
                        'significance_label': get_significance_label(p_value),
                        'permutation_source': 'baseline_feature_permutation',
                        'status': 'ok',
                    }
                )
        else:
            if os.path.exists(merged_perm_dir):
                perm_df = load_corr_series(merged_perm_dir, feature_name)
                perm_values = perm_df['corr'].dropna().to_numpy(dtype=float)
                perm_std = float(np.std(perm_values, ddof=1))
                p_value = compute_empirical_right_tail_p(observed_median, perm_values)
                row.update(
                    {
                        'n_permutation_runs': int(perm_values.size),
                        'permutation_mean_corr': float(np.mean(perm_values)),
                        'permutation_median_corr': float(np.median(perm_values)),
                        'permutation_std_corr': perm_std,
                        'z_score_vs_permutation': (
                            (observed_median - float(np.mean(perm_values))) / perm_std
                            if perm_std > 0
                            else np.nan
                        ),
                        'empirical_p_right_tail': p_value,
                        'significance_label': get_significance_label(p_value),
                        'permutation_source': 'merged_feature_permutation',
                        'status': 'ok',
                    }
                )
            else:
                row.update(
                    {
                        'permutation_source': 'unavailable',
                        'status': 'missing_merged_permutation_dir',
                    }
                )

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df['feature_name'] = pd.Categorical(
        result_df['feature_name'],
        categories=FEATURE_PLOT_ORDER,
        ordered=True,
    )
    return result_df.sort_values('feature_name').reset_index(drop=True)


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_root)

    all_results = []
    for target in args.targets:
        target_df = summarize_target(
            project_root=args.project_root,
            dataset=args.dataset,
            target=target,
        )
        all_results.append(target_df)
        print(target_df.to_string(index=False))
        print()

    result_df = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(output_dir, f'{args.dataset}_pfactor_permutation_significance.csv')
    result_df.to_csv(output_path, index=False)
    print(f'Saved permutation significance summary to {output_path}')


if __name__ == '__main__':
    main()
