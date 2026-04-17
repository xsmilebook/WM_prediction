import argparse
import glob
import os

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
TASK_TARGETS = {
    'age': ['age'],
    'cognition': [
        'nihtbx_cryst_uncorrected',
        'nihtbx_fluidcomp_uncorrected',
        'nihtbx_totalcomp_uncorrected',
    ],
    'pfactor': ['General', 'Ext', 'ADHD', 'Int'],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize baseline and merged feature prediction performance.'
    )
    parser.add_argument('--dataset', required=True, help='Dataset name, e.g. HCPD or ABCD.')
    parser.add_argument(
        '--task',
        required=True,
        choices=sorted(TASK_TARGETS.keys()),
        help='Prediction task family.',
    )
    parser.add_argument(
        '--project_root',
        default='/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data',
        help='Root directory containing per-dataset prediction folders.',
    )
    parser.add_argument(
        '--targets',
        nargs='*',
        default=None,
        help='Optional explicit target list. Defaults depend on --task.',
    )
    parser.add_argument(
        '--output_csv',
        default=None,
        help='Optional output CSV path. Defaults to <project_root>/<dataset>/prediction/feature_merge_summary_<task>.csv',
    )
    return parser.parse_args()


def extract_time_index(path):
    time_token = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return int(time_token.split('_')[1])


def load_metric_series(result_dir, feature_name):
    pattern = os.path.join(result_dir, 'Time_*', feature_name, 'Res_NFold.mat')
    result_paths = sorted(glob.glob(pattern), key=extract_time_index)
    if not result_paths:
        raise FileNotFoundError(f'No result files found for {feature_name}: {pattern}')

    corr_values = []
    mae_values = []
    for result_path in result_paths:
        result_mat = sio.loadmat(result_path)
        corr_values.append(float(result_mat['Mean_Corr'].item()))
        mae_values.append(float(result_mat['Mean_MAE'].item()))
    return np.asarray(corr_values), np.asarray(mae_values)


def summarize_target(target_dir, target_name):
    baseline_dir = os.path.join(target_dir, 'RegressCovariates_RandomCV')
    merged_dir = os.path.join(target_dir, 'V_feature_merge', 'RegressCovariates_RandomCV')

    rows = []
    for feature_name in BASELINE_FEATURES:
        corr_values, mae_values = load_metric_series(baseline_dir, feature_name)
        rows.append(
            {
                'target': target_name,
                'model_group': 'baseline',
                'feature_set': feature_name,
                'num_runs': len(corr_values),
                'median_corr': np.nanmedian(corr_values),
                'median_mae': np.nanmedian(mae_values),
            }
        )

    for feature_name in MERGED_FEATURES:
        corr_values, mae_values = load_metric_series(merged_dir, feature_name)
        rows.append(
            {
                'target': target_name,
                'model_group': 'merged',
                'feature_set': feature_name,
                'num_runs': len(corr_values),
                'median_corr': np.nanmedian(corr_values),
                'median_mae': np.nanmedian(mae_values),
            }
        )

    summary_df = pd.DataFrame(rows)
    baseline_df = summary_df[summary_df['model_group'] == 'baseline']
    best_baseline_idx = baseline_df['median_corr'].idxmax()
    best_baseline_feature = baseline_df.loc[best_baseline_idx, 'feature_set']
    best_baseline_corr = baseline_df.loc[best_baseline_idx, 'median_corr']
    best_baseline_mae = baseline_df.loc[best_baseline_idx, 'median_mae']
    summary_df['best_baseline_feature'] = best_baseline_feature
    summary_df['delta_corr_vs_best_baseline'] = summary_df['median_corr'] - best_baseline_corr
    summary_df['delta_mae_vs_best_baseline'] = summary_df['median_mae'] - best_baseline_mae
    return summary_df


def main():
    args = parse_args()
    targets = args.targets if args.targets else TASK_TARGETS[args.task]
    prediction_root = os.path.join(args.project_root, args.dataset, 'prediction')
    summary_frames = []
    for target_name in targets:
        target_dir = os.path.join(prediction_root, target_name)
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f'Target directory does not exist: {target_dir}')
        summary_frames.append(summarize_target(target_dir, target_name))

    summary_df = pd.concat(summary_frames, ignore_index=True)
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(
            prediction_root,
            f'feature_merge_summary_{args.task}.csv',
        )
    summary_df.to_csv(output_csv, index=False)
    print(summary_df.to_string(index=False))
    print(f'\nSaved summary to {output_csv}')


if __name__ == '__main__':
    main()
