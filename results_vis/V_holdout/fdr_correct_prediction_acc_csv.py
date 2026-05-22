import argparse
import glob
import os

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/results/V_holdout/prediction_acc'
P_METRIC_KEYS = [
    'GG_empirical_p',
    'GW_empirical_p',
    'WW_empirical_p',
    'GW_partial_empirical_p',
    'WW_partial_empirical_p',
]
Q_METRIC_KEYS = [
    'GG_fdr_q',
    'GW_fdr_q',
    'WW_fdr_q',
    'GW_partial_fdr_q',
    'WW_partial_fdr_q',
]
Q_SIGNIFICANCE_KEYS = [
    'GG_fdr_significance',
    'GW_fdr_significance',
    'WW_fdr_significance',
    'GW_partial_fdr_significance',
    'WW_partial_fdr_significance',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply BH-FDR correction to prediction_acc holdout CSV files.'
    )
    parser.add_argument(
        '--input_dir',
        default=DEFAULT_INPUT_DIR,
        help='Directory containing prediction_acc CSV files.',
    )
    parser.add_argument(
        '--pattern',
        default='*.csv',
        help='Glob pattern used to find input CSV files inside --input_dir.',
    )
    return parser.parse_args()


def get_significance_label(p_value):
    if np.isnan(p_value):
        return np.nan
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def compute_bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size != 15:
        raise ValueError('Expected exactly 15 p-values, got {}.'.format(p_values.size))
    if np.isnan(p_values).any():
        raise ValueError('Found NaN in p-values; cannot run 15-test FDR correction.')

    order = np.argsort(p_values, kind='mergesort')
    ranked_p = p_values[order]
    n_tests = float(ranked_p.size)
    ranked_q = ranked_p * n_tests / np.arange(1.0, n_tests + 1.0)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)

    q_values = np.empty_like(ranked_q)
    q_values[order] = ranked_q
    return q_values


def update_csv(csv_path):
    data = pd.read_csv(csv_path)

    for q_key in Q_METRIC_KEYS:
        data[q_key] = pd.to_numeric(data[q_key], errors='coerce')
    for q_sig_key in Q_SIGNIFICANCE_KEYS:
        data[q_sig_key] = data[q_sig_key].astype(object)

    flat_p_values = []
    for _, row in data.iterrows():
        for p_key in P_METRIC_KEYS:
            flat_p_values.append(float(row[p_key]))

    flat_q_values = compute_bh_fdr(flat_p_values)

    q_index = 0
    for row_idx in range(len(data)):
        for q_key, q_sig_key in zip(Q_METRIC_KEYS, Q_SIGNIFICANCE_KEYS):
            q_value = float(flat_q_values[q_index])
            data.loc[row_idx, q_key] = q_value
            data.loc[row_idx, q_sig_key] = get_significance_label(q_value)
            q_index += 1

    data.to_csv(csv_path, index=False)
    return data


def main():
    args = parse_args()
    csv_paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not csv_paths:
        raise SystemExit('No CSV files found in {} matching {}.'.format(args.input_dir, args.pattern))

    for csv_path in csv_paths:
        updated = update_csv(csv_path)
        print(
            'Updated {} with BH-FDR q-values for {} rows ({} tests).'.format(
                csv_path,
                len(updated),
                len(updated) * len(P_METRIC_KEYS),
            )
        )


if __name__ == '__main__':
    main()
