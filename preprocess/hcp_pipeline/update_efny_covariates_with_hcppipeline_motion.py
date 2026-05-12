import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def weighted_mean_fd_from_summary(row: pd.Series) -> float:
    total_frames = 0.0
    weighted_sum = 0.0
    for run_idx in range(1, 5):
        valid = row.get(f"rest{run_idx}_valid")
        frame = row.get(f"rest{run_idx}_frame")
        fd = row.get(f"rest{run_idx}_fd")

        if pd.isna(valid) or pd.isna(frame) or pd.isna(fd):
            continue
        if int(valid) != 1:
            continue
        frame = float(frame)
        fd = float(fd)
        if frame <= 0:
            continue

        total_frames += frame
        weighted_sum += frame * fd

    if total_frames == 0:
        return np.nan
    return weighted_sum / total_frames


def build_motion_update_table(motion_summary: pd.DataFrame) -> pd.DataFrame:
    motion_df = motion_summary.copy()
    motion_df["mean_FD_hcppipeline"] = motion_df.apply(weighted_mean_fd_from_summary, axis=1)
    motion_df["valid_num_hcppipeline"] = pd.to_numeric(motion_df["valid_num"], errors="coerce")
    motion_df["valid_subject_hcppipeline"] = pd.to_numeric(motion_df["valid_subject"], errors="coerce")
    keep_cols = [
        "subid",
        "mean_FD_hcppipeline",
        "valid_num_hcppipeline",
        "valid_subject_hcppipeline",
    ]
    return motion_df[keep_cols]


def update_covariates(base_covariates: pd.DataFrame, motion_update: pd.DataFrame) -> pd.DataFrame:
    result = base_covariates.copy()
    result = result.merge(motion_update, on="subid", how="left")

    result["mean_FD"] = result["mean_FD_hcppipeline"].where(
        ~result["mean_FD_hcppipeline"].isna(),
        result["mean_FD"],
    )
    return result[base_covariates.columns.tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-covariates",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/subid_meanFD_age_sex_new.csv",
    )
    parser.add_argument(
        "--motion-summary",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/rest_fd_summary_hcppipeline.csv",
    )
    parser.add_argument(
        "--out",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/subid_meanFD_age_sex_hcppipeline.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_covariates_path = Path(args.base_covariates)
    motion_summary_path = Path(args.motion_summary)
    output_path = Path(args.out)

    if not base_covariates_path.exists():
        raise FileNotFoundError(f"Base covariates file not found: {base_covariates_path}")
    if not motion_summary_path.exists():
        raise FileNotFoundError(f"Motion summary file not found: {motion_summary_path}")

    base_covariates = pd.read_csv(base_covariates_path)
    motion_summary = pd.read_csv(motion_summary_path)

    required_covariate_cols = {"subid", "age", "sex", "mean_FD"}
    if not required_covariate_cols.issubset(base_covariates.columns):
        missing = required_covariate_cols - set(base_covariates.columns)
        raise ValueError(f"Base covariates missing required columns: {sorted(missing)}")

    required_motion_cols = {"subid", "valid_num", "valid_subject"}
    if not required_motion_cols.issubset(motion_summary.columns):
        missing = required_motion_cols - set(motion_summary.columns)
        raise ValueError(f"Motion summary missing required columns: {sorted(missing)}")

    motion_update = build_motion_update_table(motion_summary)
    updated = update_covariates(base_covariates, motion_update)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(output_path, index=False)

    replaced_count = int(motion_update["mean_FD_hcppipeline"].notna().sum())
    fallback_count = int(len(updated) - replaced_count)
    print(f"Saved updated covariates to: {output_path}")
    print(f"Rows updated with hcppipeline motion: {replaced_count}")
    print(f"Rows kept from legacy motion: {fallback_count}")


if __name__ == "__main__":
    main()
