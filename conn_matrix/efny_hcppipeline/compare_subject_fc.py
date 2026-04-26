#!/usr/bin/env python3
"""Compare EFNY HCP-pipeline FC matrices against the legacy EFNY FC outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_matrix(root: Path, subject_id: str, kind: str) -> np.ndarray:
    return np.load(root / subject_id / f"{subject_id}_{kind}_FC_Z.npy")


def vectorize_for_compare(matrix: np.ndarray, kind: str) -> np.ndarray:
    if kind in {"GG", "WW"}:
        return matrix[np.tril_indices(matrix.shape[0], k=-1)]
    return matrix.reshape(-1)


def save_matrix_plot(old_matrix: np.ndarray, new_matrix: np.ndarray, out_path: Path, title: str) -> None:
    diff_matrix = new_matrix - old_matrix
    vmax = float(np.nanmax(np.abs(np.concatenate([old_matrix.ravel(), new_matrix.ravel()]))))
    diff_vmax = float(np.nanmax(np.abs(diff_matrix)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(old_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Legacy EFNY")
    axes[1].imshow(new_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("HCP pipeline")
    im2 = axes[2].imshow(diff_matrix, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
    axes[2].set_title("Difference")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.colorbar(im0, ax=axes[:2], shrink=0.8, location="right")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, location="right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_compare(subject_id: str, legacy_root: Path, new_root: Path, output_root: Path) -> None:
    rows = ["matrix_type\tvectorization\tpearson_r\tn_elements"]

    for kind in ("GG", "GW", "WW"):
        old_matrix = load_matrix(legacy_root, subject_id, kind)
        new_matrix = load_matrix(new_root, subject_id, kind)
        if old_matrix.shape != new_matrix.shape:
            raise ValueError(f"{kind} shape mismatch: legacy={old_matrix.shape}, new={new_matrix.shape}")

        vector_old = vectorize_for_compare(old_matrix, kind)
        vector_new = vectorize_for_compare(new_matrix, kind)
        corr = float(np.corrcoef(vector_old, vector_new)[0, 1])
        vectorization = "lower_triangle" if kind in {"GG", "WW"} else "full_matrix"
        rows.append(f"{kind}\t{vectorization}\t{corr:.10f}\t{vector_old.size}")

        plot_path = output_root / subject_id / f"{subject_id}_{kind}_FC_Z_compare.png"
        save_matrix_plot(old_matrix, new_matrix, plot_path, f"{subject_id} {kind} FC (Z)")

    summary_path = output_root / subject_id / f"{subject_id}_fc_correlation_summary.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HCP-pipeline FC matrices against legacy EFNY FC.")
    parser.add_argument("--subject_id", required=True, help="Subject ID")
    parser.add_argument(
        "--legacy_root",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/fc_matrix/individual_z",
        help="Legacy EFNY individual_z root",
    )
    parser.add_argument(
        "--new_root",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcppipeline_fc/fc_matrix/individual_z",
        help="New HCP-pipeline individual_z root",
    )
    parser.add_argument(
        "--output_root",
        default="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcppipeline_fc/comparison",
        help="Comparison output root",
    )
    args = parser.parse_args()

    run_compare(args.subject_id, Path(args.legacy_root), Path(args.new_root), Path(args.output_root))


if __name__ == "__main__":
    main()
