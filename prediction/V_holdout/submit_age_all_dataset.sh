#!/bin/bash
set -euo pipefail

PYTHON_SCRIPT="prediction/V_holdout/predict_age_RandomCV.py"
CONDA_INIT="/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate"

DATASETS=("EFNY" "HCPD" "CCNP" "PNC")
SEEDS=(
    42 50 73 101 118 137 149 163 177 191
    211 223 239 257 271 283 307 331 347 359
    373 389 401 419 433 449 463 479 491 503
    521 547 563 587 601 617 631 653 677 691
    709 733 751 769 787 811 829 853 877 907
)

source "${CONDA_INIT}"
conda activate ML

job_count=0
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running: python ${PYTHON_SCRIPT} --dataset ${dataset} --seed ${seed}"
        python "${PYTHON_SCRIPT}" --dataset "${dataset}" --seed "${seed}"
        job_count=$((job_count + 1))
    done
done

echo "Finished ${job_count} runs (${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds)."
