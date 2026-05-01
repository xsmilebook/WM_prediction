#!/bin/bash
#SBATCH --job-name=hcppipeline_fc
#SBATCH --cpus-per-task=1
#SBATCH --partition=q_fat,q_fat_c,q_fat_l
#SBATCH --array=1-505%200
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/hcppipeline_fc/hcppipeline_fc_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/hcppipeline_fc/hcppipeline_fc_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"
SUBJ_LIST="${1:-${PROJECT_ROOT}/data/EFNY/table/sublist_xcpd_ready505.txt}"
PYTHON_SCRIPT="${PROJECT_ROOT}/src/conn_matrix/efny_hcppipeline/run_subject_fc.py"
OUTPUT_ROOT="${PROJECT_ROOT}/data/EFNY/hcppipeline_fc"
LOG_DIR="${PROJECT_ROOT}/log/conn_matrix/hcppipeline_fc"

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

mkdir -p "$LOG_DIR"

if [[ ! -f "$SUBJ_LIST" ]]; then
    echo "Missing subject list: $SUBJ_LIST" >&2
    exit 1
fi

subj=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJ_LIST")
subj=$(echo "$subj" | tr -d '\r')

if [[ -z "$subj" ]]; then
    echo "Error: No subject found for Array ID ${SLURM_ARRAY_TASK_ID} in ${SUBJ_LIST}" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}, Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing subject: ${subj}"
echo "Subject list: ${SUBJ_LIST}"

python3 "$PYTHON_SCRIPT" \
    --subject_id "$subj" \
    --project_root "$PROJECT_ROOT" \
    --output_root "$OUTPUT_ROOT"

echo "Finished subject: ${subj}"
