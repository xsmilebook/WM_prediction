#!/usr/bin/env bash
#SBATCH --job-name=hcp_stage
#SBATCH --partition=q_fat_c
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder/logs/slurm/%x_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder/logs/slurm/%x_%A_%a.err
#SBATCH --qos=high_c

set -euo pipefail

SCRIPT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/preprocess/hcp_pipeline"
STUDY_FOLDER_DEFAULT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder"

usage() {
    cat <<'EOF'
Usage:
  sbatch --array=1-N [resources...] preprocess/hcp_pipeline/submit_hcp_efny_stage.slurm.sh \
    <stage> <subject_list> [study_folder]

Arguments:
  <stage>         prefreesurfer | freesurfer | postfreesurfer | fmrivolume | fmrisurface
  <subject_list>  Text file with one subject ID per line
  [study_folder]  Default: /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder

Examples:
  sbatch --cpus-per-task=2 --array=1-1 submit_hcp_efny_stage.slurm.sh prefreesurfer /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/sublist_test.txt

  sbatch --cpus-per-task=8 --array=1-10 \
    preprocess/hcp_pipeline/submit_hcp_efny_stage.slurm.sh freesurfer /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/sublist.txt
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 1
fi

stage="$1"
subject_list="$2"
study_folder="${3:-$STUDY_FOLDER_DEFAULT}"

subject_arg_name=""
stage_script=""

case "$stage" in
    prefreesurfer)
        stage_script="$SCRIPT_DIR/PreFreeSurferPipelineBatch.sh"
        subject_arg_name="--Session"
        ;;
    freesurfer)
        stage_script="$SCRIPT_DIR/FreeSurferPipelineBatch.sh"
        subject_arg_name="--Session"
        ;;
    postfreesurfer)
        stage_script="$SCRIPT_DIR/PostFreeSurferPipelineBatch.sh"
        subject_arg_name="--Subject"
        ;;
    fmrivolume)
        stage_script="$SCRIPT_DIR/GenericfMRIVolumeProcessingPipelineBatch.sh"
        subject_arg_name="--Subject"
        ;;
    fmrisurface)
        stage_script="$SCRIPT_DIR/GenericfMRISurfaceProcessingPipelineBatch.sh"
        subject_arg_name="--Subject"
        ;;
    *)
        echo "Unsupported stage: $stage" >&2
        exit 1
        ;;
esac

if [[ ! -f "$subject_list" ]]; then
    echo "Subject list not found: $subject_list" >&2
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=1-N." >&2
    exit 1
fi

subject=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$subject_list" | tr -d '\r')
if [[ -z "$subject" ]]; then
    echo "No subject found at line $SLURM_ARRAY_TASK_ID in $subject_list" >&2
    exit 1
fi

mkdir -p "$study_folder/logs/slurm"

exec bash "$stage_script" \
    --StudyFolder="$study_folder" \
    "${subject_arg_name}=$subject"
