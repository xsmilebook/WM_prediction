#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE_SCRIPT="$SCRIPT_DIR/submit_hcp_efny_stage.slurm.sh"
STUDY_FOLDER_DEFAULT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder"

usage() {
    cat <<'EOF'
Usage:
  bash preprocess/hcp_pipeline/submit_hcp_efny_chain.sh <subject_list> [study_folder]

Arguments:
  <subject_list>  Text file with one subject ID per line
  [study_folder]  Default: /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder

Environment variables:
  PREFREESURFER_SBATCH_ARGS   Extra sbatch args for prefreesurfer
  FREESURFER_SBATCH_ARGS      Extra sbatch args for freesurfer
  POSTFREESURFER_SBATCH_ARGS  Extra sbatch args for postfreesurfer
  FMRIVOLUME_SBATCH_ARGS      Extra sbatch args for fmrivolume

Example:
  PREFREESURFER_SBATCH_ARGS="--cpus-per-task=2" \
  FREESURFER_SBATCH_ARGS="--cpus-per-task=4" \
  POSTFREESURFER_SBATCH_ARGS="--cpus-per-task=4" \
  FMRIVOLUME_SBATCH_ARGS="--cpus-per-task=4" \
  bash preprocess/hcp_pipeline/submit_hcp_efny_chain.sh \
    /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/hcp_pipeline/reprocess.txt
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 1
fi

subject_list="$1"
study_folder="${2:-$STUDY_FOLDER_DEFAULT}"

if [[ ! -f "$subject_list" ]]; then
    echo "Subject list not found: $subject_list" >&2
    exit 1
fi

if [[ ! -f "$STAGE_SCRIPT" ]]; then
    echo "Stage submission script not found: $STAGE_SCRIPT" >&2
    exit 1
fi

subject_count=$(grep -cve '^[[:space:]]*$' "$subject_list")
if [[ "$subject_count" -lt 1 ]]; then
    echo "Subject list is empty: $subject_list" >&2
    exit 1
fi

submit_stage() {
    local stage="$1"
    local dependency_jobid="${2:-}"
    local extra_args_var="$3"
    local extra_args_string="${!extra_args_var:-}"
    local -a sbatch_cmd=(sbatch --parsable "--array=1-${subject_count}")

    if [[ -n "$dependency_jobid" ]]; then
        sbatch_cmd+=(--dependency="afterok:${dependency_jobid}")
    fi

    if [[ -n "$extra_args_string" ]]; then
        # shellcheck disable=SC2206
        local -a extra_args=( $extra_args_string )
        sbatch_cmd+=("${extra_args[@]}")
    fi

    sbatch_cmd+=("$STAGE_SCRIPT" "$stage" "$subject_list" "$study_folder")

    local jobid
    jobid=$("${sbatch_cmd[@]}")
    printf '%s\n' "$jobid"
}

prefreesurfer_jobid=$(submit_stage prefreesurfer "" PREFREESURFER_SBATCH_ARGS)
echo "Submitted prefreesurfer: ${prefreesurfer_jobid}"

freesurfer_jobid=$(submit_stage freesurfer "$prefreesurfer_jobid" FREESURFER_SBATCH_ARGS)
echo "Submitted freesurfer: ${freesurfer_jobid} (afterok:${prefreesurfer_jobid})"

postfreesurfer_jobid=$(submit_stage postfreesurfer "$freesurfer_jobid" POSTFREESURFER_SBATCH_ARGS)
echo "Submitted postfreesurfer: ${postfreesurfer_jobid} (afterok:${freesurfer_jobid})"

fmrivolume_jobid=$(submit_stage fmrivolume "$postfreesurfer_jobid" FMRIVOLUME_SBATCH_ARGS)
echo "Submitted fmrivolume: ${fmrivolume_jobid} (afterok:${postfreesurfer_jobid})"

echo "Chain submission complete."
