#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
STUDY_FOLDER_DEFAULT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder"

usage() {
    cat <<'EOF'
Usage:
  run_hcp_efny_stage.sh --stage <stage> [--subject <id> | --subject-list <txt>] [options]

Stages:
  prefreesurfer
  freesurfer
  postfreesurfer
  fmrivolume
  fmrisurface

Options:
  --study-folder <path>    Default: /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder
  --subject <id>           Process a single subject
  --subject-list <txt>     Process subjects listed in a text file
  --runlocal               Accepted for compatibility; execution is always direct
  --print-sbatch           Print an sbatch example and exit
  --partition <name>       Used only with --print-sbatch
  --cpus <n>               Used only with --print-sbatch
  --mem <size>             Used only with --print-sbatch
  --time <hh:mm:ss>        Used only with --print-sbatch
EOF
}

stage=""
study_folder="$STUDY_FOLDER_DEFAULT"
subject=""
subject_list=""
runlocal="false"
print_sbatch="false"
partition=""
cpus=""
mem=""
time_limit=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            stage="$2"
            shift 2
            ;;
        --study-folder)
            study_folder="$2"
            shift 2
            ;;
        --subject)
            subject="$2"
            shift 2
            ;;
        --subject-list)
            subject_list="$2"
            shift 2
            ;;
        --runlocal)
            runlocal="true"
            shift
            ;;
        --print-sbatch)
            print_sbatch="true"
            shift
            ;;
        --partition)
            partition="$2"
            shift 2
            ;;
        --cpus)
            cpus="$2"
            shift 2
            ;;
        --mem)
            mem="$2"
            shift 2
            ;;
        --time)
            time_limit="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$stage" ]]; then
    echo "--stage is required" >&2
    usage >&2
    exit 1
fi

case "$stage" in
    prefreesurfer|freesurfer|postfreesurfer|fmrivolume|fmrisurface)
        ;;
    *)
        echo "Unsupported stage: $stage" >&2
        exit 1
        ;;
esac

if [[ -n "$subject" && -n "$subject_list" ]]; then
    echo "Use either --subject or --subject-list, not both" >&2
    exit 1
fi

if [[ -z "$subject" && -z "$subject_list" ]]; then
    echo "Either --subject or --subject-list is required" >&2
    exit 1
fi

read_subjects() {
    if [[ -n "$subject" ]]; then
        printf '%s\n' "$subject"
        return 0
    fi

    mapfile -t subjects_from_file < <(grep -v '^[[:space:]]*$' "$subject_list")
    if [[ ${#subjects_from_file[@]} -eq 0 ]]; then
        echo "Subject list is empty: $subject_list" >&2
        exit 1
    fi

    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        local idx=$((SLURM_ARRAY_TASK_ID - 1))
        if (( idx < 0 || idx >= ${#subjects_from_file[@]} )); then
            echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for $subject_list" >&2
            exit 1
        fi
        printf '%s\n' "${subjects_from_file[$idx]}"
        return 0
    fi

    printf '%s\n' "${subjects_from_file[@]}"
}

print_sbatch_example() {
    if [[ -z "$subject_list" ]]; then
        echo "--print-sbatch requires --subject-list" >&2
        exit 1
    fi
    if [[ -z "$partition" || -z "$cpus" || -z "$mem" || -z "$time_limit" ]]; then
        echo "--print-sbatch requires --partition, --cpus, --mem, and --time" >&2
        exit 1
    fi

    local total
    total=$(grep -vc '^[[:space:]]*$' "$subject_list")
    local abs_script
    abs_script=$(readlink -f "$0")
    local log_dir="$study_folder/logs/slurm/$stage"
    mkdir -p "$log_dir"
    printf 'sbatch --job-name=hcp_%s --partition=%s --cpus-per-task=%s --mem=%s --time=%s --array=1-%s \\\n' "$stage" "$partition" "$cpus" "$mem" "$time_limit" "$total"
    printf '  --output=%s/%%A_%%a.out --error=%s/%%A_%%a.err \\\n' "$log_dir" "$log_dir"
    printf "  --wrap='bash %s --stage %s --study-folder %s --subject-list %s'\n" "$abs_script" "$stage" "$study_folder" "$subject_list"
}

if [[ "$print_sbatch" == "true" ]]; then
    print_sbatch_example
    exit 0
fi

source "$SCRIPT_DIR/hcp_efny_env.sh"

find_one() {
    local pattern="$1"
    local description="$2"
    shopt -s nullglob
    local matches=($pattern)
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "Missing required $description: $pattern" >&2
        exit 1
    fi
    printf '%s\n' "${matches[0]}"
}

run_logged() {
    local log_dir="$1"
    shift
    mkdir -p "$log_dir"
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    local stdout_log="$log_dir/stdout_${stamp}.log"
    local stderr_log="$log_dir/stderr_${stamp}.log"

    echo "Logging stdout to $stdout_log"
    echo "Logging stderr to $stderr_log"
    (
        cd "$log_dir"
        "$@"
    ) > >(tee "$stdout_log") 2> >(tee "$stderr_log" >&2)
}

run_prefreesurfer() {
    local sub="$1"
    local base="$study_folder/$sub/unprocessed/3T"
    local t1=$(find_one "$base/T1w_MPR1/${sub}_3T_T1w_MPR1.nii*" "staged T1w for $sub")
    local t2=$(find_one "$base/T2w_SPC1/${sub}_3T_T2w_SPC1.nii*" "staged T2w for $sub")
    local fmap_ap=$(find_one "$base/T1w_MPR1/${sub}_3T_SpinEchoFieldMap_AP.nii*" "staged AP fmap for $sub")
    local fmap_pa=$(find_one "$base/T1w_MPR1/${sub}_3T_SpinEchoFieldMap_PA.nii*" "staged PA fmap for $sub")

    run_logged "$study_folder/logs/prefreesurfer/$sub" \
        "$HCPPIPEDIR/PreFreeSurfer/PreFreeSurferPipeline.sh" \
        --path="$study_folder" \
        --session="$sub" \
        --t1="${t1}@" \
        --t2="${t2}@" \
        --t1template="$HCPPIPEDIR_Templates/MNI152_T1_0.7mm.nii.gz" \
        --t1templatebrain="$HCPPIPEDIR_Templates/MNI152_T1_0.7mm_brain.nii.gz" \
        --t1template2mm="$HCPPIPEDIR_Templates/MNI152_T1_2mm.nii.gz" \
        --t2template="$HCPPIPEDIR_Templates/MNI152_T2_0.7mm.nii.gz" \
        --t2templatebrain="$HCPPIPEDIR_Templates/MNI152_T2_0.7mm_brain.nii.gz" \
        --t2template2mm="$HCPPIPEDIR_Templates/MNI152_T2_2mm.nii.gz" \
        --templatemask="$HCPPIPEDIR_Templates/MNI152_T1_0.7mm_brain_mask.nii.gz" \
        --template2mmmask="$HCPPIPEDIR_Templates/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
        --brainsize=150 \
        --fnirtconfig="$HCPPIPEDIR_Config/T1_2_MNI152_2mm.cnf" \
        --fmapmag=NONE \
        --fmapphase=NONE \
        --fmapcombined=NONE \
        --echodiff=NONE \
        --SEPhaseNeg="$fmap_ap" \
        --SEPhasePos="$fmap_pa" \
        --seechospacing=0.000530007 \
        --seunwarpdir=j \
        --t1samplespacing=5.2e-06 \
        --t2samplespacing=2.6e-06 \
        --unwarpdir=z \
        --gdcoeffs=NONE \
        --avgrdcmethod=TOPUP \
        --topupconfig="$HCPPIPEDIR_Config/b02b0.cnf"
}

run_freesurfer() {
    local sub="$1"
    run_logged "$study_folder/logs/freesurfer/$sub" \
        "$HCPPIPEDIR/FreeSurfer/FreeSurferPipeline.sh" \
        --session="$sub" \
        --session-dir="$study_folder/$sub/T1w" \
        --t1w-image="$study_folder/$sub/T1w/T1w_acpc_dc_restore.nii.gz" \
        --t1w-brain="$study_folder/$sub/T1w/T1w_acpc_dc_restore_brain.nii.gz" \
        --t2w-image="$study_folder/$sub/T1w/T2w_acpc_dc_restore.nii.gz"
}

run_postfreesurfer() {
    local sub="$1"
    run_logged "$study_folder/logs/postfreesurfer/$sub" \
        "$HCPPIPEDIR/PostFreeSurfer/PostFreeSurferPipeline.sh" \
        --study-folder="$study_folder" \
        --subject="$sub" \
        --surfatlasdir="$HCPPIPEDIR_Templates/standard_mesh_atlases" \
        --grayordinatesdir="$HCPPIPEDIR_Templates/91282_Greyordinates" \
        --grayordinatesres=2 \
        --hiresmesh=164 \
        --lowresmesh=32 \
        --subcortgraylabels="$HCPPIPEDIR_Config/FreeSurferSubcorticalLabelTableLut.txt" \
        --freesurferlabels="$HCPPIPEDIR_Config/FreeSurferAllLut.txt" \
        --refmyelinmaps="$HCPPIPEDIR_Templates/standard_mesh_atlases/Conte69.MyelinMap_BC.164k_fs_LR.dscalar.nii" \
        --regname=MSMSulc \
        --use-ind-mean=YES
}

collect_rest_runs() {
    local sub="$1"
    find "$study_folder/$sub/unprocessed/3T" -maxdepth 1 -type d -name 'rfMRI_REST*_*' | sort -V
}

run_fmrivolume() {
    local sub="$1"
    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        local run_name
        run_name=$(basename "$run_dir")
        local bold=$(find_one "$run_dir/${sub}_3T_${run_name}.nii*" "staged bold for $sub $run_name")
        local fmap_ap=$(find_one "$run_dir/${sub}_3T_SpinEchoFieldMap_AP.nii*" "staged AP fmap for $sub $run_name")
        local fmap_pa=$(find_one "$run_dir/${sub}_3T_SpinEchoFieldMap_PA.nii*" "staged PA fmap for $sub $run_name")

        run_logged "$study_folder/logs/fmrivolume/$sub/$run_name" \
            "$HCPPIPEDIR/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh" \
            --path="$study_folder" \
            --subject="$sub" \
            --fmriname="$run_name" \
            --fmritcs="$bold" \
            --fmriscout=NONE \
            --SEPhaseNeg="$fmap_ap" \
            --SEPhasePos="$fmap_pa" \
            --fmapmag=NONE \
            --fmapphase=NONE \
            --fmapcombined=NONE \
            --echospacing=0.000269996 \
            --echodiff=NONE \
            --unwarpdir=y \
            --fmrires=2 \
            --dcmethod=TOPUP \
            --gdcoeffs=NONE \
            --topupconfig="$HCPPIPEDIR_Config/b02b0.cnf" \
            --biascorrection=SEBASED \
            --mctype=MCFLIRT
    done < <(collect_rest_runs "$sub")
}

run_fmrisurface() {
    local sub="$1"
    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        local run_name
        run_name=$(basename "$run_dir")
        run_logged "$study_folder/logs/fmrisurface/$sub/$run_name" \
            "$HCPPIPEDIR/fMRISurface/GenericfMRISurfaceProcessingPipeline.sh" \
            --path="$study_folder" \
            --subject="$sub" \
            --fmriname="$run_name" \
            --lowresmesh=32 \
            --fmrires=2 \
            --smoothingFWHM=2 \
            --grayordinatesres=2 \
            --regname=MSMSulc
    done < <(collect_rest_runs "$sub")
}

if [[ "$runlocal" == "true" ]]; then
    echo "--runlocal is accepted for compatibility; execution is direct in all cases."
fi

mapfile -t subjects_to_run < <(read_subjects)

for current_subject in "${subjects_to_run[@]}"; do
    case "$stage" in
        prefreesurfer)
            run_prefreesurfer "$current_subject"
            ;;
        freesurfer)
            run_freesurfer "$current_subject"
            ;;
        postfreesurfer)
            run_postfreesurfer "$current_subject"
            ;;
        fmrivolume)
            run_fmrivolume "$current_subject"
            ;;
        fmrisurface)
            run_fmrisurface "$current_subject"
            ;;
    esac
done
