#!/usr/bin/env bash
set -euo pipefail

STUDY_FOLDER_DEFAULT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder"

usage() {
    cat <<'EOF'
Usage:
  bash preprocess/hcp_pipeline/cleanup_hcp_stage_outputs.sh <stage> <subject_list> [study_folder] [--dry-run]

Arguments:
  <stage>         prefreesurfer | freesurfer | postfreesurfer | fmrivolume | fmrisurface
  <subject_list>  Text file with one subject ID per line
  [study_folder]  Default: /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder
  [--dry-run]     Print matched paths without deleting them

Notes:
  - The script deletes outputs for the selected stage together with downstream outputs that
    should be rebuilt to avoid mixing old and new files.
  - For fmrisurface, HCP stores final run-level outputs together with volume-stage outputs
    under MNINonLinear/Results, so this cleanup removes that directory and requires rerunning
    fMRIVolume before rerunning fMRISurface.
EOF
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
    usage >&2
    exit 1
fi

stage="$1"
subject_list="$2"
study_folder="$STUDY_FOLDER_DEFAULT"
dry_run=0

shift 2
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            dry_run=1
            ;;
        *)
            if [[ "$study_folder" != "$STUDY_FOLDER_DEFAULT" ]]; then
                echo "Unrecognized extra argument: $arg" >&2
                usage >&2
                exit 1
            fi
            study_folder="$arg"
            ;;
    esac
done

case "$stage" in
    prefreesurfer|freesurfer|postfreesurfer|fmrivolume|fmrisurface)
        ;;
    *)
        echo "Unsupported stage: $stage" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ ! -f "$subject_list" ]]; then
    echo "Subject list not found: $subject_list" >&2
    exit 1
fi

remove_path() {
    local path="$1"
    if [[ -e "$path" ]]; then
        printf 'remove\t%s\n' "$path"
        if (( dry_run == 0 )); then
            rm -rf "$path"
        fi
    else
        printf 'skip_missing\t%s\n' "$path"
    fi
}

cleanup_subject_stage() {
    local current_stage="$1"
    local subject="$2"
    local subject_root="$study_folder/$subject"

    case "$current_stage" in
        prefreesurfer)
            remove_path "$subject_root/T1w"
            remove_path "$subject_root/MNINonLinear"
            ;;
        freesurfer)
            remove_path "$subject_root/T1w/$subject"
            remove_path "$subject_root/T1w/Native"
            remove_path "$subject_root/T1w/fsaverage_LR32k"
            remove_path "$subject_root/T1w/Results"
            remove_path "$subject_root/MNINonLinear"
            ;;
        postfreesurfer)
            remove_path "$subject_root/T1w/Native"
            remove_path "$subject_root/T1w/fsaverage_LR32k"
            remove_path "$subject_root/T1w/Results"
            remove_path "$subject_root/MNINonLinear"
            ;;
        fmrivolume)
            remove_path "$subject_root/T1w/Results"
            remove_path "$subject_root/MNINonLinear/Results"
            ;;
        fmrisurface)
            remove_path "$subject_root/MNINonLinear/Results"
            ;;
    esac
}

while IFS= read -r raw_subject || [[ -n "$raw_subject" ]]; do
    subject=$(printf '%s' "$raw_subject" | tr -d '\r')
    [[ -n "$subject" ]] || continue
    printf 'subject\t%s\t%s\n' "$stage" "$subject"
    cleanup_subject_stage "$stage" "$subject"
done < "$subject_list"

if (( dry_run == 1 )); then
    echo "Dry run only; no files were deleted."
fi
