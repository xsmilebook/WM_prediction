#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY_FOLDER_DEFAULT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder"

get_hcp_efny_batch_options() {
    local arguments=("$@")

    command_line_specified_study_folder=""
    command_line_specified_subject=""
    command_line_specified_session=""

    local index=0
    local num_args=${#arguments[@]}
    local argument

    while [[ $index -lt $num_args ]]; do
        argument=${arguments[index]}
        case "$argument" in
            --StudyFolder=*)
                command_line_specified_study_folder=${argument#*=}
                ;;
            --Subject=*)
                command_line_specified_subject=${argument#*=}
                ;;
            --Session=*)
                command_line_specified_session=${argument#*=}
                ;;
            *)
                echo "" >&2
                echo "ERROR: Unrecognized Option: ${argument}" >&2
                echo "" >&2
                exit 1
                ;;
        esac
        index=$((index + 1))
    done
}

setup_hcp_efny_batch_env() {
    source "$SCRIPT_DIR/hcp_efny_env.sh"
    if [[ -n "${HCPPIPEDEBUG:-}" ]]; then
        set -x
    fi
}

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

make_log_dir() {
    local study_folder="$1"
    local stage_name="$2"
    local id="$3"
    local run_name="${4:-}"
    local dir="$study_folder/logs/$stage_name/$id"
    if [[ -n "$run_name" ]]; then
        dir="$dir/$run_name"
    fi
    mkdir -p "$dir"
    printf '%s\n' "$dir"
}

run_with_queue() {
    local log_dir="$1"
    shift
    local queuing_command=("$HCPPIPEDIR/global/scripts/captureoutput.sh")

    (
        cd "$log_dir"
        "${queuing_command[@]}" "$@"
    )
}

build_t1_list() {
    local study_folder="$1"
    local session="$2"
    local list=""
    local found=0
    local folder
    for folder in "$study_folder/$session/unprocessed/3T"/T1w_MPR?; do
        [[ -d "$folder" ]] || continue
        local folderbase
        local image
        folderbase=$(basename "$folder")
        image=$(find_one "$folder/${session}_3T_${folderbase}.nii*" "T1w image $folderbase for $session")
        list+="${image}@"
        found=$((found + 1))
    done
    if (( found == 0 )); then
        echo "No T1w_MPR? folders found for $session under $study_folder" >&2
        exit 1
    fi
    printf '%s\n' "$list"
}

build_t2_list() {
    local study_folder="$1"
    local session="$2"
    local list=""
    local found=0
    local folder
    for folder in "$study_folder/$session/unprocessed/3T"/T2w_SPC?; do
        [[ -d "$folder" ]] || continue
        local folderbase
        local image
        folderbase=$(basename "$folder")
        image=$(find_one "$folder/${session}_3T_${folderbase}.nii*" "T2w image $folderbase for $session")
        list+="${image}@"
        found=$((found + 1))
    done
    if (( found == 0 )); then
        echo "No T2w_SPC? folders found for $session under $study_folder" >&2
        exit 1
    fi
    printf '%s\n' "$list"
}

collect_rest_runs() {
    local study_folder="$1"
    local subject="$2"
    find "$study_folder/$subject/unprocessed/3T" -maxdepth 1 -mindepth 1 -type d -name 'rfMRI_REST*_*' | sort -V
}

phase_to_unwarpdir() {
    local phase_encoding_dir="$1"
    case "$phase_encoding_dir" in
        PA)
            printf '%s\n' "y"
            ;;
        AP)
            printf '%s\n' "y-"
            ;;
        RL)
            printf '%s\n' "x"
            ;;
        LR)
            printf '%s\n' "x-"
            ;;
        *)
            echo "Unrecognized phase encoding direction: $phase_encoding_dir" >&2
            exit 1
            ;;
    esac
}

run_prefreesurfer_session() {
    local study_folder="$1"
    local session="$2"
    local t1w_input_images
    local t2w_input_images
    local fmap_ap
    local fmap_pa

    t1w_input_images=$(build_t1_list "$study_folder" "$session")
    t2w_input_images=$(build_t2_list "$study_folder" "$session")
    fmap_ap=$(find_one "$study_folder/$session/unprocessed/3T/T1w_MPR?/${session}_3T_SpinEchoFieldMap_AP.nii*" "AP structural spin echo fieldmap for $session")
    fmap_pa=$(find_one "$study_folder/$session/unprocessed/3T/T1w_MPR?/${session}_3T_SpinEchoFieldMap_PA.nii*" "PA structural spin echo fieldmap for $session")

    run_with_queue "$(make_log_dir "$study_folder" prefreesurfer "$session")" \
        "$HCPPIPEDIR/PreFreeSurfer/PreFreeSurferPipeline.sh" \
        --path="$study_folder" \
        --session="$session" \
        --t1="$t1w_input_images" \
        --t2="$t2w_input_images" \
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

run_freesurfer_session() {
    local study_folder="$1"
    local session="$2"

    run_with_queue "$(make_log_dir "$study_folder" freesurfer "$session")" \
        "$HCPPIPEDIR/FreeSurfer/FreeSurferPipeline.sh" \
        --session="$session" \
        --session-dir="$study_folder/$session/T1w" \
        --t1w-image="$study_folder/$session/T1w/T1w_acpc_dc_restore.nii.gz" \
        --t1w-brain="$study_folder/$session/T1w/T1w_acpc_dc_restore_brain.nii.gz" \
        --t2w-image="$study_folder/$session/T1w/T2w_acpc_dc_restore.nii.gz"
}

run_postfreesurfer_subject() {
    local study_folder="$1"
    local subject="$2"

    run_with_queue "$(make_log_dir "$study_folder" postfreesurfer "$subject")" \
        "$HCPPIPEDIR/PostFreeSurfer/PostFreeSurferPipeline.sh" \
        --study-folder="$study_folder" \
        --subject="$subject" \
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

run_fmrivolume_subject() {
    local study_folder="$1"
    local subject="$2"
    local run_dir

    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue

        local fmri_name
        local phase_encoding_dir
        local unwarp_dir
        local fmri_time_series
        local spin_echo_neg
        local spin_echo_pos

        fmri_name=$(basename "$run_dir")
        phase_encoding_dir="${fmri_name##*_}"
        unwarp_dir=$(phase_to_unwarpdir "$phase_encoding_dir")
        fmri_time_series=$(find_one "$run_dir/${subject}_3T_${fmri_name}.nii*" "fMRI time series for $subject $fmri_name")
        spin_echo_neg=$(find_one "$run_dir/${subject}_3T_SpinEchoFieldMap_AP.nii*" "negative polarity spin echo fieldmap for $subject $fmri_name")
        spin_echo_pos=$(find_one "$run_dir/${subject}_3T_SpinEchoFieldMap_PA.nii*" "positive polarity spin echo fieldmap for $subject $fmri_name")

        run_with_queue "$(make_log_dir "$study_folder" fmrivolume "$subject" "$fmri_name")" \
            "$HCPPIPEDIR/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh" \
            --studyfolder="$study_folder" \
            --subject="$subject" \
            --fmritcs="$fmri_time_series" \
            --fmriname="$fmri_name" \
            --fmrires=2 \
            --biascorrection=SEBASED \
            --fmriscout=NONE \
            --mctype=MCFLIRT \
            --gdcoeffs=NONE \
            --dcmethod=TOPUP \
            --echospacing=0.000269996 \
            --unwarpdir="$unwarp_dir" \
            --SEPhaseNeg="$spin_echo_neg" \
            --SEPhasePos="$spin_echo_pos" \
            --topupconfig="$HCPPIPEDIR_Config/b02b0.cnf" \
            --fmapmag=NONE \
            --fmapphase=NONE \
            --echodiff=NONE \
            --fmapcombined=NONE
    done < <(collect_rest_runs "$study_folder" "$subject")
}

run_fmrisurface_subject() {
    local study_folder="$1"
    local subject="$2"
    local run_dir

    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        local fmri_name
        fmri_name=$(basename "$run_dir")

        run_with_queue "$(make_log_dir "$study_folder" fmrisurface "$subject" "$fmri_name")" \
            "$HCPPIPEDIR/fMRISurface/GenericfMRISurfaceProcessingPipeline.sh" \
            --path="$study_folder" \
            --subject="$subject" \
            --fmriname="$fmri_name" \
            --lowresmesh=32 \
            --fmrires=2 \
            --smoothingFWHM=2 \
            --grayordinatesres=2 \
            --regname=MSMSulc
    done < <(collect_rest_runs "$study_folder" "$subject")
}
