#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 20G
#SBATCH -p q_fat_c
#SBATCH --qos=high_c

set -euo pipefail

module load singularity

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <subject_id>" >&2
    exit 1
fi

raw_subj="$1"
if [[ "$raw_subj" == sub-* ]]; then
    subj="$raw_subj"
else
    subj="sub-$raw_subj"
fi
participant_label="${subj#sub-}"

PROJECT_ROOT_DEFAULT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"
PROJECT_ROOT="${WM_PREDICTION_ROOT:-$PROJECT_ROOT_DEFAULT}"
SCRIPT_DIR="$PROJECT_ROOT/src/preprocess/hcp_pipeline"
DATA_ROOT="$PROJECT_ROOT/data/EFNY"
HCP_STUDYFOLDER="$DATA_ROOT/hcp_studyfolder"
HCP_SUBJECT_DIR="$HCP_STUDYFOLDER/$subj"
RESULTS_ROOT="$HCP_SUBJECT_DIR/MNINonLinear/Results"

XCPD_ROOT="$DATA_ROOT/xcpd_hcp"
FMRIPREP_BRIDGE_ROOT="$XCPD_ROOT/fmriprep_bridge"
FMRIPREP_BRIDGE_SUBJECT="$FMRIPREP_BRIDGE_ROOT/$subj"
FMRIPREP_BRIDGE_FUNC="$FMRIPREP_BRIDGE_SUBJECT/func"
FMRIPREP_BRIDGE_ANAT="$FMRIPREP_BRIDGE_SUBJECT/anat"

CUSTOM_CP="$XCPD_ROOT/custom_confounds_csf_global_24p/$subj/func"
OUTPUT="$XCPD_ROOT/step_2nd_24PcsfGlobal"
WD="$XCPD_ROOT/step_2nd_wd/$subj"

FS_LICENSE_DIR=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/freesurfer
TEMPLATEFLOW=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/templateflow
XCPD_IMAGE=/ibmgpfs/cuizaixu_lab/xulongzhou/apps/singularity/xcpd-0.7.1rc5.simg

if [[ ! -d "$HCP_SUBJECT_DIR" ]]; then
    echo "Missing HCP subject directory: $HCP_SUBJECT_DIR" >&2
    exit 1
fi

rm -rf "$FMRIPREP_BRIDGE_SUBJECT" "$CUSTOM_CP" "$WD"
mkdir -p "$FMRIPREP_BRIDGE_FUNC" "$FMRIPREP_BRIDGE_ANAT" "$CUSTOM_CP" "$OUTPUT" "$WD"

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/etc/profile.d/conda.sh
conda activate ML

cat > "$FMRIPREP_BRIDGE_ROOT/dataset_description.json" <<EOF
{
  "Name": "EFNY HCP bridge for XCP-D",
  "BIDSVersion": "1.9.0",
  "DatasetType": "derivative",
  "GeneratedBy": [
    {
      "Name": "HCP",
      "Version": "5.0.0",
      "CodeURL": "https://github.com/Washington-University/HCPpipelines"
    }
  ]
}
EOF

link_file() {
    local src="$1"
    local dst="$2"
    if [[ ! -f "$src" ]]; then
        echo "Missing required file: $src" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    rm -f "$dst"
    ln "$src" "$dst"
}

write_identity_xfm() {
    local dst="$1"
    cat > "$dst" <<EOF
#Insight Transform File V1.0
#Transform 0
Transform: MatrixOffsetTransformBase_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
FixedParameters: 0 0 0
EOF
}

link_file \
    "$HCP_SUBJECT_DIR/MNINonLinear/T1w.nii.gz" \
    "$FMRIPREP_BRIDGE_ANAT/${subj}_desc-preproc_T1w.nii.gz"
link_file \
    "$HCP_SUBJECT_DIR/MNINonLinear/brainmask_fs.2.nii.gz" \
    "$FMRIPREP_BRIDGE_ANAT/${subj}_desc-brain_mask.nii.gz"
link_file \
    "$HCP_SUBJECT_DIR/MNINonLinear/ROIs/wmparc.2.nii.gz" \
    "$FMRIPREP_BRIDGE_ANAT/${subj}_dseg.nii.gz"
write_identity_xfm "$FMRIPREP_BRIDGE_ANAT/${subj}_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.txt"
write_identity_xfm "$FMRIPREP_BRIDGE_ANAT/${subj}_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.txt"

shopt -s nullglob
run_dirs=("$RESULTS_ROOT"/rfMRI_REST*_*)
shopt -u nullglob
if [[ ${#run_dirs[@]} -eq 0 ]]; then
    echo "No completed HCP fMRIVolume runs found under $RESULTS_ROOT" >&2
    exit 1
fi

for run_dir in "${run_dirs[@]}"; do
    run_name=$(basename "$run_dir")
    if [[ ! "$run_name" =~ ^rfMRI_REST([0-9]+)_[A-Z]+$ ]]; then
        echo "Unexpected run name: $run_name" >&2
        exit 1
    fi
    run_id="${BASH_REMATCH[1]}"
    prefix="${subj}_task-rest_run-${run_id}_space-MNI152NLin6Asym_res-2"

    bold_file="$run_dir/${run_name}.nii.gz"
    boldref_file="$run_dir/${run_name}_SBRef.nii.gz"
    mask_file="$run_dir/brainmask_fs.2.nii.gz"
    motion_file="$run_dir/Movement_Regressors.txt"
    rmsd_file="$run_dir/Movement_RelativeRMS.txt"

    link_file "$bold_file" "$FMRIPREP_BRIDGE_FUNC/${prefix}_desc-preproc_bold.nii.gz"
    link_file "$boldref_file" "$FMRIPREP_BRIDGE_FUNC/${prefix}_boldref.nii.gz"
    link_file "$mask_file" "$FMRIPREP_BRIDGE_FUNC/${prefix}_desc-brain_mask.nii.gz"

    python3 "$SCRIPT_DIR/extract_confounds_by_title.py" \
        --bold-file "$bold_file" \
        --motion-file "$motion_file" \
        --rmsd-file "$rmsd_file" \
        --seg-file "$HCP_SUBJECT_DIR/MNINonLinear/ROIs/wmparc.2.nii.gz" \
        --brain-mask-file "$mask_file" \
        --base-confounds-out "$FMRIPREP_BRIDGE_FUNC/${prefix}_desc-confounds_timeseries.tsv" \
        --base-confounds-json-out "$FMRIPREP_BRIDGE_FUNC/${prefix}_desc-confounds_timeseries.json" \
        --custom-confounds-out "$CUSTOM_CP/${subj}_task-rest_run-${run_id}_space-MNI152NLin6Asym_res-2_desc-confounds_timeseries.tsv" \
        --custom-confounds-json-out "$CUSTOM_CP/${subj}_task-rest_run-${run_id}_space-MNI152NLin6Asym_res-2_desc-confounds_timeseries.json" \
        --bold-json-out "$FMRIPREP_BRIDGE_FUNC/${prefix}_desc-preproc_bold.json" \
        --task-name rest
done

unset PYTHONPATH
export SINGULARITYENV_TEMPLATEFLOW_HOME="$TEMPLATEFLOW"
singularity run --cleanenv \
    -B "$FMRIPREP_BRIDGE_ROOT:/fmriprep" \
    -B "$OUTPUT:/output" \
    -B "$WD:/wd" \
    -B "$CUSTOM_CP:/custom_confounds" \
    -B "$FS_LICENSE_DIR:/fslic" \
    -B "$TEMPLATEFLOW:$TEMPLATEFLOW" \
    "$XCPD_IMAGE" \
    /fmriprep /output participant \
    --input-type fmriprep \
    --participant_label "$participant_label" --task-id rest \
    --fs-license-file /fslic/license.txt \
    -w /wd --nthreads 2 --mem-gb 40 \
    --nuisance-regressors 24P --despike -c /custom_confounds \
    --lower-bpf=0.01 --upper-bpf=0.1 \
    --smoothing 0 \
    --motion-filter-type lp --band-stop-min 6 \
    --skip-parcellation \
    --fd-thresh -1

rm -rf "$WD" "$FMRIPREP_BRIDGE_SUBJECT" "$CUSTOM_CP"
