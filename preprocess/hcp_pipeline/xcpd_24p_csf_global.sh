#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 20G
#SBATCH -p q_fat_c
#SBATCH --qos=high_c

module load singularity
subj=$1

#export TMPDIR='/ibmgpfs/cuizaixu_lab/zhaoshaoling/MSC_data/PNC/code/xcpd/tmp'
#echo "Running with TMPDIR: $TMPDIR"

fmriprep_Path=/ibmgpfs/cuizaixu_lab/xuhaoshu/PNC/fmriprep
xcpd_Path=/home/cuizaixu_lab/xuhaoshu/DATA_C/PNC/xcpd

fslic=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/freesurfer
templateflow=/ibmgpfs/cuizaixu_lab/xulongzhou/tool/templateflow

# construct custom confounds file
customCP=/home/cuizaixu_lab/xuhaoshu/DATA_C/PNC/xcpd/custom_confounds_csf_global_24p
mkdir -p $customCP
module load MATLAB/R2019a
#filePath=${fmriprep_Path}/sub-${subj}/ses-PNC1/func/sub-${subj}_ses-PNC1_task-rest_acq-singleband_desc-confounds_timeseries.tsv
#filePath=${fmriprep_Path}/sub-${subj}/ses-PNC1/func/sub-${subj}_ses-PNC1_task-rest_acq-*_desc-confounds_timeseries.tsv
#### some filePaths without fmap have name with 'VARIANTNoFmap' string
filePath=$(find "${fmriprep_Path}/sub-${subj}/ses-PNC1/func/" -type f -name "sub-${subj}_ses-PNC1_task-rest_acq-*_desc-confounds_timeseries.tsv")
fileName=$(basename "$filePath")
#savePath_customCP=${customCP}/sub-${subj}_ses-PNC1_task-rest_acq-singleband_desc-confounds_timeseries.tsv
savePath_customCP=${customCP}/$fileName
rm -rf ${savePath_customCP}

matlab -nodisplay -nosplash -nodesktop -r \
   "extract_confounds_by_title('${filePath}', '${savePath_customCP}', {'csf','global_signal','csf_derivative1', 'global_signal_derivative1','csf_power2','global_signal_power2','csf_derivative1_power2','global_signal_derivative1_power2'}); exit;"

# running xcpd
output=${xcpd_Path}/step_2nd_24PcsfGlobal
mkdir -p ${output}
wd=${xcpd_Path}/step_2nd_wd/sub-${subj}
mkdir -p ${wd}

unset PYTHONPATH
export SINGULARITYENV_TEMPLATEFLOW_HOME=$templateflow
singularity run --cleanenv \
        -B $fmriprep_Path:/fmriprep \
        -B $output:/output \
        -B $wd:/wd \
        -B $customCP:/custom_confounds \
        -B $fslic:/fslic \
        -B $templateflow:$templateflow \
        /ibmgpfs/cuizaixu_lab/xulongzhou/apps/singularity/xcpd-0.7.1rc5.simg \
        /fmriprep /output participant \
        --participant_label ${subj} --task-id rest \
        --fs-license-file /fslic/license.txt \
        -w /wd --nthreads 2 --mem-gb 40 \
        --nuisance-regressors 24P --despike -c /custom_confounds \
        --lower-bpf=0.01 --upper-bpf=0.1 \
        --smoothing 6 \
        --motion-filter-type lp --band-stop-min 6 \
        --skip-parcellation \
        --fd-thresh -1

rm -rf $wd