#!/bin/bash
#SBATCH --job-name=compute_fc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=q_fat
#SBATCH --array=1-5%600
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/compute_fc/compute_fc_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/compute_fc/compute_fc_%A_%a.err

# ================= 配置区域 =================
PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"

# 1. 设置被试列表文件路径
SUBJ_LIST="${PROJECT_ROOT}/data/ABCD/table/sublist.txt"

# 2. 设置 Python 脚本路径
PYTHON_SCRIPT="${PROJECT_ROOT}/src/conn_matrix/compute_individual_fc.py"

# 3. 设置输入输出路径
# 输入路径：generate_mask.py 的输出路径
INPUT_DIR="${PROJECT_ROOT}/data/ABCD/mri_data/wm_postproc"

# 图谱路径 (使用本地 reslice_atlases.py 生成的文件)
ATLAS_DIR="${PROJECT_ROOT}/data/atlas/resliced_abcd"
GM_ATLAS="${ATLAS_DIR}/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz"
WM_ATLAS="${ATLAS_DIR}/rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz"

# 输出路径 (输出为 .npy 格式)
OUTPUT_DIR="${PROJECT_ROOT}/data/ABCD/fc_matrix/individual"

# ================= 核心逻辑 =================

# 获取当前任务对应的 Subject ID
# sed -n "${SLURM_ARRAY_TASK_ID}p" 从列表中读取第 N 行
subj=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${SUBJ_LIST})

# 去除可能存在的 Windows 换行符 (\r)
subj=$(echo $subj | tr -d '\r')

echo "Processing subject: ${subj}"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"

# 创建日志目录 (如果不存在)
mkdir -p /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/compute_fc

# 运行 Python 脚本
python3 ${PYTHON_SCRIPT} \
    --sub_id ${subj} \
    --input_dir ${INPUT_DIR} \
    --gm_atlas ${GM_ATLAS} \
    --wm_atlas ${WM_ATLAS} \
    --output_dir ${OUTPUT_DIR}

echo "Finished subject: ${subj}"
