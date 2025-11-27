#!/bin/bash
#SBATCH --job-name=smooth_fmri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=q_fat
#SBATCH --array=1-4532%4532            # TODO: 修改 5 为实际被试数量 (根据 sublist.txt 行数)
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/generate_mask/generate_mask_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/generate_mask/generate_mask_%A_%a.err

# ================= 配置区域 =================
PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"

# 1. 设置被试列表文件路径 (以 ABCD 为例)
SUBJ_LIST="${PROJECT_ROOT}/data/ABCD/table/sublist.txt"

# 2. 设置 Python 脚本路径
PYTHON_SCRIPT="${PROJECT_ROOT}/src/conn_matrix/generate_mask.py"

# 3. 设置数据路径 (ABCD 路径)
# Anat/Dseg 所在路径
FMRIPREP_DIR="/ibmgpfs/cuizaixu_lab/congjing/WM_prediction/ABCD/data/bids"
# Bold 所在路径
XCPD_DIR="/ibmgpfs/cuizaixu_lab/congjing/WM_prediction/ABCD/data/step_2nd_24PcsfGlobal"
# 输出路径
OUTPUT_DIR="${PROJECT_ROOT}/data/ABCD/mri_data/wm_postproc"

# ================= 环境加载 =================
# 加载包含 nibabel 和 numpy 的 Python 环境
# env path: /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/envs/ML
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

# ================= 核心逻辑 =================

# 1. 获取当前任务对应的被试 ID
# sed -n "${SLURM_ARRAY_TASK_ID}p" 会提取文本中的第 N 行
subj=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${SUBJ_LIST})

# 检查是否成功获取 ID
if [ -z "$subj" ]; then
    echo "Error: No subject found for Array ID ${SLURM_ARRAY_TASK_ID} in ${SUBJ_LIST}"
    exit 1
fi

# 去除可能存在的 Windows 回车符 (\r)
subj=$(echo $subj | tr -d '\r')

echo "Job ID: $SLURM_JOB_ID, Array ID: $SLURM_ARRAY_TASK_ID"
echo "Processing subject: $subj"

# 2. 运行 Python 脚本
python3 ${PYTHON_SCRIPT} \
    --sub_id ${subj} \
    --fmriprep_dir ${FMRIPREP_DIR} \
    --xcpd_dir ${XCPD_DIR} \
    --output_dir ${OUTPUT_DIR}

echo "Finished subject: $subj"
