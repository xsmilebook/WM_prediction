#!/bin/bash
#SBATCH --job-name=fisher_z
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-4532%4532
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/fisher_z/fisher_z_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/fisher_z/fisher_z_%A_%a.err

# ================= 配置区域 =================
PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"

# 1. 设置被试列表文件路径
SUBJ_LIST="${PROJECT_ROOT}/data/ABCD/table/sublist.txt"

# 2. 设置 Python 脚本路径
PYTHON_SCRIPT="${PROJECT_ROOT}/src/conn_matrix/apply_fisher_z.py"

# 3. 设置输入输出路径
# 输入路径：compute_individual_fc.py 的输出路径 (包含 .npy 文件)
INPUT_DIR="${PROJECT_ROOT}/data/ABCD/fc_matrix/individual"

# 输出路径 (输出为 .npy 格式)
OUTPUT_DIR="${PROJECT_ROOT}/data/ABCD/fc_matrix/individual_z"

# ================= 核心逻辑 =================

# 获取当前任务对应的 Subject ID
subj=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${SUBJ_LIST})

# 去除可能存在的 Windows 换行符 (\r)
subj=$(echo $subj | tr -d '\r')

echo "Processing subject: ${subj}"

# 创建日志目录
mkdir -p /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/fisher_z

# 运行 Python 脚本
python3 ${PYTHON_SCRIPT} \
    --sub_id ${subj} \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR}

echo "Finished subject: ${subj}"
