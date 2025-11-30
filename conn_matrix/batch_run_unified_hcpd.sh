#!/bin/bash
#SBATCH --job-name=unified_hcpd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=q_cn
#SBATCH --array=1-531%531            # TODO: 修改 100 为实际被试数量 (根据 sublist.txt 行数)
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/unified_hcpd/unified_hcpd_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/unified_hcpd/unified_hcpd_%A_%a.err

# ================= 配置区域 =================
PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction"

# 1. 设置被试列表文件路径 (HCPD)
SUBJ_LIST="${PROJECT_ROOT}/data/HCPD/table/sublist.txt"

# 2. 设置 Python 脚本路径 (使用统一脚本)
PYTHON_SCRIPT="${PROJECT_ROOT}/src/conn_matrix/process_dataset_unified.py"

# 3. 设置数据路径 (HCPD 路径)
DATASET_PATH="${PROJECT_ROOT}/data/HCPD"
OUTPUT_DIR="${PROJECT_ROOT}/data/HCPD/fc_matrix/individual_z"

# 图谱路径 (使用本地 reslice_atlases.py 生成的文件)
ATLAS_DIR="${PROJECT_ROOT}/data/atlas/resliced_hcpd"
GM_ATLAS="${ATLAS_DIR}/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz"
WM_ATLAS="${ATLAS_DIR}/rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz"

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

# 2. 运行 Python 统一脚本 (包含所有步骤：mask生成、FC计算、Fisher Z变换)
python3 ${PYTHON_SCRIPT} \
    --dataset_name HCPD \
    --subject_id ${subj} \
    --dataset_path ${DATASET_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --gm_atlas ${GM_ATLAS} \
    --wm_atlas ${WM_ATLAS}

echo "Finished subject: $subj"