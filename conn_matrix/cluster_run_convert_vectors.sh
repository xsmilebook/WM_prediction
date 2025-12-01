#!/bin/bash
#SBATCH --job-name=convert_fc_vectors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=q_fat_c
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/convert_fc_vectors/convert_fc_vectors_%A_%a.out
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/conn_matrix/convert_fc_vectors/convert_fc_vectors_%A_%a.err

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

# Set up paths (adjust these based on your cluster environment)
PROJECT_ROOT="/ibmgpfs/cuizaixu_lab/xuhaoshu/WM_prediction"
INPUT_PATH="${PROJECT_ROOT}/data/ABCD/fc_matrix/individual"
SUBLIST_FILE="${PROJECT_ROOT}/data/ABCD/sublist.txt"
OUTPUT_PATH="${PROJECT_ROOT}/data/ABCD/fc_vector"
DATASET_NAME="ABCD"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"
mkdir -p logs

# Run the conversion script
echo "Starting FC matrix to vector conversion for ${DATASET_NAME}"
echo "Input path: ${INPUT_PATH}"
echo "Subject list: ${SUBLIST_FILE}"
echo "Output path: ${OUTPUT_PATH}"

python ${PROJECT_ROOT}/src/conn_matrix/convert_matrices_to_vectors.py \
    --input_path "${INPUT_PATH}" \
    --sublist_file "${SUBLIST_FILE}" \
    --output_path "${OUTPUT_PATH}" \
    --dataset_name "${DATASET_NAME}"

echo "FC matrix to vector conversion completed"