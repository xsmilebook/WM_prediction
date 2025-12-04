#!/bin/bash
#SBATCH --job-name=submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -p q_fat_c
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/predict/PNC/age/job.%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/predict/PNC/age/job.%j.error.txt

source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

# python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/predict_cognition_RandomCV.py
# python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/predict_pfactor_RandomCV.py
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/predict_pfactor_RandomCV.py