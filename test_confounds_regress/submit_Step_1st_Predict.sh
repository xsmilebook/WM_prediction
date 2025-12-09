#!/bin/bash
#SBATCH --job-name=submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p q_fat_c
#SBATCH -q high_c
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/predict/HCPD/age_test/job.%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/log/predict/HCPD/age_test/job.%j.error.txt

python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/test_confounds_regress/Step_1st_Prediction_OverallPsyFactor_RandomCV_HCPD.py
