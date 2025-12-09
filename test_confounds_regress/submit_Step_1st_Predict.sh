#!/bin/bash
#SBATCH --job-name=submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p q_fat_c
#SBATCH -q high_c
#SBATCH -o /ibmgpfs/cuizaixu_lab/congjing/WM_prediction/HCPD/code/4th_prediction/s02_prediction/nosmooth/job.%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/congjing/WM_prediction/HCPD/code/4th_prediction/s02_prediction/nosmooth/job.%j.error.txt

python /ibmgpfs/cuizaixu_lab/congjing/WM_prediction/HCPD/code/4th_prediction/s02_PLSprediction/nosmooth/Step_1st_Prediction_OverallPsyFactor_RandomCV_HCPD.py
