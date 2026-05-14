import os 
import shutil

folder_path = "/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/ABCD/prediction"

targetStr = ["nihtbx_cryst_uncorrected", "nihtbx_fluidcomp_uncorrected", "nihtbx_totalcomp_uncorrected", "General", "Ext", "ADHD"]

for target in targetStr:
    split_index_file = os.path.join(folder_path, target, "V_holdout", "SharedSplitIndex.mat")
    pred_results_folder = os.path.join(folder_path, target, "V_holdout", "RegressCovariates_Holdout")

    if os.path.isfile(split_index_file):
        os.remove(split_index_file)
    
    if os.path.exists(pred_results_folder):
        shutil.rmtree(pred_results_folder)
    

