import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import warnings

# --- 1. Helper Function: Partial Correlation ---
def partial_corr(x, y, z):
    """
    Calculate partial correlation between x and y, controlling for z.
    Formula equivalent to MATLAB's partialcorr.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    # Handling NaNs in input for robust calculation
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    if np.sum(mask) < 2:
        return np.nan
        
    x = x[mask]
    y = y[mask]
    z = z[mask]

    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)

    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denominator == 0:
        return np.nan
        
    return (r_xy - (r_xz * r_yz)) / denominator

# --- 2. Configuration & Target List ---

# ==========================================
# 在这里定义你的目标变量列表 (targetStr_List)
# ==========================================
# targetStr_List = ["nihtbx_cryst_uncorrected", "nihtbx_fluidcomp_uncorrected", "nihtbx_totalcomp_uncorrected"]
# targetStr_List = ['General','Ext','ADHD','Int']
targetStr_List = ['age']

targetStr_total = targetStr_List # 保持变量名与原逻辑一致
num_targets = len(targetStr_total)

# Project Folder
ProjectFolder = '/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/HCPD/prediction'

num_cv_runs = 101  # Time_0 to Time_100
num_folds = 5      # Fold_0 to Fold_4

# Initialize storage to mimic MATLAB Cell Arrays (N rows, M cols)
# We use lists of dictionaries temporarily, then convert to object arrays for .mat saving
results_summary = []

# Dictionaries to hold raw data for all targets
all_data = {
    'R_gg': {},
    'R_gw': {},
    'R_ww': {},
    'partialR_gw': {},
    'partialR_ww': {}
}

# --- 3. Main Loop for Each Target ---
for i_str in range(num_targets):
    target_str = targetStr_total[i_str]
    print(f"\n[{i_str+1}/{num_targets}] Processing target: {target_str}")

    # Define path
    base_folder = os.path.join(ProjectFolder, target_str, 'RegressCovariates_RandomCV')
    
    # Check if folder exists
    if not os.path.exists(base_folder):
        warnings.warn(f"Base folder not found for {target_str}: {base_folder}. Skipping.")
        continue # Skip to next target

    # Initialize arrays for this specific target
    corr_actual_gg = np.full(num_cv_runs, np.nan)
    mae_actual_gg = np.full(num_cv_runs, np.nan)
    corr_actual_gw = np.full(num_cv_runs, np.nan)
    mae_actual_gw = np.full(num_cv_runs, np.nan)
    corr_actual_ww = np.full(num_cv_runs, np.nan)
    mae_actual_ww = np.full(num_cv_runs, np.nan)

    # -------------------------------------------
    # Load Overall Results (GG, GW, WW)
    # -------------------------------------------
    
    # --- GGFC ---
    fc_current = 'GGFC'
    print(f"  Loading {fc_current}...")
    for i in range(num_cv_runs):
        res_file = os.path.join(base_folder, f"Time_{i}", fc_current, 'Res_NFold.mat')
        if os.path.isfile(res_file):
            try:
                mat = sio.loadmat(res_file)
                corr_actual_gg[i] = mat['Mean_Corr'].item()
                mae_actual_gg[i] = mat['Mean_MAE'].item()
            except: pass
    
    median_corr_gg = np.nanmedian(corr_actual_gg)
    # Sort descending (handling NaNs)
    temp_gg = corr_actual_gg.copy()
    temp_gg[np.isnan(temp_gg)] = -np.inf
    ind_gg = np.argsort(-temp_gg) 

    # --- GWFC ---
    fc_current = 'GWFC'
    print(f"  Loading {fc_current}...")
    for i in range(num_cv_runs):
        res_file = os.path.join(base_folder, f"Time_{i}", fc_current, 'Res_NFold.mat')
        if os.path.isfile(res_file):
            try:
                mat = sio.loadmat(res_file)
                corr_actual_gw[i] = mat['Mean_Corr'].item()
                mae_actual_gw[i] = mat['Mean_MAE'].item()
            except: pass

    median_corr_gw = np.nanmedian(corr_actual_gw)
    temp_gw = corr_actual_gw.copy()
    temp_gw[np.isnan(temp_gw)] = -np.inf
    ind_gw = np.argsort(-temp_gw)

    # --- WWFC ---
    fc_current = 'WWFC'
    print(f"  Loading {fc_current}...")
    for i in range(num_cv_runs):
        res_file = os.path.join(base_folder, f"Time_{i}", fc_current, 'Res_NFold.mat')
        if os.path.isfile(res_file):
            try:
                mat = sio.loadmat(res_file)
                corr_actual_ww[i] = mat['Mean_Corr'].item()
                mae_actual_ww[i] = mat['Mean_MAE'].item()
            except: pass

    median_corr_ww = np.nanmedian(corr_actual_ww)
    temp_ww = corr_actual_ww.copy()
    temp_ww[np.isnan(temp_ww)] = -np.inf
    ind_ww = np.argsort(-temp_ww)

    # Save raw data to dictionary
    all_data['R_gg'][target_str] = {'Corr': corr_actual_gg, 'MAE': mae_actual_gg}
    all_data['R_gw'][target_str] = {'Corr': corr_actual_gw, 'MAE': mae_actual_gw}
    all_data['R_ww'][target_str] = {'Corr': corr_actual_ww, 'MAE': mae_actual_ww}

    # -------------------------------------------
    # Calculate Partial Correlations
    # -------------------------------------------
    print("  Calculating partial correlations...")
    partial_r_gw_total = np.full(num_cv_runs, np.nan)
    partial_r_ww_total = np.full(num_cv_runs, np.nan)

    for i_cv in range(num_cv_runs):
        # Skip if any main result is missing at this rank
        if (np.isnan(corr_actual_gg[ind_gg[i_cv]]) or 
            np.isnan(corr_actual_gw[ind_gw[i_cv]]) or 
            np.isnan(corr_actual_ww[ind_ww[i_cv]])):
            continue

        id_gg = ind_gg[i_cv]
        id_gw = ind_gw[i_cv]
        id_ww = ind_ww[i_cv]

        # Inner function to load all 5 folds
        def load_all_folds(run_id, fc_type):
            idx, pred, test = [], [], []
            for k in range(num_folds):
                f_path = os.path.join(base_folder, f"Time_{run_id}", fc_type, f"Fold_{k}_Score.mat")
                if not os.path.isfile(f_path): return None, None, None
                try:
                    m = sio.loadmat(f_path)
                    idx.extend(m['Index'].flatten())
                    pred.extend(m['Predict_Score'].flatten())
                    test.extend(m['Test_Score'].flatten())
                except: return None, None, None
            return np.array(idx), np.array(pred), np.array(test)

        # Load Data
        gg_idx, gg_pred, _ = load_all_folds(id_gg, 'GGFC')
        gw_idx, gw_pred, gw_test = load_all_folds(id_gw, 'GWFC')
        ww_idx, ww_pred, ww_test = load_all_folds(id_ww, 'WWFC')

        if gg_idx is None or gw_idx is None or ww_idx is None:
            continue # Missing files

        # Sort and Align
        sort_gg = np.argsort(gg_idx)
        sort_gw = np.argsort(gw_idx)
        sort_ww = np.argsort(ww_idx)

        # Verify Alignment
        if not (np.array_equal(gg_idx[sort_gg], gw_idx[sort_gw]) and 
                np.array_equal(gg_idx[sort_gg], ww_idx[sort_ww])):
            warnings.warn(f"  Index mismatch at rank {i_cv}. Skipping.")
            continue

        # Calc Partial Corr
        try:
            partial_r_gw_total[i_cv] = partial_corr(gw_pred[sort_gw], gw_test[sort_gw].astype(float), gg_pred[sort_gg])
            partial_r_ww_total[i_cv] = partial_corr(ww_pred[sort_ww], ww_test[sort_ww].astype(float), gg_pred[sort_gg])
        except:
            pass

    # Store partial results
    all_data['partialR_gw'][target_str] = partial_r_gw_total
    all_data['partialR_ww'][target_str] = partial_r_ww_total

    median_partial_gw = np.nanmedian(partial_r_gw_total)
    median_partial_ww = np.nanmedian(partial_r_ww_total)

    print(f"    Median Partial GW: {median_partial_gw:.4f}")
    print(f"    Median Partial WW: {median_partial_ww:.4f}")

    # Add to summary list
    results_summary.append({
        'targetStr': target_str,
        'GG_median': median_corr_gg,
        'GW_median': median_corr_gw,
        'WW_median': median_corr_ww,
        'GW_partial_median': median_partial_gw,
        'WW_partial_median': median_partial_ww
    })

# --- 4. Format Data for MATLAB & Save ---
print("\nFormatting data for MATLAB compatibility...")

# Create Object Arrays to mimic MATLAB Cell Arrays {N x 3} or {N x 2}
# R_gg_totalStr: Col1=Name, Col2=Corr, Col3=MAE
# partialR_gw_totalStr: Col1=Name, Col2=PartialCorr

# Filter valid targets (in case some were skipped)
valid_targets = [res['targetStr'] for res in results_summary]
n_valid = len(valid_targets)

if n_valid == 0:
    print("No valid targets processed. Exiting.")
    exit()

# Initialize Object Arrays
cell_R_gg = np.zeros((n_valid, 3), dtype=object)
cell_R_gw = np.zeros((n_valid, 3), dtype=object)
cell_R_ww = np.zeros((n_valid, 3), dtype=object)
cell_partial_gw = np.zeros((n_valid, 2), dtype=object)
cell_partial_ww = np.zeros((n_valid, 2), dtype=object)
cell_median_results = np.zeros((n_valid, 6), dtype=object)

for idx, res in enumerate(results_summary):
    t_str = res['targetStr']
    
    # Fill R_gg / gw / ww (Name, Corr, MAE)
    cell_R_gg[idx, 0] = t_str
    cell_R_gg[idx, 1] = all_data['R_gg'][t_str]['Corr']
    cell_R_gg[idx, 2] = all_data['R_gg'][t_str]['MAE']
    
    cell_R_gw[idx, 0] = t_str
    cell_R_gw[idx, 1] = all_data['R_gw'][t_str]['Corr']
    cell_R_gw[idx, 2] = all_data['R_gw'][t_str]['MAE']

    cell_R_ww[idx, 0] = t_str
    cell_R_ww[idx, 1] = all_data['R_ww'][t_str]['Corr']
    cell_R_ww[idx, 2] = all_data['R_ww'][t_str]['MAE']

    # Fill Partial (Name, Values)
    cell_partial_gw[idx, 0] = t_str
    cell_partial_gw[idx, 1] = all_data['partialR_gw'][t_str]

    cell_partial_ww[idx, 0] = t_str
    cell_partial_ww[idx, 1] = all_data['partialR_ww'][t_str]

    # Fill Median Results (Name, GG, GW, WW, pGW, pWW)
    cell_median_results[idx, 0] = t_str
    cell_median_results[idx, 1] = res['GG_median']
    cell_median_results[idx, 2] = res['GW_median']
    cell_median_results[idx, 3] = res['WW_median']
    cell_median_results[idx, 4] = res['GW_partial_median']
    cell_median_results[idx, 5] = res['WW_partial_median']

# Prepare Table for display
df_results = pd.DataFrame(results_summary)
print("\nFinal Results Table:")
print(df_results)

# Save
# Note: Saving to ProjectFolder root instead of specific target folder since it contains multiple
output_file = os.path.join(ProjectFolder, 'partial_results_total_multi_targets.mat')
boxplot_file = os.path.join(ProjectFolder, 'partial_results_forBoxplot_multi_targets.mat')

mat_dict = {
    'R_gg_totalStr': cell_R_gg,
    'R_gw_totalStr': cell_R_gw,
    'R_ww_totalStr': cell_R_ww,
    'partialR_gw_totalStr': cell_partial_gw,
    'partialR_ww_totalStr': cell_partial_ww,
    'medianResults_totalStr': cell_median_results
    # Note: MATLAB table conversion is complex, saving the cell array 'medianResults_totalStr' is safer
}

sio.savemat(output_file, mat_dict)

# Prepare Boxplot Data Cell (Header + Data)
# Header row
header = np.array(['targetStr', 'R_gg_total', 'R_gw_total', 'partialR_gw_total', 'R_ww_total', 'partialR_ww_total'], dtype=object)
# Data rows
data_rows = np.zeros((n_valid, 6), dtype=object)
data_rows[:, 0] = valid_targets
data_rows[:, 1] = cell_R_gg[:, 1]       # GG Corr
data_rows[:, 2] = cell_R_gw[:, 1]       # GW Corr
data_rows[:, 3] = cell_partial_gw[:, 1] # GW Partial
data_rows[:, 4] = cell_R_ww[:, 1]       # WW Corr
data_rows[:, 5] = cell_partial_ww[:, 1] # WW Partial

# Combine
dataCell = np.vstack((header, data_rows))

sio.savemat(boxplot_file, {'dataCell': dataCell})

print(f"\nSaved successfully to:\n  {output_file}\n  {boxplot_file}")