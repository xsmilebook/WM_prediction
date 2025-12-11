# WM_prediction/src Overview and Key Results

This directory provides the end-to-end pipeline from fMRI preprocessing outputs to functional connectivity (FC) matrices, vectorization, and modeling/analysis across multiple datasets (ABCD, CCNP, EFNY, HCPD, PNC). It summarizes module roles, typical usage, and key results.

## Directory Structure
- `conn_matrix/`
  - `process_dataset_unified.py`: Unified script that, per-dataset, performs valid-run selection → locate dseg and functional files → build GM/WM masks → extract atlas ROI time series → compute FC (GG/GW/WW) → apply Fisher Z → save outputs.
  - `generate_mask.py`, `compute_individual_fc.py`, `apply_fisher_z.py`: Single-subject tools for masking, FC computation, and Fisher Z transform.
  - `convert_matrices_to_vectors.py`: Vectorizes the upper triangle of FC matrices to feature vectors for prediction.
  - `reslice_atlases.py`: Reslices atlases to target space/resolution.
  - `batch_run_unified_*.sh`: Cluster batch scripts to run the unified pipeline.
- `preprocess/`
  - `screen_head_motion_*.py`: Computes `framewise_displacement` per dataset and writes `rest_fd_summary.csv` used to select valid runs.
  - `generate_*_covariates.py`, `generate_sublists.py`: Builds covariates and subject lists.
- `prediction/`
  - `predict_*_RandomCV.py`, `PLSr1_CZ_Random_RegressCovariates.py`: Predict age, cognition, and p-factor with random cross-validation and covariate regression.
- `results_vis/`
  - `compute_haufe_median.py`, `compute_partial_corr.py`: Model interpretability and statistical analysis utilities.

## Unified Pipeline Highlights (HCPD example)
- Valid run selection: Read `table/rest_fd_summary.csv` and select `REST1_acq-AP/PA`, `REST2_acq-AP/PA` under thresholds (e.g., FD ≤ 0.5 and low-motion ratio > 0.4).
- File binding strategy:
  - First bind the root using the subject’s functional BOLD file path, then locate dseg under the same-root `bids` directory to ensure per-subject consistency across runs.
- Masks and time series: Build GM/WM masks from dseg, extract atlas ROI mean time series from each `desc-denoised_bold.nii.gz`, and concatenate along the time axis.
- FC and Fisher Z: Compute `GG_FC` (GM-GM), `GW_FC` (GM-WM), `WW_FC` (WM-WM), apply Fisher Z (`arctanh`), and save to `fc_matrix/individual` and `fc_matrix/individual_z`.
- Vectorization: Expand the upper triangle of matrices to feature vectors for prediction scripts.

## Typical Usage
- Unified processing (example):
  - `python src/conn_matrix/process_dataset_unified.py --dataset_name HCPD --subject_id sub-HCDxxxxx --dataset_path d:\code\WM_prediction\data\HCPD --mask_output_dir d:\code\WM_prediction\data\HCPD\mri_data\wm_postproc --fc_output_dir d:\code\WM_prediction\data\HCPD\fc_matrix\individual --z_output_dir d:\code\WM_prediction\data\HCPD\fc_matrix\individual_z --gm_atlas d:\code\WM_prediction\data\atlas\resliced_hcpd\Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz --wm_atlas d:\code\WM_prediction\data\atlas\resliced_hcpd\rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz`
- Vectorize matrices:
  - `python src/conn_matrix/convert_matrices_to_vectors.py --input_dir d:\code\WM_prediction\data\HCPD\fc_matrix\individual_z --output_dir d:\code\WM_prediction\data\HCPD\fc_vector`
- Run prediction (examples):
  - `python src/prediction/predict_age_RandomCV.py`
  - `python src/prediction/predict_cognition_RandomCV.py`
  - `python src/prediction/predict_pfactor_RandomCV.py`

## Key Results Overview
Below we list representative metrics (e.g., correlations or effect sizes). `GG/GW/WW` denote GM-GM, GM-WM, WM-WM connectivity, and `GW/GG`, `WW/GG` are performance ratios relative to GG.

### Age
| Dataset | GG | GW | WW | GW/GG | WW/GG |
|---|---:|---:|---:|---:|---:|
| HCPD | 0.650068 | 0.577211 | 0.543921 | 0.270227 | 0.301559 |
| EFNY | 0.732051 | 0.641377 | 0.522290 | 0.251358 | 0.198356 |
| CCNP | 0.635269 | 0.644752 | 0.547240 | 0.366328 | 0.338548 |
| PNC  | 0.586958 | 0.557732 | 0.482474 | 0.326453 | 0.294717 |

Observation: Age prediction is strongest with GG; GW and WW are slightly weaker but informative, with GW typically outperforming WW.

### Cognition (cross-dataset averages)
| Metric | GG | GW | WW | GW/GG | WW/GG |
|---|---:|---:|---:|---:|---:|
| cryst     | 0.328468 | 0.173201 | 0.087723 | 0.047690 | 0.035476 |
| fluidcomp | 0.191065 | 0.106584 | 0.070561 | 0.048001 | 0.043687 |
| totalcomp | 0.286826 | 0.158176 | 0.089702 | 0.054424 | 0.043921 |

Observation: Cognition-related targets show GG clearly outperforming GW/WW; GW is about 50–60% of GG, WW is weaker but still adds incremental signal.

### pfactor
| Metric | GG | GW | WW | GW/GG | WW/GG |
|---|---:|---:|---:|---:|---:|
| General | 0.044384 | -0.005122 | -0.016414 | -0.012287 | -0.018811 |
| Ext     | 0.094642 | 0.055894  | 0.023014  | 0.023681  | 0.010617  |
| ADHD    | 0.076630 | 0.081794  | 0.025235  | 0.057621  | 0.018569  |
| Int     | 0.016003 | 0.018152  | 0.043755  | 0.017475  | 0.042150  |

Observation: p-factor signals are generally weak and vary by subdomain; some (e.g., Ext, ADHD) show GW comparable to or slightly above GG.

## Practical Tips
- Keep all runs for a subject and its dseg from the same root to avoid cross-path mixing (space/version mismatches).
- If atlas and functional resolutions differ, resample instead of broadcasting to prevent distorted ROI time series.
- Use Fisher Z-transformed features for modeling to improve linear separability and stability.

---
This README helps you quickly locate scripts, understand the processing pipeline, and grasp key results for reproducibility and further analysis.
