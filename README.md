# WM_prediction/src Overview and Key Results

This directory provides the end-to-end pipeline from fMRI preprocessing outputs to functional connectivity (FC) matrices, vectorization, and modeling/analysis across multiple datasets (ABCD, CCNP, EFNY, HCPD, PNC). It summarizes module roles, typical usage, and key results.

## Directory Structure
- `conn_matrix/`
  - `process_dataset_unified.py`: Unified script that, per-dataset, performs valid-run selection → locate dseg and functional files → build GM/WM masks → extract atlas ROI time series → compute FC (GG/GW/WW) → apply Fisher Z → save outputs.
  - `generate_mask.py`, `compute_individual_fc.py`, `apply_fisher_z.py`: Single-subject tools for masking, FC computation, and Fisher Z transform.
  - `efny_hcppipeline/run_subject_fc.py`, `efny_hcppipeline/compare_subject_fc.py`: Reuse the existing EFNY FC logic on HCP-pipeline + XCP-D outputs and compare the new matrices against the legacy EFNY `individual_z` results.
  - `convert_matrices_to_vectors.py`: Vectorizes the upper triangle of FC matrices to feature vectors for prediction.
  - `reslice_atlases.py`: Reslices atlases to target space/resolution.
  - `batch_run_unified_*.sh`: Cluster batch scripts to run the unified pipeline.
- `preprocess/`
  - `screen_head_motion_*.py`: Computes `framewise_displacement` per dataset and writes `rest_fd_summary.csv` used to select valid runs.
  - `generate_*_covariates.py`, `generate_sublists.py`: Builds covariates and subject lists.
  - `hcp_pipeline/`: Stages EFNY BIDS inputs into an HCP-style StudyFolder and provides five HCP-style batch entry scripts for structural and resting-state preprocessing stages.
- `prediction/`
  - `predict_*_RandomCV.py`, `PLSr1_CZ_Random_RegressCovariates.py`: Predict age, cognition, and p-factor with random cross-validation and covariate regression.
  - `V_feature_merge/`: Re-run the same prediction workflow on concatenated GG/GW/WW feature sets while reusing the original `RandIndex.mat` splits.
- `results_vis/`
  - `compute_haufe_median.py`, `compute_partial_corr.py`, `compare_feature_merge_performance.py`: Model interpretability, statistical analysis, and merged-feature performance summaries.
  - `V_feature_merge/`: Paired t-test and one-way ANOVA scripts for merged-feature result comparison and plotting.

## Unified Pipeline Highlights (HCPD example)
- Valid run selection: Read `table/rest_fd_summary.csv` and select `REST1_acq-AP/PA`, `REST2_acq-AP/PA` under thresholds (e.g., FD ≤ 0.5 and low-motion ratio > 0.4).
- File binding strategy:
  - First bind the root using the subject’s functional BOLD file path, then locate dseg under the same-root `bids` directory to ensure per-subject consistency across runs.
- Masks and time series: Build GM/WM masks from dseg, extract atlas ROI mean time series from each `desc-denoised_bold.nii.gz`, and concatenate along the time axis.
- FC and Fisher Z: Compute `GG_FC` (GM-GM), `GW_FC` (GM-WM), `WW_FC` (WM-WM), apply Fisher Z (`arctanh`), and save to `fc_matrix/individual` and `fc_matrix/individual_z`.
- Vectorization: Expand the upper triangle of matrices to feature vectors for prediction scripts.

## Typical Usage
- Stage EFNY BIDS data for HCP preprocessing:
  - `python src/preprocess/hcp_pipeline/prepare_hcp_studyfolder_efny.py --subject sub-THU20231118133GYC --study-folder /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder`
  - The staging script automatically includes all available EFNY rest runs for a subject, with the current dataset convention constrained to `run-1` through `run-4`.
- Run one HCP stage for EFNY:
  - `bash src/preprocess/hcp_pipeline/PreFreeSurferPipelineBatch.sh --StudyFolder=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder --Session=sub-THU20231118133GYC`
- Submit one HCP stage as a Slurm array for EFNY:
  - `sbatch --partition=q_cn --cpus-per-task=4 --mem=24G --time=48:00:00 --array=1-10 src/preprocess/hcp_pipeline/submit_hcp_efny_stage.slurm.sh prefreesurfer /path/to/efny_subjects.txt`
- Run XCP-D after EFNY HCP `fMRIVolume`:
  - `bash src/preprocess/hcp_pipeline/xcpd_24p_csf_global.sh sub-THU20231118133GYC`
  - The script builds a per-subject temporary fMRIPrep-style bridge from `data/EFNY/hcp_studyfolder/<sub>/MNINonLinear/Results`, generates Python-based bridge/custom confounds, and writes XCP-D results to `data/EFNY/xcpd_hcp/step_2nd_24PcsfGlobal`.
- Generate FC matrices from EFNY HCP-pipeline XCP-D outputs:
  - `python src/conn_matrix/efny_hcppipeline/run_subject_fc.py --subject_id sub-THU20231118133GYC`
  - The script builds a compatibility tissue `dseg` from HCP `ribbon.nii.gz` and `aparc+aseg.nii.gz`, then reuses the existing `DatasetProcessor` logic to write independent outputs under `data/EFNY/hcppipeline_fc/`.
- Compare the new HCP-pipeline FC matrices against legacy EFNY results:
  - `python src/conn_matrix/efny_hcppipeline/compare_subject_fc.py --subject_id sub-THU20231118133GYC`
  - Comparison figures and vector correlations are written to `data/EFNY/hcppipeline_fc/comparison/`.
- Batch-submit EFNY HCP->XCP-D jobs:
  - `bash src/preprocess/hcp_pipeline/batch_xcpd.sh`
  - Optional subject list override: `bash src/preprocess/hcp_pipeline/batch_xcpd.sh /path/to/subjects.txt`
- Unified processing (example):
  - `python src/conn_matrix/process_dataset_unified.py --dataset_name HCPD --subject_id sub-HCDxxxxx --dataset_path d:\code\WM_prediction\data\HCPD --mask_output_dir d:\code\WM_prediction\data\HCPD\mri_data\wm_postproc --fc_output_dir d:\code\WM_prediction\data\HCPD\fc_matrix\individual --z_output_dir d:\code\WM_prediction\data\HCPD\fc_matrix\individual_z --gm_atlas d:\code\WM_prediction\data\atlas\resliced_hcpd\Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_resliced.nii.gz --wm_atlas d:\code\WM_prediction\data\atlas\resliced_hcpd\rICBM_DTI_81_WMPM_60p_FMRIB58_resliced.nii.gz`
- Vectorize matrices:
  - `python src/conn_matrix/convert_matrices_to_vectors.py --input_dir d:\code\WM_prediction\data\HCPD\fc_matrix\individual_z --output_dir d:\code\WM_prediction\data\HCPD\fc_vector`
- Run prediction (examples):
  - `python src/prediction/predict_age_RandomCV.py`
  - `python src/prediction/predict_cognition_RandomCV.py`
  - `python src/prediction/predict_pfactor_RandomCV.py`
- Run merged-feature prediction (examples):
  - `python src/prediction/V_feature_merge/predict_age_RandomCV.py`
  - `python src/prediction/V_feature_merge/predict_cognition_RandomCV.py`
  - `python src/prediction/V_feature_merge/predict_pfactor_RandomCV.py`
- Summarize merged-feature performance against baseline:
  - `python src/results_vis/compare_feature_merge_performance.py --dataset HCPD --task age`
  - `python src/results_vis/compare_feature_merge_performance.py --dataset ABCD --task cognition`
  - `python src/results_vis/compare_feature_merge_performance.py --dataset ABCD --task pfactor`
- Run paired t-test against the best child feature:
  - `/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/envs/ML/bin/python src/results_vis/V_feature_merge/paired_ttest_best_child.py --dataset HCPD --task age`
- Run one-way ANOVA against all child features:
  - `/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/envs/ML/bin/python src/results_vis/V_feature_merge/rm_anova_all_children.py --dataset HCPD --task age`

## Merged Feature Evaluation
The `prediction/V_feature_merge/` workflow concatenates the original GG, GW, and WW vectors into four combinations:

- `GG_GW_MergedFC`
- `GG_WW_MergedFC`
- `GW_WW_MergedFC`
- `GG_GW_WW_MergedFC`

All other modeling steps remain unchanged. The merged scripts reuse the existing baseline `Time_i/RandIndex.mat` files so performance differences are attributable to feature composition rather than a new random split. Results are written to:

- `data/<dataset>/prediction/<target>/V_feature_merge/RegressCovariates_RandomCV`

Merged-feature statistical summaries and figures are written to:

- `data/<dataset>/prediction/<target>/V_feature_merge/statistics/paired_ttest_best_child.csv`
- `data/<dataset>/prediction/<target>/V_feature_merge/statistics/rm_anova_all_children.csv`
- `data/<dataset>/prediction/<target>/V_feature_merge/statistics/figures/paired_ttest/`
- `data/<dataset>/prediction/<target>/V_feature_merge/statistics/figures/rm_anova/`

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
