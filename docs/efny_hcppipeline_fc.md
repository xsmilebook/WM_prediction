# EFNY HCP pipeline 功能连接说明

本文说明如何基于 EFNY 的 HCP pipeline + XCP-D 结果，复用 `src/conn_matrix` 现有逻辑生成独立的功能连接矩阵，并与旧 EFNY `fc_matrix/individual_z` 结果进行比对。

## 入口脚本

- `conn_matrix/efny_hcppipeline/run_subject_fc.py`
- `conn_matrix/efny_hcppipeline/compare_subject_fc.py`

这两个脚本不改动现有 FC 计算逻辑：

- `run_subject_fc.py` 只负责准备一个兼容旧 `DatasetProcessor` 的 tissue `dseg`
- 真正的掩膜、时序提取、GG/GW/WW 计算和 Fisher Z 变换仍由 `conn_matrix/process_dataset_unified.py` 完成
- `compare_subject_fc.py` 只负责读取新旧 `individual_z` 结果并做图、算相关

## 兼容性 dseg

当前 `data/EFNY/xcpd_hcp/step_2nd_24PcsfGlobal/` 下的 `dseg` 为 HCP/FreeSurfer 风格的多标签分割，不能直接给旧 EFNY FC 流程使用。为复用旧逻辑，`run_subject_fc.py` 会在独立输出目录中生成兼容性 `dseg`：

- GM：来自 `MNINonLinear/ribbon.nii.gz` 的标签 `3` 和 `42`
- WM：来自 `MNINonLinear/ribbon.nii.gz` 的标签 `2` 和 `41`
- CSF：来自 `MNINonLinear/aparc+aseg.nii.gz` 的标签 `4, 5, 14, 15, 24, 31, 43, 44, 63`

输出文件位置：

- `data/EFNY/hcppipeline_fc/compat_fmriprep/sub-<id>/anat/sub-<id>_space-MNI152NLin6Asym_dseg.nii.gz`

该文件仅用于让旧 unified FC 逻辑继续按 `gm_label=1`、`wm_label=2` 工作，不覆盖原始 HCP/XCP-D 结果。

## 输出目录

所有新结果写入独立目录：

`data/EFNY/hcppipeline_fc/`

内部结构：

- `compat_fmriprep/`
- `wm_postproc/`
- `fc_matrix/individual/`
- `fc_matrix/individual_z/`
- `comparison/`

旧结果仍保持在：

`data/EFNY/fc_matrix/individual_z/`

## 运行命令

### 1. 生成单被试 FC

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

python src/conn_matrix/efny_hcppipeline/run_subject_fc.py \
  --subject_id sub-THU20231118133GYC
```

### 2. 比对新旧结果

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

python src/conn_matrix/efny_hcppipeline/compare_subject_fc.py \
  --subject_id sub-THU20231118133GYC
```

## 已生成的 133 被试结果

`sub-THU20231118133GYC` 已完成新 FC 生成与比对。

新矩阵：

- `data/EFNY/hcppipeline_fc/fc_matrix/individual/sub-THU20231118133GYC/`
- `data/EFNY/hcppipeline_fc/fc_matrix/individual_z/sub-THU20231118133GYC/`

比对结果：

- `data/EFNY/hcppipeline_fc/comparison/sub-THU20231118133GYC/sub-THU20231118133GYC_GG_FC_Z_compare.png`
- `data/EFNY/hcppipeline_fc/comparison/sub-THU20231118133GYC/sub-THU20231118133GYC_GW_FC_Z_compare.png`
- `data/EFNY/hcppipeline_fc/comparison/sub-THU20231118133GYC/sub-THU20231118133GYC_WW_FC_Z_compare.png`
- `data/EFNY/hcppipeline_fc/comparison/sub-THU20231118133GYC/sub-THU20231118133GYC_fc_correlation_summary.tsv`

## 133 被试相关系数

以旧 EFNY `individual_z` 为参照，新 HCP pipeline 结果的向量相关为：

- `GG`：下三角向量相关 `r = 0.8034156659`
- `GW`：全矩阵展开相关 `r = 0.2663570234`
- `WW`：下三角向量相关 `r = 0.5956048856`

其中：

- `GG`、`WW` 为对称矩阵，因此使用下三角向量
- `GW` 为非对称的 `100 x 68` 矩阵，因此使用全矩阵展开向量

## 注意事项

- 当前新流程不依赖旧 `rest_fd_summary.csv`，因此会直接使用 `xcpd_hcp` 目录下当前存在的 `task-rest_run-1/2/3` 输出
- `task-rest_run-4` 在 133 被试的 `xcpd_hcp` 目录中不存在，因此不会参与 FC 计算
- 新流程仅复用旧 FC 计算逻辑，不改写旧结果目录
