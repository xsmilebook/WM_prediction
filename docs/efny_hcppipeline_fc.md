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

当前 `data/EFNY/xcpd_hcp/step_2nd_24PcsfGlobal/` 下的 `dseg` 为 HCP/FreeSurfer 风格的多标签分割，不能直接给旧 EFNY FC 流程使用。为复用旧逻辑，`run_subject_fc.py` 会在独立输出目录中生成兼容性 `dseg`。新逻辑统一使用 `ribbon + wmparc`，并直接生成到 `wmparc.2` 的 `2 mm` 网格：

- GM：来自 `MNINonLinear/ribbon.nii.gz` 的标签 `3` 和 `42`
- WM：来自 `MNINonLinear/ribbon.nii.gz` 的标签 `2` 和 `41`，并额外纳入 `MNINonLinear/ROIs/wmparc.2.nii.gz` 的小脑白质标签 `7` 和 `46`
- CSF：来自 `MNINonLinear/ROIs/wmparc.2.nii.gz` 的标签 `4, 5, 14, 15, 24, 31, 43, 44, 63`

输出文件位置：

- `data/EFNY/hcppipeline_fc/compat_fmriprep/sub-<id>/anat/sub-<id>_space-MNI152NLin6Asym_dseg.nii.gz`

该文件仅用于让旧 unified FC 逻辑继续按 `gm_label=1`、`wm_label=2` 工作，不覆盖原始 HCP/XCP-D 结果。由于输出已经位于 `2 mm` 功能像网格，`wm_postproc` 不再需要额外把这个兼容 `dseg` 从 `0.7 mm` 重采样到 BOLD。

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

### 3. Slurm 批量提交 FC

新增批量脚本：

- `conn_matrix/batch_run_hcppipeline_fc.sh`

该脚本按被试列表逐个调用 `run_subject_fc.py`，默认读取：

- `data/EFNY/table/sublist_xcpd_ready505.txt`

默认 `#SBATCH --array=1-505%200` 也对应这份 `505` 人名单。若后续名单长度变化，需要同步修改脚本中的 array 范围。

当前仓库中这份名单已生成：

- `data/EFNY/table/sublist_xcpd_ready505.txt`

推荐先准备一份只包含“已完成 HCP + XCP-D、准备进入 FC 计算”的被试列表，再提交：

```bash
python - <<'PY' > /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/sublist_xcpd_ready505.txt
from pathlib import Path

project = Path('/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction')
source_list = project / 'data/EFNY/table/sublist_new_left521.txt'
study_root = project / 'data/EFNY/hcp_studyfolder'
xcpd_root = project / 'data/EFNY/xcpd_hcp/step_2nd_24PcsfGlobal'

for line in source_list.read_text().splitlines():
    subj = line.strip()
    if not subj:
        continue
    if not (study_root / subj / 'MNINonLinear/Results').exists():
        continue
    func_dir = xcpd_root / subj / 'func'
    func_files = list(func_dir.glob(f'{subj}_task-rest_run-*_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz'))
    if func_files:
        print(subj)
PY
```

提交命令：

```bash
sbatch conn_matrix/batch_run_hcppipeline_fc.sh \
  /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/table/sublist_xcpd_ready505.txt
```

日志目录：

- `log/conn_matrix/hcppipeline_fc/`

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
- 批量提交时，建议只对已完成 HCP + XCP-D 的被试列表提交，避免把缺少 `desc-denoised_bold` 的被试一并送入 array
