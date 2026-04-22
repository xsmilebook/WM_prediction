# EFNY HCP 预处理说明

本文说明如何将 EFNY 的 BIDS 原始数据整理为 HCP `StudyFolder`，并用 HCP 5.0.0 依次运行 `PreFreeSurfer`、`FreeSurfer`、`PostFreeSurfer`、`fMRIVolume`、`fMRISurface`。默认输出目录统一为：

`/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder`

原始 BIDS 输入仅从：

`/ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/BIDS`

读取，不向该目录写入任何文件。

## 目录与脚本

- 共享参数与日志函数：`preprocess/hcp_pipeline/hcp_efny_batch_common.sh`
- 环境脚本：`preprocess/hcp_pipeline/hcp_efny_env.sh`
- BIDS 转 HCP staging：`preprocess/hcp_pipeline/prepare_hcp_studyfolder_efny.py`
- `PreFreeSurfer` 入口：`preprocess/hcp_pipeline/PreFreeSurferPipelineBatch.sh`
- `FreeSurfer` 入口：`preprocess/hcp_pipeline/FreeSurferPipelineBatch.sh`
- `PostFreeSurfer` 入口：`preprocess/hcp_pipeline/PostFreeSurferPipelineBatch.sh`
- `fMRIVolume` 入口：`preprocess/hcp_pipeline/GenericfMRIVolumeProcessingPipelineBatch.sh`
- `fMRISurface` 入口：`preprocess/hcp_pipeline/GenericfMRISurfaceProcessingPipelineBatch.sh`
- Slurm 数组提交脚本：`preprocess/hcp_pipeline/submit_hcp_efny_stage.slurm.sh`

## 环境约定

环境脚本会自动执行以下配置：

- `module load freesurfer/6.0.0`
- `module load fsl/6.3.0`
- `source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate`
- `conda activate ML`
- `HCPPIPEDIR=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/HCPpipelines-5.0.0`
- `CARET7DIR` 自动解析到 workbench 可执行目录
- `MSMBINDIR` 从 `/ibmgpfs/cuizaixu_lab/xuhaoshu/packages/MSM_HOCR-3.0FSL` 下搜索可执行 `msm`

注意：当前 `MSM_HOCR-3.0FSL` 如果只有源码、没有编译出的 `msm`，环境脚本会直接报错退出。需要先完成编译，或把 `msm` 可执行文件放到该目录或其子目录中。

## 参数来源

以 `sub-THU20231118133GYC` 为例，脚本中固定使用以下参数：

- rest run：自动识别 `task-rest_run-1` 到 `task-rest_run-4`
- HCP run 名：按实际存在的 run 生成 `rfMRI_REST{1..4}_{PA/AP/RL/LR}`
- `fMRIVolume` 的 `echospacing`：`0.000269996`
- spin-echo fieldmap 的 `seechospacing`：`0.000530007`
- `fMRIVolume` 的 `unwarpdir`：`y`
- `PreFreeSurfer` 的 `seunwarpdir`：`j`
- `PreFreeSurfer` 的结构像 `unwarpdir`：`z`
- `T1wSampleSpacing`：`5.2e-06`
- `T2wSampleSpacing`：`2.6e-06`
- `GradientDistortionCoeffs`：`NONE`
- `RegName`：`MSMSulc`
- `fMRIScout`：`NONE`

## 单被试测试

### 1. 先做 staging

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

python preprocess/hcp_pipeline/prepare_hcp_studyfolder_efny.py \
  --subject sub-THU20231118133GYC \
  --study-folder /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder
```

完成后可检查：

- `data/EFNY/hcp_studyfolder/sub-THU20231118133GYC/unprocessed/3T/T1w_MPR1/`
- `data/EFNY/hcp_studyfolder/sub-THU20231118133GYC/unprocessed/3T/T2w_SPC1/`
- `data/EFNY/hcp_studyfolder/sub-THU20231118133GYC/unprocessed/3T/rfMRI_REST*_*/`
- `data/EFNY/hcp_studyfolder/manifests/hcp_efny_manifest.tsv`

说明：

- `prepare_hcp_studyfolder_efny.py` 会自动纳入当前被试实际存在的全部 rest run
- 当前按 EFNY 约束要求，run 数量必须在 1 到 4 之间；若为 0 次或超过 4 次，脚本会直接报错

### 2. 依次运行 HCP 阶段

```bash
bash preprocess/hcp_pipeline/PreFreeSurferPipelineBatch.sh \
  --StudyFolder=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder \
  --Session=sub-THU20231118133GYC

bash preprocess/hcp_pipeline/FreeSurferPipelineBatch.sh \
  --StudyFolder=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder \
  --Session=sub-THU20231118133GYC

bash preprocess/hcp_pipeline/PostFreeSurferPipelineBatch.sh \
  --StudyFolder=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder \
  --Subject=sub-THU20231118133GYC

bash preprocess/hcp_pipeline/GenericfMRIVolumeProcessingPipelineBatch.sh \
  --StudyFolder=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder \
  --Subject=sub-THU20231118133GYC

bash preprocess/hcp_pipeline/GenericfMRISurfaceProcessingPipelineBatch.sh \
  --StudyFolder=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder \
  --Subject=sub-THU20231118133GYC
```

### 3. 关键输出

- `PreFreeSurfer`
  - `.../sub-THU20231118133GYC/T1w/T1w_acpc_dc_restore.nii.gz`
  - `.../sub-THU20231118133GYC/T1w/T2w_acpc_dc_restore.nii.gz`
- `FreeSurfer`
  - `.../sub-THU20231118133GYC/T1w/sub-THU20231118133GYC/`
- `PostFreeSurfer`
  - `.../sub-THU20231118133GYC/MNINonLinear/`
  - `.../sub-THU20231118133GYC/MNINonLinear/Native/`
  - `...sphere.MSMSulc.native.surf.gii`
- `fMRIVolume`
  - `.../sub-THU20231118133GYC/MNINonLinear/Results/rfMRI_REST*_*/`
- `fMRISurface`
  - 相同 `Results/rfMRI_REST*_*/` 目录下的 surface/CIFTI 输出

## 日志位置

每次运行都会将 stdout/stderr 写到：

`data/EFNY/hcp_studyfolder/logs/<stage>/<subject>/`

对 `fMRIVolume` 和 `fMRISurface`，日志进一步分到 run 子目录。

## 批量提交

### 1. 准备被试列表

例如：

```text
sub-THU20231118133GYC
sub-THU20231118134LZT
sub-THU20231118136YHY
```

### 2. 批量做 staging

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

python preprocess/hcp_pipeline/prepare_hcp_studyfolder_efny.py \
  --subject-list /path/to/efny_subjects.txt \
  --study-folder /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/hcp_studyfolder
```

### 3. 提交 Slurm array

```bash
sbatch --partition=q_cn --cpus-per-task=4 --mem=24G --time=48:00:00 --array=1-3 \
  preprocess/hcp_pipeline/submit_hcp_efny_stage.slurm.sh \
  prefreesurfer \
  /path/to/efny_subjects.txt
```

`submit_hcp_efny_stage.slurm.sh` 会读取 `SLURM_ARRAY_TASK_ID`，然后按阶段名调用对应的五个独立 batch 入口脚本之一。
如果只测试单个被试，也建议准备一个只含一行 subject ID 的列表文件，并用 `--array=1-1` 提交为单个作业。

其他阶段建议资源：

- `freesurfer`：`--cpus 8 --mem 32G --time 72:00:00`
- `postfreesurfer`：`--cpus 4 --mem 24G --time 24:00:00`
- `fmrivolume`：`--cpus 4 --mem 24G --time 24:00:00`
- `fmrisurface`：`--cpus 4 --mem 16G --time 12:00:00`

### 4. `SLURM_ARRAY_TASK_ID` 驱动

脚本支持 Slurm array。若设置了 `SLURM_ARRAY_TASK_ID`，则会读取 `--subject-list` 的第 N 行作为当前被试；若没有设置，则会顺序处理列表中的全部被试。

## 常见失败点

- `wb_command` 找不到
  - 检查 workbench 是否完整，尤其是 `exe_rh_linux64/wb_command`
- `msm` 找不到
  - 当前 `MSM_HOCR-3.0FSL` 路径如果只有源码，需要先编译
- `PreFreeSurfer` 在 TOPUP 阶段失败
  - 检查 AP/PA spin-echo 文件是否存在，且 `seechospacing=0.000530007`
- `fMRIVolume` 报 scout 相关错误
  - 当前方案固定 `--fmriscout=NONE`，由 HCP 自动提取第一帧；若原始时序异常，可单独检查该 run 的前几帧质量
- `MSMSulc` 在 `PostFreeSurfer` 或 `fMRISurface` 失败
  - 优先检查 `$MSMBINDIR/msm` 是否可执行，以及 workbench、MSM 配套库是否可用
