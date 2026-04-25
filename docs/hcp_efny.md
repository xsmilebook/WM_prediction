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
- HCP 到 XCP-D 的 confounds helper：`preprocess/hcp_pipeline/extract_confounds_by_title.py`
- 单被试 XCP-D 入口：`preprocess/hcp_pipeline/xcpd_24p_csf_global.sh`
- 批量提交 XCP-D：`preprocess/hcp_pipeline/batch_xcpd.sh`
- Slurm 数组提交脚本：`preprocess/hcp_pipeline/submit_hcp_efny_stage.slurm.sh`

## 环境约定

环境脚本会自动执行以下配置：

- `module load freesurfer/6.0.0`
- `module load fsl/6.3.0`
- `module load openblas/0.3.7`
- `source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate`
- `conda activate ML`
- `HCPPIPEDIR=/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/HCPpipelines-5.0.0`
- `CARET7DIR` 自动解析到运行时生成的 workbench wrapper 目录，`wb_command` 通过该 wrapper 调用真实的 `exe_rh_linux64/wb_command`
- workbench 依赖的 `LD_LIBRARY_PATH` 只在 `wb_command` 的局部 wrapper 中设置，不再全局污染 shell 环境
- `MSMBINDIR` 解析到 `/ibmgpfs/cuizaixu_lab/xuhaoshu/packages/msm_centos_v3_bin`
- wrapper 目录中的 `wb_command` 必须是单独生成的普通脚本文件，不能先对 `wb_command` 建软链接再写入，否则会反向覆盖 workbench 安装目录中的目标文件

注意：当前默认使用官网预编译的 `msm_centos_v3`，需确保 `msm_centos_v3_bin/msm` 链接存在，并且 `module load openblas/0.3.7` 后可正常执行。

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

- `prepare_hcp_studyfolder_efny.py` 会自动纳入当前被试实际存在的 `run-1` 到 `run-4`
- 若存在 `run-5` 及之后的 rest 扫描，脚本会直接忽略，不纳入 HCP staging
- 若 `run-1` 到 `run-4` 中一个都没有，脚本会报错

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

## HCP 结果继续接 XCP-D

当前 EFNY 的 HCP run 命名为 `rfMRI_REST{1..4}_{PA/AP/...}`，而仓库当前使用的 XCP-D 版本仍沿用旧版调用方式。因此本仓库不直接把 `hcp_studyfolder` 当作 XCP-D 的 HCP 输入，而是在单被试脚本内部临时生成最小 `fMRIPrep` 风格 bridge，再继续运行 XCP-D。这个 bridge 只在脚本运行期间使用，不作为新的独立目录规范。

### 单被试运行

```bash
bash preprocess/hcp_pipeline/xcpd_24p_csf_global.sh sub-THU20231118133GYC
```

脚本会执行以下步骤：

- 从 `data/EFNY/hcp_studyfolder/sub-THU20231118133GYC/MNINonLinear/Results/rfMRI_REST*_*/` 发现已完成的 `fMRIVolume` run
- 在 `data/EFNY/xcpd_hcp/fmriprep_bridge/` 下为当前被试创建临时 bridge
- 用 `extract_confounds_by_title.py` 生成两类 TSV
  - bridge 所需的 `desc-confounds_timeseries.tsv`
  - 仅包含 `csf` 与 `global_signal` 及其导数/平方项的 custom confounds TSV
- `Movement_Regressors.txt` 中的旋转参数在 HCP 输出里使用角度制，helper 会先转换为弧度，再写入 bridge confounds 和基于这些参数计算的 `framewise_displacement`
- 运行 XCP-D，并将结果写到 `data/EFNY/xcpd_hcp/step_2nd_24PcsfGlobal/`

### 批量提交

```bash
bash preprocess/hcp_pipeline/batch_xcpd.sh
```

默认被试列表为：

`data/EFNY/table/sublist_new_left521.txt`

也可显式指定列表文件：

```bash
bash preprocess/hcp_pipeline/batch_xcpd.sh /path/to/subjects.txt
```

批量日志默认写到：

`log/hcp_pipeline/xcpd/`

### 当前已生成的 bridge 示例

本次已为单被试 `sub-THU20231118133GYC` 生成持久化 bridge，位置如下：

- `data/EFNY/xcpd_hcp/fmriprep_bridge/sub-THU20231118133GYC/`
- `data/EFNY/xcpd_hcp/custom_confounds_csf_global_24p/sub-THU20231118133GYC/func/`

对应 3 个已完成的 HCP `fMRIVolume` run：

- `task-rest_run-1`
- `task-rest_run-2`
- `task-rest_run-3`

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
- `PostFreeSurfer` 长时间停在 `wmparc` 附近且 `MNINonLinear/Native` 没有继续产出
  - 优先检查环境脚本是否错误解析到了 `bin_rh_linux64/wb_command` 的包装脚本；当前应使用 `exe_rh_linux64/wb_command`
- `wb_command` 报 `libglapi.so.0` 或 OpenGL 相关动态库缺失
  - 先确认 `ML` 环境已安装 `mesa-libglapi-cos7-x86_64`，并重新 `source .../activate && conda activate ML`
- `msm` 找不到
  - 优先检查 `/ibmgpfs/cuizaixu_lab/xuhaoshu/packages/msm_centos_v3_bin/msm` 是否存在且可执行
- `msm` 报 `libopenblas.so.0` 缺失
  - 检查 `module load openblas/0.3.7` 是否在当前 shell 或 Slurm 作业中生效
- `PreFreeSurfer` 在 TOPUP 阶段失败
  - 检查 AP/PA spin-echo 文件是否存在，且 `seechospacing=0.000530007`
- `fMRIVolume` 报 scout 相关错误
  - 当前方案固定 `--fmriscout=NONE`，由 HCP 自动提取第一帧；若原始时序异常，可单独检查该 run 的前几帧质量
- `xcpd_24p_csf_global.sh` 报缺少 `Results/rfMRI_REST*_*/` 或 `Movement_Regressors.txt`
  - 说明当前被试尚未完成 `fMRIVolume`，或该 run 关键输出不完整
- `xcpd_24p_csf_global.sh` 报 bridge confounds 生成失败
  - 优先检查 `MNINonLinear/ROIs/wmparc.2.nii.gz`、`brainmask_fs.2.nii.gz` 与对应 BOLD 的维度是否一致
- `MSMSulc` 在 `PostFreeSurfer` 或 `fMRISurface` 失败
  - 优先检查 `$MSMBINDIR/msm` 是否可执行，以及 workbench、MSM 配套库是否可用
