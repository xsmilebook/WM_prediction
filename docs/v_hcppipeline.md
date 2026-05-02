# V_hcppipeline 使用说明

`prediction/V_hcppipeline/predict_age_RandomCV.py` 现用于对 EFNY 的 HCP pipeline 功能连接结果执行年龄预测。

## 输入

脚本固定使用以下输入：

- 被试列表：`data/EFNY/table/sublist_xcpd_ready505.txt`
- FC 矩阵目录：`data/EFNY/hcppipeline_fc/fc_matrix/individual_z`
- 标签与协变量：`data/EFNY/table/subid_meanFD_age_sex.csv`

与原先直接读取 `fc_vector/*.npy` 不同，当前脚本会按 `sublist_xcpd_ready505.txt` 的顺序，直接从 `individual_z` 目录读取：

- `*_GG_FC_Z.npy`
- `*_GW_FC_Z.npy`
- `*_WW_FC_Z.npy`

并在脚本内完成向量化：

- `GG`、`WW`：取上三角展开
- `GW`：按全矩阵展开

这样可以直接使用 `hcppipeline_fc` 结果，而不依赖尚未单独生成的 `fc_vector` 目录。

## 被试对齐规则

脚本先读取 `sublist_xcpd_ready505.txt`，再与以下条件取交集：

- `hcppipeline_fc/fc_matrix/individual_z` 中三类矩阵都存在
- `subid_meanFD_age_sex.csv` 中存在年龄、性别和头动信息

当前检查结果为：

- `sublist_xcpd_ready505.txt` 共 505 人
- 其中 3 人缺少标签信息，因此实际进入预测的为 502 人

缺少标签的 3 人为：

- `sub-THU20240115173LYJ`
- `sub-THU20240413255SJZ`
- `sub-THU20250614607WL`

## 输出

结果目录固定为：

`/ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/data/EFNY/prediction/age/V_hcppipeline`

具体 random CV 结果写到：

`data/EFNY/prediction/age/V_hcppipeline/RegressCovariates_RandomCV/`

## 运行命令

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML

python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_hcppipeline/predict_age_RandomCV.py
```

## 说明

- 预测逻辑本身未改动，仍复用 `prediction/PLSr1_CZ_Random_RegressCovariates.py`
- 只调整了输入来源、被试对齐方式和输出目录
- 脚本内部新增了对上一级 `prediction/` 目录的导入路径，以保证能够正确加载 `PLSr1_CZ_Random_RegressCovariates.py`
