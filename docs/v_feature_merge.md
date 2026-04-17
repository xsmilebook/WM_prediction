# V_feature_merge 使用说明

`prediction/V_feature_merge/` 用于在不改变现有建模方法的前提下，比较拼接 GG、GW、WW 特征后是否提升预测性能。

## 设计原则

- 保持与原始单模态脚本相同的协变量回归、MinMax 归一化、内层成分数选择和外层随机 5-fold CV。
- 只改变输入特征：将不同连接类型按列拼接为新的特征矩阵。
- 与原始结果比较时复用已有 `RandIndex.mat`，避免 split 差异影响结论。

## 支持的四种拼接方式

- `GG_GW_MergedFC`
- `GG_WW_MergedFC`
- `GW_WW_MergedFC`
- `GG_GW_WW_MergedFC`

## 输出路径

每个 target 的 merged 结果写入：

```text
data/<dataset>/prediction/<target>/V_feature_merge/RegressCovariates_RandomCV/
```

其中每个 `Time_i/` 下包含 4 个子目录，对应四种 merged 特征组合。

## 运行方式

年龄预测：

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_feature_merge/predict_age_RandomCV.py
```

认知预测：

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_feature_merge/predict_cognition_RandomCV.py
```

p-factor 预测：

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_feature_merge/predict_pfactor_RandomCV.py
```

这些脚本会读取原始单模态结果中的 `Time_0` 到 `Time_100` 的 `RandIndex.mat`。如果基线结果不存在，脚本会直接报错停止。

## 结果汇总

新增脚本 `results_vis/compare_feature_merge_performance.py` 可将基线模型与 merged 模型的 `Res_NFold.mat` 汇总为一张表，输出每个 target 的：

- `median_corr`
- `median_mae`
- 相对最佳基线模型的 `delta_corr_vs_best_baseline`
- 相对最佳基线模型的 `delta_mae_vs_best_baseline`

示例：

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/compare_feature_merge_performance.py \
  --dataset HCPD \
  --task age
```

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/compare_feature_merge_performance.py \
  --dataset ABCD \
  --task cognition
```

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/compare_feature_merge_performance.py \
  --dataset ABCD \
  --task pfactor
```

默认输出 CSV 为：

```text
data/<dataset>/prediction/feature_merge_summary_<task>.csv
```
