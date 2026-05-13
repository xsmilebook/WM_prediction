# V_holdout 使用说明

`prediction/V_holdout/` 用于回应审稿人关于大样本数据集应采用 holdout 评估的意见，在 ABCD cognition 与 p-factor 任务上提供单次 holdout 复现。

## 设计原则

- 尽量少改动原始 `prediction/` 管线，只在 `V_holdout/` 本地副本中修改外层评估方式。
- 保留原有协变量回归、MinMax 归一化和 PLS 组件数搜索逻辑。
- 将原先的 repeated random 5-fold CV 改为单次 family-aware holdout。

## 划分方式

- 外层划分固定为 `train/test = 1:1` 的单次 half-split。
- 该 outer split 使用 `data/ABCD/table/abcd_y_lt_baseline.csv` 中的 baseline `rel_family_id` 作为 group，确保 siblings 落在训练集或测试集的同一侧。
- 为了让两侧目标变量分布尽量接近，脚本会先对连续标签做分位数分箱，再使用 `StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=0)` 生成 split。
- 每个 target 会先在 `V_holdout/SharedSplitIndex.mat` 中生成并保存一份固定的 half-split，observed 与全部 permutation 共用这同一份 split。

## 建模流程

1. 在 outer train half 上拟合每个特征的协变量回归模型，并应用到 outer test half。
2. 在 outer train half 上进行 MinMax 归一化，并应用到 outer test half。
3. 在 outer train half 内部执行与主分析一致的 `5-fold CV`，用 `corr + inverse MAE` 选择最佳 PLS component 数。
4. 选定 component 后，在整个 outer train half 上重新拟合 PLS，并在固定的 outer test half 上进行一次最终评估。

## permutation 口径

- permutation 与主分析保持一致，只打乱 outer train half 的标签。
- outer test half 的标签保持真实，不参与标签打乱。
- observed 与全部 permutation 共用同一个 `SharedSplitIndex.mat`，因此 null 分布只反映训练标签置换带来的变化。

## 输出路径

每个 target 的 holdout 结果写入：

```text
data/ABCD/prediction/<target>/V_holdout/RegressCovariates_Holdout/
```

其中：

- `SharedSplitIndex.mat` 保存该 target 的固定 outer half-split，供 observed 与 permutation 共同复用。
- `Time_0/SplitIndex.mat` 保存训练集、测试集索引。
- `Time_0/GGFC/Holdout_Score.mat`
- `Time_0/GWFC/Holdout_Score.mat`
- `Time_0/WWFC/Holdout_Score.mat`

每个 `Holdout_Score.mat` 包含：

- `Train_Index`
- `Test_Index`
- `Test_Score`
- `Predict_Score`
- `Corr`
- `MAE`
- `ComponentNumber`
- `Inner_Corr`
- `Inner_MAE_inv`

## 运行方式

认知预测：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_holdout/predict_cognition_RandomCV.py
```

p-factor 预测：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_holdout/predict_pfactor_RandomCV.py
```

这两个入口脚本会先生成共享的 `SharedSplitIndex.mat`，然后：

- observed 提交 `1` 个 Slurm 作业；
- permutation 默认提交 `1000` 个 Slurm 作业。

因此默认输出为单次 holdout observed 结果及其固定 split 下的 permutation null。

## 结果汇总脚本

`results_vis/V_holdout/compute_partial_corr.py` 用于汇总已经完成的 holdout 与 permutation 结果。

- 脚本会自动扫描 `data/ABCD/prediction/` 下实际存在 `V_holdout/RegressCovariates_Holdout` 的 target。
- observed 部分直接读取 `Time_*/<GGFC|GWFC|WWFC>/Holdout_Score.mat` 中的 `Corr`、`MAE`、`Test_Index`、`Predict_Score` 与 `Test_Score`，并将单次 half test set 的 `Corr` 作为 holdout 评估指标。
- `GW_partial_corr` 和 `WW_partial_corr` 按同一个 `Time_i` 内的测试集预测计算，而不是跨 `Time_i` 重新配对。
- 由于 permutation 复用 `SharedSplitIndex.mat`，因此 null 分布只反映训练标签置换带来的变化，不再混入额外的 holdout split 变化。
- 如果 observed 目录下存在多个 `Time_i`，脚本会使用第一个有效的 half test set 结果，并给出 warning。
- permutation 部分读取 `V_holdout/RegressCovariates_Holdout_Permutation` 下所有 `Time_i`，并以右尾经验分布计算显著性：
  `p = (count(null >= observed) + 1) / (n_perm + 1)`。

运行方式：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_holdout/compute_partial_corr.py
```

输出文件写入：

```text
data/ABCD/prediction/V_holdout_partial_results_total_multi_targets.csv
data/ABCD/prediction/V_holdout_partial_results_total_multi_targets.mat
data/ABCD/prediction/V_holdout_partial_results_forBoxplot_multi_targets.mat
```

其中 CSV 的主要列包括：

- `GG_corr`、`GW_corr`、`WW_corr`
- `GW_partial_corr`、`WW_partial_corr`
- `GG_empirical_p`、`GW_empirical_p`、`WW_empirical_p`
- `GW_partial_empirical_p`、`WW_partial_empirical_p`
