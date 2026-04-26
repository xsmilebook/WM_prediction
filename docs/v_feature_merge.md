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

## 分布可视化

新增脚本 `results_vis/V_feature_merge/plot_feature_merge_distributions.py`，用于直接读取基线与 merged 结果中的 `Res_NFold.mat`，绘制 101 次 random CV 的 `Mean_Corr` 分布图。

绘图顺序固定为：

- `GG`
- `GW`
- `WW`
- `GG+GW`
- `GW+WW`
- `GG+WW`
- `GG+GW+WW`

图形形式为每个 feature 一组半边小提琴图加紧邻箱线图，其中：

- 左侧半边小提琴图展示分布形状
- 右侧窄箱线图展示四分位数和中位数
- 单 feature（GG/GW/WW）与 merged feature 之间用竖虚线分隔

默认行为：

- age：为 `HCPD`、`CCNP`、`EFNY`、`PNC` 各输出 1 张图
- ABCD cognition：为 `nihtbx_cryst_uncorrected`、`nihtbx_fluidcomp_uncorrected`、`nihtbx_totalcomp_uncorrected` 各输出 1 张图
- ABCD pfactor：为 `General`、`Ext`、`ADHD`、`Int` 各输出 1 张图

运行示例：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_feature_merge/plot_feature_merge_distributions.py
```

默认输出路径：

```text
code/WM_prediction/results/V_feature_merge/age/
code/WM_prediction/results/V_feature_merge/ABCD/cognition/
code/WM_prediction/results/V_feature_merge/ABCD/pfactor/
code/WM_prediction/results/V_feature_merge/feature_merge_distribution_summary.csv
```

## 统计比较

新增目录 `results_vis/V_feature_merge/`，用于在已有 `Res_NFold.mat` 基础上直接对 101 次 random CV 的 `Mean_Corr` 序列进行统计检验与可视化。

### 1. merged FC 与最佳子 feature 的配对 t 检验

脚本：

```bash
/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/envs/ML/bin/python \
  /home/cuizaixu_lab/xuhaoshu/DATA_C/code/WM_prediction/src/results_vis/V_feature_merge/paired_ttest_best_child.py \
  --dataset HCPD \
  --task age
```

该脚本对每个 merged feature：

- 读取其 101 次 `Mean_Corr`
- 读取对应子 feature 的 101 次 `Mean_Corr`
- 按子 feature 的整体 `median_corr` 选择表现更好的 baseline
- 对 merged 与该最佳子 feature 做配对样本 t 检验
- 输出 before-after 箱线图，并在图中添加显著性标记

输出路径：

```text
data/<dataset>/prediction/<target>/V_feature_merge/statistics/paired_ttest_best_child.csv
data/<dataset>/prediction/<target>/V_feature_merge/statistics/figures/paired_ttest/
```

### 2. merged FC 与全部子 feature 的普通单因素 ANOVA

脚本：

```bash
/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/envs/ML/bin/python \
  /home/cuizaixu_lab/xuhaoshu/DATA_C/code/WM_prediction/src/results_vis/V_feature_merge/rm_anova_all_children.py \
  --dataset HCPD \
  --task age
```

该脚本对每个 merged feature：

- 将 merged 与其所有子 feature 的 `Mean_Corr` 作为独立组输入普通单因素 ANOVA
- 输出 F 值、自由度、p 值和 partial eta squared
- 绘制各 feature 的箱线图，并在标题中写入 ANOVA 统计量

输出路径：

```text
data/<dataset>/prediction/<target>/V_feature_merge/statistics/rm_anova_all_children.csv
data/<dataset>/prediction/<target>/V_feature_merge/statistics/figures/rm_anova/
```
