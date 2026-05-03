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

## GG 与 All 的共享 target permutation

如果要做 `1000` 次 permutation，并且在每一次中同时比较：

- `GGFC`
- `GG_GW_WW_MergedFC`

当前实现使用的是“打乱训练折内 `target`”的 null。

具体规则：

- 每个 `Time_i` 会同时拟合 `GGFC` 和 `GG_GW_WW_MergedFC`
- 两个模型在同一个 `Time_i` 下共用同一次 outer CV split
- 每个 outer fold 内，两者共用同一次打乱后的 `subjects_score_train`
- 因此可直接在每次 permutation 后计算 `all_acc - gg_acc`

三个入口脚本 `predict_age_RandomCV.py`、`predict_cognition_RandomCV.py`、`predict_pfactor_RandomCV.py` 中都新增了以下开关：

```python
RUN_GG_ALL_TARGET_PERMUTATION = True
GG_ALL_PERMUTATION_TIMES = 1000
```

默认值中 `RUN_GG_ALL_TARGET_PERMUTATION = False`，避免误提交大批量作业。需要运行时，将其改为 `True` 后执行对应脚本即可。

示例：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_feature_merge/predict_pfactor_RandomCV.py
```

默认输出目录：

```text
data/<dataset>/prediction/<target>/V_feature_merge/RegressCovariates_RandomCV_Permutation_GG_All/
```

其中：

- `Time_i/GGFC/` 保存第 `i` 次 permutation 下 `GGFC` 的结果
- `Time_i/GG_GW_WW_MergedFC/` 保存第 `i` 次 permutation 下 `GG_GW_WW_MergedFC` 的结果
- `Time_i/PermutationIndex.mat` 记录该次 permutation 中每个 outer fold 的训练集 `target` 打乱顺序

## GG 与 All permutation 差值汇总

新增脚本 `results_vis/V_feature_merge/compute_gg_all_permutation_significance.py`，用于汇总共享 target permutation 下的：

- `GGFC`
- `GG_GW_WW_MergedFC`

并直接计算：

- 每次 permutation 的 `all_corr - gg_corr`
- 实际观测值 `median(all_corr) - median(gg_corr)`
- 该观测差值相对于 permutation null 的右尾经验 `p` 值

统计口径：

- 观测值来自实际 `101` 次 random CV
- null 分布来自 `RegressCovariates_RandomCV_Permutation_GG_All/` 下 `1000` 次 permutation
- permutation 统计量为每个 `Time_i` 的 `Mean_Corr(all) - Mean_Corr(gg)`
- observed 统计量为 `median(Mean_Corr(all)) - median(Mean_Corr(gg))`
- 显著性使用右尾经验 `p` 值：`(count(null >= observed) + 1) / (n_perm + 1)`

运行示例：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_feature_merge/compute_gg_all_permutation_significance.py \
  --dataset ABCD \
  --task pfactor
```

默认输出路径：

```text
code/WM_prediction/results/V_feature_merge/<dataset>_<task>_gg_all_permutation_detail.csv
code/WM_prediction/results/V_feature_merge/<dataset>_<task>_gg_all_permutation_significance.csv
```

其中：

- `*_detail.csv` 同时保存 observed `101` 次 random CV 和 permutation `1000` 次的逐次 `gg_corr`、`all_corr`、`delta_corr`
- `*_significance.csv` 汇总每个 target 的 `observed_median_delta_corr`、null 分布均值/中位数/标准差，以及经验 `p` 值和显著性标签

## GG 与 All detail 分布图

新增脚本 `results_vis/V_feature_merge/plot_gg_all_permutation_detail.py`，用于直接读取 `*_gg_all_permutation_detail.csv`，并绘制 observed 与 permutation 的分布图。

图像包含 3 个面板：

- `GG corr`
- `GG+GW+WW corr`
- `All-GG corr`

每个面板都会同时叠加：

- observed `101` 次 random CV 的分布
- permutation `1000` 次 null 的分布
- 两者各自的 median 竖线

运行示例：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_feature_merge/plot_gg_all_permutation_detail.py \
  --input_csv /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/results/V_feature_merge/PNC_age_gg_all_permutation_detail.csv
```

默认输出路径：

```text
code/WM_prediction/results/V_feature_merge/<dataset>_<task>_gg_all_permutation_detail_distribution.tiff
```

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

当前脚本只绘制两类 feature：

- `GG`
- `GG+GW+WW`

当前配色为：

- `GG`：蓝色
- `GG+GW+WW`：橘色（沿用原 `GG+GW` 的配色）

图中常规文本优先使用 `Arial`；若当前运行环境未安装 `Arial`，则自动回退到可用的 sans-serif 字体。显著性标记 `* / ** / ns` 单独使用 `DejaVu Sans`。

图像导出格式为 `TIFF`，默认分辨率为 `300 DPI`。

为保证不同任务图之间的视觉一致性，脚本以 `age` 图作为参考，按分组数量动态换算半边小提琴图、箱线图及组内左右偏移的横向参数。也就是说，不同任务使用的不是固定数据坐标宽度，而是固定视觉宽度；同时整张图的画布宽度也会随分组数量缩放，因此只包含 `ADHD` 的 `pfactor` 图会比 `age` 图明显更窄。当前 `age` 图宽度设置为 `cognition` 图与 `pfactor` 图宽度之和。三张图的高度统一并较之前进一步压低。

图形形式为组内对照的半边小提琴图加箱线图。三张图都在右上角显示图例，且图例字号较原设置缩小；每个分组上方标注 `GG` 与 `GG+GW+WW` 的配对 `t test` 显著性。其中显著性规则为：

- `p < 0.001`：`**`
- `0.001 <= p < 0.05`：`*`
- `p >= 0.05`：`ns`

显著性标记连接线使用较短的竖直端点，以减少对箱线图上方留白的占用。

箱线图离群点使用黑色边框显示。

另外，每个 feature 自身的半边小提琴图与箱线图之间上方固定添加 `**` 标记，用于表示该 feature 的预测性能本身已确认显著；其纵向位置按该 feature 自身分布的最高值自适应确定，不与同组另一 feature 共用高度，横向位置则由箱线图中心加减半个箱线图宽度确定，放在更靠近半边小提琴的一侧。

默认输出 3 张图：

- age：按 `EFNY`、`devCCNP`、`HCP-D`、`PNC` 的顺序在同一张图中绘制 4 个数据集
- cognition：按 `Total`、`Crystal`、`Fluid` 的顺序在同一张图中绘制 3 个指标
- pfactor：只绘制 `ADHD`

其中 `age` 图的 y 轴主刻度间隔固定为 `0.05`。
其中 `pfactor` 图的 y 轴固定为 `0~0.12`，不再随数据范围自适应变化；同时单组 `ADHD` 图会进一步缩窄，并压缩左右空白区间。`pfactor` 的 y 轴与 ylabel 均位于左侧，与 `age` 和 `cognition` 保持一致。

运行示例：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_feature_merge/plot_feature_merge_distributions.py
```

默认输出路径：

```text
code/WM_prediction/results/V_feature_merge/age/age_all_datasets_GG_vs_GG_GW_WW_half_violin_box.tiff
code/WM_prediction/results/V_feature_merge/ABCD/cognition/cognition_all_targets_GG_vs_GG_GW_WW_half_violin_box.tiff
code/WM_prediction/results/V_feature_merge/ABCD/pfactor/pfactor_all_targets_GG_vs_GG_GW_WW_half_violin_box.tiff
code/WM_prediction/results/V_feature_merge/feature_merge_distribution_summary.csv
code/WM_prediction/results/V_feature_merge/feature_merge_distribution_significance.csv
```

其中：

- `feature_merge_distribution_summary.csv` 汇总每个分组中 `GG` 与 `GG+GW+WW` 的 `n_runs`、`median_corr`、`mean_corr`、`std_corr`、`min_corr`、`max_corr`
- `feature_merge_distribution_significance.csv` 汇总每个分组的配对 `t test` 结果，包括 `n_pairs`、`p_value`、`t_stat`、`mean_delta_corr`、`median_delta_corr` 和显著性标签

## pfactor permutation 显著性

新增脚本 `results_vis/V_feature_merge/compute_pfactor_permutation_significance.py`，用于对 `ABCD` 的 `General`、`Ext`、`ADHD` 三个 pfactor 指标进行经验 permutation 显著性评估。

统计口径：

- 实际统计量：每个 feature 在 `101` 次 random CV 上的 `median corr`
- null 分布：对应 permutation 目录下 `1000` 次 `Mean_Corr`
- 显著性：右尾经验 p 值，计算方式为 `(count(null >= observed) + 1) / (n_perm + 1)`

当前数据限制：

- `GG`、`GW`、`WW` 可直接使用各自已有的 permutation 结果计算 p 值
- `GG+GW`、`GW+WW`、`GG+WW` 目前仍没有独立的 `V_feature_merge` permutation 目录
- `GG+GW+WW` 可通过 `RegressCovariates_RandomCV_Permutation_GG_All/` 生成，但只有在实际运行该流程后才可用于显著性评估

运行示例：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_feature_merge/compute_pfactor_permutation_significance.py
```

默认输出路径：

```text
code/WM_prediction/results/V_feature_merge/ABCD_pfactor_permutation_significance.csv
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
