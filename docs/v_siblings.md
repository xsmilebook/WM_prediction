# V_siblings 使用说明

`prediction/V_siblilngs/` 用于在不改动现有 PLS 预测逻辑的前提下，在 ABCD 的 cognition 和 p-factor 任务中控制 siblings / twins。

## 设计原则

- 不改动建模主流程，只改动输入样本。
- 家庭信息来自 `data/ABCD/table/abcd_y_lt.csv` 的 `rel_family_id`。
- `rel_family_id` 仅在 `baseline_year_1_arm_1` 行出现，因此只使用 baseline 行建立 `subid -> rel_family_id` 映射。
- 对同一 `rel_family_id`，按现有任务 `sublist` 的顺序保留第一个被试。

## 新增 sublist

新增脚本：

```text
src/preprocess/V_siblings/generate_familywise_sublists.py
```

运行方式：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/preprocess/V_siblings/generate_familywise_sublists.py
```

输出文件：

```text
data/ABCD/table/cognition_sublist_unique_family.txt
data/ABCD/table/pfactor_sublist_unique_family.txt
```

当前样本量变化：

- cognition: `4388 -> 3883`
- pfactor: `4465 -> 3950`

## 预测输入处理

新增辅助模块：

```text
src/prediction/V_siblilngs/familywise_inputs.py
```

逻辑如下：

- 读取原始 `cognition_sublist.txt` 或 `pfactor_sublist.txt`
- 读取 family-wise `*_unique_family.txt`
- 计算 family-wise 被试在原始 `sublist` 中的行索引
- 用该索引同步裁剪 GG / GW / WW 三个 `.npy` 特征矩阵
- 标签表和协变量表继续按 family-wise `sublist` 过滤并排序

这样可以保证：

- 特征矩阵的行顺序与原始 `.npy` 保存顺序一致
- 裁剪后的特征、标签、协变量样本数完全对齐

## 运行方式

认知预测：

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_siblilngs/predict_cognition_RandomCV.py
```

p-factor 预测：

```bash
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/prediction/V_siblilngs/predict_pfactor_RandomCV.py
```

说明：

- `predict_cognition_RandomCV.py` 当前写入 `RegressCovariates_RandomCV`
- `predict_pfactor_RandomCV.py` 当前保持与现有脚本一致，写入 `RegressCovariates_RandomCV_Permutation`

输出目录位于：

```text
data/ABCD/prediction/<target>/V_siblilngs/
```

## permutation 显著性

新增脚本：

```text
src/results_vis/V_siblings/compute_permutation_significance.py
```

用途：

- 对 `GG`、`GW`、`WW` 三个基线 feature，比较实际 `101` 次 random CV 的 `Mean_Corr` 中位数与 `1000` 次 permutation `Mean_Corr` null 分布
- 对 `GW/GG`、`WW/GG` 两个指标，先在每个 `Time_i` 内拼接 5 个 fold 的测试样本，再分别计算：
  - `GW/GG`：`GW` 预测与真实标签的偏相关，控制变量为同一 `Time_i` 的 `GG` 预测
  - `WW/GG`：`WW` 预测与真实标签的偏相关，控制变量为同一 `Time_i` 的 `GG` 预测
- 对上述两个偏相关指标，同样以实际 `101` 次结果的中位数对比 `1000` 次 permutation null 分布，计算右尾经验 `p` 值

经验 `p` 值定义为：

```text
p = (count(null >= observed_median) + 1) / (n_perm + 1)
```

运行方式：

```bash
source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate
conda activate ML
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_siblings/compute_permutation_significance.py --task cognition
python /ibmgpfs/cuizaixu_lab/xuhaoshu/code/WM_prediction/src/results_vis/V_siblings/compute_permutation_significance.py --task pfactor
```

默认目标列表：

- `cognition`
  - `nihtbx_cryst_uncorrected`
  - `nihtbx_fluidcomp_uncorrected`
  - `nihtbx_totalcomp_uncorrected`
- `pfactor`
  - `General`
  - `Ext`
  - `ADHD`

输出文件：

```text
results/V_siblings/ABCD_cognition_permutation_significance.csv
results/V_siblings/ABCD_pfactor_permutation_significance.csv
```

输出字段包括：

- `dataset`
- `task`
- `target`
- `metric_name`
- `n_actual_runs`
- `observed_mean`
- `observed_median`
- `n_permutation_runs`
- `permutation_mean`
- `permutation_median`
- `permutation_std`
- `z_score_vs_permutation`
- `empirical_p_right_tail`
- `significance_label`

## 验证结果

本次实现已完成以下静态检查：

- family-wise sublist 已成功生成
- cognition 特征裁剪后维度为：
  - `GG (3883, 4950)`
  - `GW (3883, 6800)`
  - `WW (3883, 2278)`
- pfactor 特征裁剪后维度为：
  - `GG (3950, 4950)`
  - `GW (3950, 6800)`
  - `WW (3950, 2278)`
- family-wise 标签与协变量维度分别为：
  - cognition: `labels (3883, 121)`, `covariates (3883, 5)`
  - pfactor: `labels (3950, 6)`, `covariates (3950, 5)`
- `compute_permutation_significance.py` 已通过 `py_compile` 语法检查
