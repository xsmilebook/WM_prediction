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
