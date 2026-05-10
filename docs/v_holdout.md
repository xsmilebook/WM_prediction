# V_holdout 使用说明

`prediction/V_holdout/` 用于回应审稿人关于大样本数据集应采用 holdout 评估的意见，在 ABCD cognition 与 p-factor 任务上提供单次 holdout 复现。

## 设计原则

- 尽量少改动原始 `prediction/` 管线，只在 `V_holdout/` 本地副本中修改外层评估方式。
- 保留原有协变量回归、MinMax 归一化和 PLS 组件数搜索逻辑。
- 将原先的 repeated random 5-fold CV 改为单次分层 holdout。

## 划分方式

- 将样本先按目标变量排序，再按原脚本的分层随机思路生成 `10` 份近似均衡的 split。
- 其中前 `8` 份合并为训练集，倒数第 `2` 份作为验证集，最后 `1` 份作为测试集。
- 因此外层比例固定为 `train/validation/test = 8:1:1`。

## 建模流程

1. 在训练集上拟合每个特征的协变量回归模型，并分别应用到验证集与测试集。
2. 在训练集上进行 MinMax 归一化，并应用到验证集与测试集。
3. 使用验证集上的 `corr + inverse MAE` 选择最佳 PLS component 数。
4. 选定 component 后，将训练集与验证集合并为 development set。
5. 在 development set 上重新拟合协变量回归、归一化与 PLS，并在测试集上进行一次最终评估。

## 输出路径

每个 target 的 holdout 结果写入：

```text
data/ABCD/prediction/<target>/V_holdout/RegressCovariates_Holdout/
```

其中：

- `Time_0/RandIndex.mat` 保存分层随机后的 10-way 顺序。
- `Time_0/SplitIndex.mat` 保存训练集、验证集、测试集索引。
- `Time_0/GGFC/Holdout_Score.mat`
- `Time_0/GWFC/Holdout_Score.mat`
- `Time_0/WWFC/Holdout_Score.mat`

每个 `Holdout_Score.mat` 包含：

- `Train_Index`
- `Validation_Index`
- `Test_Index`
- `Test_Score`
- `Predict_Score`
- `Corr`
- `MAE`
- `ComponentNumber`

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

这两个入口脚本会各自提交 `1` 个 Slurm 作业，因此默认只产生一次 holdout 结果，而不是 `101` 次 random CV 结果。
