# V_merge 表格 slide 说明

本文件对应 `docs/v_merge_table_slides.pptx` 与 `docs/create_v_merge_table_slides.js`。

## 内容说明

- slide 1 汇总 `HCPD`、`PNC`、`CCNP`、`EFNY` 的年龄预测结果。
- slide 2 汇总 `ABCD` 的 cognition 结果，包括 `Crystallized`、`Fluid`、`Total` 三个指标。
- slide 3 汇总 `ABCD` 的 p-factor 结果，包括 `General`、`Ext`、`ADHD`、`Int` 四个指标。
- 三页表格均只展示 `feature_merge_summary_*.csv` 中的 `median_corr`。

## 标记规则

- 横轴为特征集合：`GGFC`、`GWFC`、`WWFC`、`GG+GW`、`GG+WW`、`GW+WW`、`GG+GW+WW`。
- 纵轴为数据集或目标指标。
- 若某个融合特征的相关系数高于同一行内最佳未融合特征（`GGFC`、`GWFC`、`WWFC` 中最大者），则该单元格使用红色加粗显示。

## 数据来源

- `../../data/HCPD/prediction/feature_merge_summary_age.csv`
- `../../data/PNC/prediction/feature_merge_summary_age.csv`
- `../../data/CCNP/prediction/feature_merge_summary_age.csv`
- `../../data/EFNY/prediction/feature_merge_summary_age.csv`
- `../../data/ABCD/prediction/feature_merge_summary_cognition.csv`
- `../../data/ABCD/prediction/feature_merge_summary_pfactor.csv`

## 重新生成

当前作者脚本使用 `PptxGenJS` 生成 `.pptx`，并调用共享的 `slides` helper 进行越界检查。若需重新生成，可在安装 `pptxgenjs` 后执行：

```bash
NODE_PATH=/tmp/wm_prediction_v_merge_tables/node_modules \
node /home/cuizaixu_lab/xuhaoshu/DATA_C/code/WM_prediction/src/docs/create_v_merge_table_slides.js
```
