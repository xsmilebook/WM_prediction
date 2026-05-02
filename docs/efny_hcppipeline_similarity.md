# EFNY HCP pipeline 功能连接相似性分析

本文比较 `EFNY hcppipeline` 生成功能连接矩阵后，与旧 `EFNY fmriprep` 结果以及 `HCPD` 结果的相似程度，并结合处理流程分析差异来源。

## 数据与比较对象

- `EFNY hcppipeline`：`data/EFNY/hcppipeline_fc/fc_matrix/individual_z`
- 旧 `EFNY fmriprep`：`data/EFNY/fc_matrix/individual_z`
- `HCPD`：`data/HCPD/fc_matrix/individual_z`

比较时使用三类矩阵：

- `GG`：GM-GM
- `GW`：GM-WM
- `WW`：WM-WM

比较策略分为两层：

- `EFNY hcppipeline` vs 旧 `EFNY fmriprep`：在同被试交集上做逐被试矩阵相关，并计算群体平均矩阵相似性。
- `EFNY hcppipeline` vs `HCPD`：由于不存在同被试，只比较群体平均矩阵；这一部分同时混合了流程差异与队列差异。

本次实际纳入数量：

- 旧 `EFNY fmriprep`：522 人
- `EFNY hcppipeline`：506 人
- 两者同被试交集：503 人
- `HCPD`：531 人

## 计算方法

- `GG` 和 `WW` 为对称矩阵，取下三角向量后计算 Pearson 相关。
- `GW` 为非对称矩阵，直接展开全矩阵后计算 Pearson 相关。
- 为了尽量隔离流程影响，`EFNY hcppipeline` 与旧 `EFNY fmriprep` 的群体平均矩阵都只用 503 个交集被试计算。

## 结果一：`EFNY hcppipeline` 与旧 `EFNY fmriprep` 的同被试相似性

逐被试相关结果如下：

| 矩阵 | 平均 r | 中位数 r | Q1 | Q3 | 最小值 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GG | 0.7488 | 0.7518 | 0.7236 | 0.7804 | 0.2348 | 0.8456 |
| GW | 0.3502 | 0.3471 | 0.2922 | 0.4040 | 0.0711 | 0.6704 |
| WW | 0.5952 | 0.6055 | 0.5132 | 0.6942 | 0.1473 | 0.8594 |

解读：

- `GG` 相似性最高，说明新的 HCP pipeline 结果保留了较多 GM-GM 主体结构。
- `WW` 次之，说明 WM-WM 结构能部分保留，但变化明显大于 `GG`。
- `GW` 明显最低，是三类矩阵中最不稳定的一部分，提示跨组织连接最容易受到流程改变影响。

## 结果二：群体平均矩阵相似性

### 1. `EFNY hcppipeline` 与旧 `EFNY fmriprep`

| 矩阵 | 群体平均相关 r | 平均绝对差 MAD | 符号一致率 |
| --- | ---: | ---: | ---: |
| GG | 0.9396 | 0.0597 | 0.8895 |
| GW | 0.8447 | 0.0233 | 0.7465 |
| WW | 0.9534 | 0.0353 | 0.7243 |

说明即使逐被试层面 `GW` 相似性较低，群体平均后整体拓扑仍然与旧 `EFNY fmriprep` 高度一致。

### 2. `EFNY hcppipeline` 与 `HCPD`

| 矩阵 | 群体平均相关 r | 平均绝对差 MAD | 符号一致率 |
| --- | ---: | ---: | ---: |
| GG | 0.9291 | 0.0722 | 0.8731 |
| GW | 0.7627 | 0.0299 | 0.6765 |
| WW | 0.3915 | 0.0620 | 0.5123 |

### 3. 作为参考：旧 `EFNY fmriprep` 与 `HCPD`

| 矩阵 | 群体平均相关 r | 平均绝对差 MAD | 符号一致率 |
| --- | ---: | ---: | ---: |
| GG | 0.9748 | 0.0465 | 0.9166 |
| GW | 0.7407 | 0.0342 | 0.7591 |
| WW | 0.3738 | 0.0860 | 0.6651 |

## 结果三：到底更像谁

如果直接看 `EFNY hcppipeline` 的群体平均矩阵：

- `GG`：更像旧 `EFNY fmriprep`，也相当像 `HCPD`，但与旧 `EFNY fmriprep` 更接近。
- `GW`：更像旧 `EFNY fmriprep`，与 `HCPD` 的差距更明显。
- `WW`：明显更像旧 `EFNY fmriprep`，与 `HCPD` 的相似性较低。

如果把每个 `EFNY hcppipeline` 被试分别拿去和两个模板比较：

- `GG`：503 人中，377 人更接近旧 `EFNY fmriprep` 模板，126 人更接近 `HCPD` 模板。
- `GW`：503 人中，371 人更接近旧 `EFNY fmriprep` 模板，132 人更接近 `HCPD` 模板。
- `WW`：503 人中，502 人更接近旧 `EFNY fmriprep` 模板，仅 1 人更接近 `HCPD` 模板。

结论是：`EFNY hcppipeline` 整体上仍然更像同队列的旧 `EFNY fmriprep` 结果，而不是 `HCPD`。

但也能看到一个细节：

- 与旧 `EFNY fmriprep` 相比，新的 HCP pipeline 结果在 `GW` 和 `WW` 上确实比旧结果更靠近 `HCPD` 一些。
- 这说明 HCP 风格的预处理和分割主要改变了 WM 相关连接，而不是全面重写整个 FC 拓扑。

## 结果四：run 数量是否是主要原因

对 503 个交集被试，比较旧流程 `rest_fd_summary.csv` 中的有效 run 数与新 `xcpd_hcp` 目录下实际进入 FC 的 run 数：

- 498 人两套流程的 run 数完全一致。
- 仅 5 人在新流程中比旧流程多 1 个 run。
- 没有发现新流程比旧流程少 run 的情况。

分布如下：

- 旧流程：2 run 为 7 人，3 run 为 338 人，4 run 为 158 人。
- 新流程：2 run 为 5 人，3 run 为 337 人，4 run 为 161 人。

因此，`run` 数量不一致不是本次整体差异的主要来源，最多只能解释极少数个体的偏差。

## 差异主要集中在哪里

从定量结果看，差异主要集中在 `GW` 与 `WW`：

- `GG` 逐被试平均相关已达到 0.75 左右，说明 GM-GM 主结构较稳定。
- `GW` 逐被试平均相关只有 0.35，说明跨组织连接最敏感。
- `WW` 逐被试平均相关约 0.60，说明 WM 内部连接也明显受影响。

这说明差异更可能来自 WM 掩膜、WM 时序提取和跨组织配准，而不是单纯来自全脑整体信号形态。

## 流程上最可能导致差异的因素

### 1. 组织分割来源改变，尤其会影响 WM 相关矩阵

新流程没有直接使用旧 `fmriprep` 的 `dseg`，而是用：

- `MNINonLinear/ribbon.nii.gz` 提取 GM 和 WM
- `MNINonLinear/aparc+aseg.nii.gz` 提取 CSF

再生成兼容旧逻辑的三类 `dseg`。这一步会改变：

- GM/WM 边界
- WM 体素的纳入范围
- 某些边缘体素被归入 GM 还是 WM

因此最先受影响的通常就是 `GW` 和 `WW`，这与本次结果完全一致。

### 2. HCP 与旧 `fmriprep` 的空间配准和重采样链路不同

即便最后都落在 `MNI152NLin6Asym_res-2`，两条流程在前面的处理中并不相同：

- HCP pipeline 的结构配准、失真校正、`fMRIVolume` 输出链路不同
- 旧 `fmriprep` 的标准空间归一化与插值链路不同

这会导致同一个 atlas parcel 覆盖到的实际体素集合发生变化，尤其在：

- 白质边界
- 脑室附近
- 组织交界处

对应地，`GW` 和 `WW` 比 `GG` 更容易被放大。

### 3. 新流程的 confounds 是从 HCP 输出“桥接”到 XCP-D 的，不是原生 `fmriprep confounds`

新流程中 `extract_confounds_by_title.py` 会根据 HCP 输出重建：

- bridge 用的 `desc-confounds_timeseries.tsv`
- `csf/global` custom confounds
- `rmsd`
- 由 HCP `Movement_Regressors.txt` 转换而来的运动参数

虽然名义上仍然是 `24P + csf + global`，但 confounds 的来源与旧流程并不完全相同，可能引入：

- 运动参数数值细节差异
- `csf/global` 时序提取差异
- FD 或相关 QC 指标的微小差异

这类差异通常不会把 `GG` 完全打散，但会进一步增加 `GW` 和 `WW` 的不稳定性。

### 4. 新流程更接近 HCPD 的地方，主要体现在 WM 相关部分

从群体平均矩阵看：

- `GW`：`EFNY hcppipeline` 与 `HCPD` 的相关高于旧 `EFNY fmriprep` 与 `HCPD`
- `WW`：`EFNY hcppipeline` 与 `HCPD` 的相关也高于旧 `EFNY fmriprep` 与 `HCPD`

这提示 HCP 风格的预处理确实在把 EFNY 的 WM 相关连接推向 `HCPD` 的形态。

但这种“推近”并不足以超过同队列旧结果的相似性，说明：

- 队列本身的被试特异性结构仍然占主导
- HCP 风格流程主要是在 WM 相关部分施加系统性偏移，而不是把 EFNY 变成 `HCPD`

## 哪些更像“流程差异”，哪些更像“数据差异”

更像流程差异的证据：

- 同一批 EFNY 被试中，`GG` 高而 `GW/WW` 低，说明不是单纯由个体差异驱动。
- run 数几乎一致，排除了“大量 run 数不同”这一解释。
- 新流程相对旧流程，主要在 WM 相关部分向 `HCPD` 靠近，符合 HCP 风格分割与配准对 WM 更敏感的预期。

更像数据差异的证据：

- `EFNY hcppipeline` 与 `HCPD` 不是同被试比较。
- `HCPD` 与 EFNY 的年龄结构、采集方案、静息态长度、相位编码组合、站点条件都可能不同。
- 因此 `EFNY hcppipeline` 与 `HCPD` 的差异中，不能把全部差异都归因于流程。

## 总结

- `EFNY hcppipeline` 与旧 `EFNY fmriprep` 总体相似，三类矩阵都明显比它与 `HCPD` 更接近。
- 差异最主要集中在 `GW` 和 `WW`，而不是 `GG`。
- `run` 数差异只出现在 5/503 人，不是主因。
- 更可能的主因是 HCP 风格的组织分割、空间配准/重采样链路，以及桥接到 XCP-D 时 confounds 的构造方式。
- 新流程确实让 EFNY 的 WM 相关连接比旧流程更接近 `HCPD`，但这个变化幅度不足以改变“整体仍更像旧 EFNY”的结论。
