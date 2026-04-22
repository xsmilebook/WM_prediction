# Project Structure

```
WM_prediction/src/                        # 当前仓库根目录
├── README.md                             # 项目概览、典型流程与代表性结果
├── AGENTS.md                             # AI 协作规则与修改约束
├── PLAN.md                               # 当前实现计划
├── ARCHITECTURE.md                       # 结构说明（本文）
├── docs/                                 # 补充文档
│   ├── architecture_update.md            # 本次架构文档同步说明
│   ├── hcp_efny.md                       # EFNY 数据集 HCP 预处理的 staging、单被试测试与批量提交说明
│   └── v_feature_merge.md                # merged FC 预测流程与结果汇总说明
│
├── conn_matrix/                          # 功能连接矩阵生成、变换与批处理脚本
│   ├── process_dataset_unified.py        # 统一处理多数据集：筛选有效 run、生成掩膜、提取时序、计算 FC、Fisher Z
│   ├── generate_mask.py                  # 基于 dseg 与 atlas 生成 GM/WM 掩膜
│   ├── compute_individual_fc.py          # 计算单被试 GG/GW/WW 功能连接矩阵
│   ├── apply_fisher_z.py                 # 对连接矩阵应用 Fisher Z 变换
│   ├── convert_matrices_to_vectors.py    # 将连接矩阵向量化为预测特征
│   ├── compute_group_avg_fc.py           # 汇总群体平均功能连接矩阵
│   ├── reslice_atlases.py                # 将 atlas 重采样到目标空间/分辨率
│   ├── batch_run_compute_fc.sh           # 集群批量计算 FC
│   ├── batch_run_fisher_z.sh             # 集群批量执行 Fisher Z
│   ├── batch_run_generate_mask.sh        # 集群批量生成掩膜
│   ├── batch_run_unified_ccnp.sh         # CCNP 数据集统一流程提交脚本
│   ├── batch_run_unified_efny.sh         # EFNY 数据集统一流程提交脚本
│   ├── batch_run_unified_hcpd.sh         # HCPD 数据集统一流程提交脚本
│   ├── batch_run_unified_pnc.sh          # PNC 数据集统一流程提交脚本
│   ├── cluster_run_convert_vectors.sh    # 集群批量向量化 FC 矩阵
│   └── logs/                             # `conn_matrix/` 相关日志目录
│
├── prediction/                           # 预测建模与交叉验证脚本
│   ├── PLSr1_CZ_Random_RegressCovariates.py # PLS 回归核心实现，含协变量回归与随机交叉验证
│   ├── predict_age_RandomCV.py           # 年龄预测入口脚本
│   ├── predict_cognition_RandomCV.py     # 认知指标预测入口脚本
│   ├── predict_pfactor_RandomCV.py       # p-factor 预测入口脚本
│   ├── cluster_run_predict.sh            # 集群批量提交预测任务
│   └── V_feature_merge/                  # merged FC 组合预测脚本
│       ├── common.py                     # merged 特征组合与 RandIndex 路径构建
│       ├── PLSr1_CZ_Random_RegressCovariates.py # merged FC 使用的本地 PLS 管线
│       ├── predict_age_RandomCV.py       # 年龄 merged FC 预测入口
│       ├── predict_cognition_RandomCV.py # 认知 merged FC 预测入口
│       └── predict_pfactor_RandomCV.py   # p-factor merged FC 预测入口
│
├── preprocess/                           # 数据预处理与样本筛选脚本
│   ├── abcd_preprocess_final.py          # ABCD 认知与 p-factor 样本预处理
│   ├── compare_sublists.py               # 对比不同筛选得到的 sublist
│   ├── generate_abcd_covariates.py       # 生成 ABCD 协变量表
│   ├── generate_pnc_covariates.py        # 生成 PNC 协变量表
│   ├── generate_sublists.py              # 生成建模用被试列表
│   ├── hcp_pipeline/                     # EFNY 数据集 HCP 预处理辅助脚本
│   │   ├── hcp_efny_batch_common.sh      # 五个 HCP batch 入口共享的 EFNY 参数、日志与输入发现函数
│   │   ├── hcp_efny_env.sh               # 加载 HCP 5.0.0、FSL、FreeSurfer、Workbench、MSM 与 ML 环境
│   │   ├── PreFreeSurferPipelineBatch.sh # EFNY 的 PreFreeSurfer 批处理入口，对齐 HCP 示例脚本命名
│   │   ├── FreeSurferPipelineBatch.sh    # EFNY 的 FreeSurfer 批处理入口，对齐 HCP 示例脚本命名
│   │   ├── PostFreeSurferPipelineBatch.sh # EFNY 的 PostFreeSurfer 批处理入口，对齐 HCP 示例脚本命名
│   │   ├── GenericfMRIVolumeProcessingPipelineBatch.sh # EFNY 的 fMRIVolume 批处理入口
│   │   ├── GenericfMRISurfaceProcessingPipelineBatch.sh # EFNY 的 fMRISurface 批处理入口
│   │   ├── prepare_hcp_studyfolder_efny.py # 将 EFNY BIDS 数据整理为 HCP StudyFolder 结构并生成 manifest
│   │   └── submit_hcp_efny_stage.slurm.sh # Slurm array 入口，按 subject list 提交单阶段批处理
│   ├── screen_head_motion_abcd.py        # ABCD 静息态头动筛选
│   ├── screen_head_motion_ccnp.py        # CCNP 静息态头动筛选
│   ├── screen_head_motion_efny.py        # EFNY 静息态头动筛选
│   ├── screen_head_motion_hcpd.py        # HCPD 静息态头动筛选
│   └── screen_head_motion_pnc.py         # PNC 静息态头动筛选
│
└── results_vis/                          # 结果解释与统计分析脚本
    ├── compute_haufe_median.py           # 计算 Haufe 权重中位数并重建矩阵/作图
    ├── compute_partial_corr.py           # 计算结果相关性的偏相关统计
    ├── compare_feature_merge_performance.py # 汇总基线与 merged FC 预测结果
    └── V_feature_merge/                  # merged FC 统计比较脚本
        ├── common.py                     # merged 结果读取、目标映射与作图辅助函数
        ├── paired_ttest_best_child.py    # merged FC 与最佳子 feature 的配对 t 检验和 before-after 图
        └── rm_anova_all_children.py      # merged FC 与全部子 feature 的普通单因素 ANOVA 和箱线图
```
