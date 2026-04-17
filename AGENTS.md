# AGENTS.md

`WM_prediction` 仓库的 AI 协作执行规则。
本仓库将 `ARCHITECTURE.md` 视为稳定目录结构的唯一事实来源；除非用户明确要求，否则不要提出结构性改动。

## Basic workflow

1) 在规划或编辑前，先阅读项目根目录下的 `AGENTS.md` 和 `ARCHITECTURE.md`。
2) 仅按用户的明确指令执行变更。
3) 若变更影响代码的用法或结构，需同步更新 `README.md`。
3.1) 只要代码被修改或生成了新结果，必须新增或更新补充文档 `/docs` 进行说明。
3.2) 运行时间短且不需并行的脚本可直接执行（执行方式：source /GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/activate; conda activate ML）；并行任务如多次CV以及permutation，提供提交命令或 sbatch 脚本，供用户执行。

## 范围与约束

- 不允许在没有用户允许的情况下进行git commit。
- 该项目目前处于补充、验证分析阶段，仅允许在当前代码结构上微调，不允许出现大规模变动。
- 默认范围为仅文档，除非用户明确要求工程改动。
- 不要修改运行时产物。
- 非用户明确要求时，不要改动目录命名或结构。
- 保持最小差异；避免与当前请求无关的重构。

## Engineering standards

- 脚本尽量简洁、易懂。
- 不要自行添加兜底和回退代码。
- 避免临时 `sys.path` hack；若不可避免，只允许在入口脚本中使用并写明原因。

## 文档规范

- 使用精确、科学的表述。
- `docs/` 下文档全部使用中文，文件名保持英文。
- 仓库根目录文件夹名称保持英文。