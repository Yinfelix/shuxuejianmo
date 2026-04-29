# 数学建模协作工作区

这个仓库用于 2026 同济数学建模竞赛 C 题的代码、实验结果和论文协作。

仓库已经排除了赛题原件、原始 Excel 数据和本地工具链目录，适合多人并行开发、复现实验和同步论文内容。

## 仓库内容

- `scripts/c_problem/`：C 题的主要求解脚本与实验入口
- `data/processed/`：已经落盘的中间结果与实验明细
- `outputs/tables/`：论文中使用的汇总表
- `outputs/figures/`：导出的图像结果
- `reports/paper/main.tex`：论文主文件
- `PLAN.md`：当前计划
- `PROGRESS.md`：阶段性进展与结论

## 协作前提

本仓库不包含以下内容，这些内容需要协作者自行准备：

- 赛题原件目录：`2026同济数学建模竞赛赛题/`
- 原始数据目录：`data/raw/`
- 本地 Octave 工具链：`tools/octave/`

当前 C 题核心数据加载代码固定读取下列工作簿：

- `2026同济数学建模竞赛赛题/2026C数据.xlsx`

如果这个文件不存在，大部分 C 题脚本都无法从头运行。

## 环境要求

推荐平台：Windows + VS Code。

Python 依赖通过项目本地虚拟环境管理，当前常用解释器为：

- `.venv/Scripts/python.exe`

核心 Python 包见 `requirements.txt`，包括：

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- sympy
- scikit-learn
- statsmodels
- networkx
- pulp
- openpyxl
- xlsxwriter
- tabulate
- tqdm

论文编译依赖：

- XeLaTeX
- latexmk

可选运行环境：

- Octave，用于 `scripts/matlab_examples/` 和部分 `.m` 示例

更完整的工具建议见 [docs/downloads.md](docs/downloads.md)。

## 快速开始

### 克隆后初始化

1. 创建并激活虚拟环境。
2. 安装依赖：`pip install -r requirements.txt`。
3. 准备赛题 Excel：放到 `2026同济数学建模竞赛赛题/2026C数据.xlsx`。
4. 运行冒烟检查：`python scripts/smoke_check.py`。

如果你在 VS Code 中工作，可以直接使用工作区任务：

- `Install core packages`
- `Run smoke check`
- `Build paper (XeLaTeX)`

### 论文编译

论文主文件：`reports/paper/main.tex`

推荐命令：

- `latexmk -xelatex -interaction=nonstopmode -synctex=1 -outdir=reports/paper/build reports/paper/main.tex`

## 数据缺失项与当前回退逻辑

当前 C 题数据加载逻辑在 `scripts/c_problem/load_c_data.py`。

由于原工作簿中存在缺项，本仓库已经内置以下回退规则：

1. `effective_energy_limit_J` 为空时，自动按 `battery_capacity_J - safety_reserve_J` 回填。
2. `GroundTime` 全空时，根据 `ManualPoints` 中的坐标、步行速度和绕行系数，自动生成地面时间矩阵。
3. `FlightTime` 和 `FlightEnergy` 中存在非有限边时，加载器会根据节点坐标、水平/垂直速度和分段能耗补全缺失飞行边，从而恢复问题一从零构造所需的可达图。

这意味着：

- 问题一的 f_guided 构造现在可以在 K=1 到 K=4 上直接从零生成可行航线。
- 问题二 F 引导实验可以在这组新航线之上稳定复现。
- 当前落盘结果已经包含 K-means 起点、任务级 route summary 和问题二快搜对比表。

## 当前实验状态

目前仓库里最稳定、可直接复现的是 C 题第二阶段实验链：

- F 状态引导默认策略
- F-GA
- F-PSO
- F-ACO
- 问题一路径重排 + 问题二 F 引导联合实验

问题一脚本已经改造成“F 感知构造 + K-means 起点 + 自动回退”模式，当前主结果不再依赖 `cached_fallback`。在最新结果中，`K=1` 到 `K=4` 均直接生成 `f_guided` 新解，其中 makespan 分别约为 1312.2 s、537.4 s、361.3 s 和 322.4 s。

## 实验入口

### 基础检查

- `python scripts/smoke_check.py`

### 问题一

- `python scripts/c_problem/run_problem1_heuristic.py`

输出：

- `outputs/tables/c_problem_problem1_heuristic_summary.csv`
- `outputs/tables/c_problem_problem1_route_summary.csv`
- `data/processed/c_problem_problem1_route_detail.csv`

### 问题二主实验

- `python scripts/c_problem/run_problem2_f_experiments.py`

输出核心汇总表：

- `outputs/tables/c_problem_problem2_f_guidance_compare.csv`

### F 权重搜索对比

- `python scripts/c_problem/run_problem2_ga.py`
- `python scripts/c_problem/run_problem2_pso.py`
- `python scripts/c_problem/run_problem2_aco.py`

对应输出在：

- `outputs/tables/c_problem_problem2_f_ga_compare.csv`
- `outputs/tables/c_problem_problem2_f_pso_compare.csv`
- `outputs/tables/c_problem_problem2_f_aco_compare.csv`

### 联合实验与敏感性分析

- `python scripts/c_problem/run_problem2_joint_reorder_f.py`
- `python scripts/c_problem/run_problem2_sensitivity.py`

### MATLAB / Octave 示例

- `scripts/matlab_examples/hello_demo.m`
- `scripts/matlab_examples/interpolation_demo.m`
- `scripts/matlab_examples/topsis_demo.m`

## 协作建议

建议团队按下面的边界并行：

1. 一人维护 `reports/paper/main.tex` 与论文表述。
2. 一人维护 `scripts/c_problem/run_problem1_heuristic.py` 的问题一构造逻辑。
3. 一人维护问题二 F 引导、GA/PSO/ACO 和敏感性分析脚本。
4. 所有人共享 `data/processed/` 与 `outputs/tables/` 中的结果文件，但不要把赛题原件和原始数据提交到仓库。

## 结果文件与同步原则

可以提交：

- 代码
- 论文源码
- 已处理的结果表
- 已导出的图表
- 计划与进度文档

不要提交：

- 赛题原始附件
- 原始 Excel 数据
- 本地解释器和工具链目录
- 编辑器本地配置与缓存

这些规则已经体现在 `.gitignore` 中。
