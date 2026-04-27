# Math Modeling School Contest Workbench

This workspace is prepared for school-level mathematical modeling contests in VS Code on Windows.

## Included

- A Python and Jupyter oriented analysis workflow
- A report writing path with a ready-to-edit LaTeX paper template
- Recommended VS Code extensions and workspace settings
- A custom Copilot agent and skill for mathematical modeling tasks

## Quick Start

1. Accept the workspace extension recommendations in VS Code.
2. Use the project-local `.venv` environment for this folder.
3. Install the packages listed in `requirements.txt`.
4. Put datasets in `data/raw/` and cleaned outputs in `data/processed/`.
5. Keep notebook drafts in `notebooks/` and reusable scripts in `scripts/`.
6. Export charts to `outputs/figures/` and tables to `outputs/tables/`.
7. Write the final report in `reports/paper/main.tex`.
8. Use the `Math Modeling Coach` agent or call `/math-modeling` in chat.

## Layout

```text
.
|-- .github/
|   |-- agents/
|   `-- skills/
|-- .vscode/
|-- data/
|   |-- processed/
|   `-- raw/
|-- docs/
|-- notebooks/
|-- outputs/
|   |-- figures/
|   `-- tables/
|-- reports/
|   `-- paper/
|-- scripts/
|-- .gitignore
`-- requirements.txt
```

See `docs/downloads.md` for the recommended external tools and reference sources.

## Python Environment

This workspace is configured to use `.venv/Scripts/python.exe` as its default interpreter on Windows.
This keeps modeling dependencies isolated from the shared Conda base environment.

## External Materials Pack

An external reference bundle named `数学建模资料` has been inspected and cataloged for this workspace.

Its top level content can be grouped into four buckets:

- Contest handbooks and notices: for example `2022年亚太赛参赛手册含优秀论文(3).pdf`, `2023年MathorCup数学建模挑战赛参赛手册.pdf`, `2023年华数杯全国大学生数学建模竞赛参赛手册.pdf`, and `2023年数维杯国际大学生数学建模挑战赛报名通知.pdf`
- Excellent paper collections: for example `2022国赛优秀论文集.zip`, `2022美赛特等奖论文集.zip`, `华数杯-C题-优秀论文.rar`, `历届小美赛优秀论文汇总.zip`, and `数维杯国际赛真题及优秀论文集.rar`
- General books and tutorials: `参考书籍/`, `参考书籍.zip`, `MATLAB智能算法三十个案例分析.pdf`, `scientific-visualization-python-matplotlib.pdf`, `archivetemp王铮-matlab软件求解偏微分方程模型.pdf`
- Aggregated links and lecture packs: `资料链接(1).txt`, `学习资料篇.pdf`, `讲座.zip`, and `数学建模资料.zip`

The bundle has been extracted in place where possible and indexed by `scripts/knowledge/catalog_math_materials.py`.
Structured outputs for AI-assisted reading are stored under `knowledge/math_materials/`, including:

- `materials_inventory.md`: extracted inventory summary
- `excellent_papers_index.md`: cleaned excellent-paper collection summary
- `ai_reading_schema.json`: normalized slices for writing framework, style, formatting, and explanation patterns

Note: some legacy zip archives use older filename encodings, so a few extracted internal filenames may appear garbled on Windows. The PDF files themselves remain readable, and the knowledge summaries are based on collection names plus extracted text evidence rather than filename appearance alone.
