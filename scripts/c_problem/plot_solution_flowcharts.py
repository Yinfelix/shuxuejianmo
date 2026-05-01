from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs" / "figures"


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


BOX_EDGE = "#3a3a3a"
BOX_FACE = "#f4f4f4"
TEXT_MAIN = "#222222"
TEXT_SUB = "#4a4a4a"
LINE_COLOR = "#4f4f4f"


def _flow_box(ax, x: float, y: float, width: float, height: float, title: str, body: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=1.4,
        edgecolor=BOX_EDGE,
        facecolor=BOX_FACE,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center", fontsize=12, weight="bold", color=TEXT_MAIN)
    ax.text(x + width / 2, y + height * 0.31, body, ha="center", va="center", fontsize=9.5, color=TEXT_SUB)


def _straight_arrow(ax, start: tuple[float, float], end: tuple[float, float], text: str | None = None, text_pos: tuple[float, float] | None = None) -> None:
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=LINE_COLOR)
    ax.add_patch(patch)
    if text and text_pos:
        ax.text(text_pos[0], text_pos[1], text, ha="center", va="center", fontsize=9, color=TEXT_SUB)


def _orth_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    via_y: float | None = None,
    text: str | None = None,
    text_pos: tuple[float, float] | None = None,
) -> None:
    if via_y is None:
        via_y = (start[1] + end[1]) / 2

    first = (start[0], via_y)
    second = (end[0], via_y)
    ax.plot([start[0], first[0]], [start[1], first[1]], color=LINE_COLOR, linewidth=1.4)
    ax.plot([first[0], second[0]], [first[1], second[1]], color=LINE_COLOR, linewidth=1.4)
    patch = FancyArrowPatch(second, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=LINE_COLOR)
    ax.add_patch(patch)
    if text and text_pos:
        ax.text(text_pos[0], text_pos[1], text, ha="center", va="center", fontsize=9, color=TEXT_SUB)


def plot_problem1_flowchart() -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("问题一求解流程图", fontsize=17, weight="bold")

    _flow_box(ax, 0.05, 0.60, 0.18, 0.18, "数据输入", "节点、飞行时间、\n飞行能耗与参数表")
    _flow_box(ax, 0.29, 0.60, 0.18, 0.18, "预处理", "字段统一、可达性检查、\n收益与松弛度计算")
    _flow_box(ax, 0.53, 0.60, 0.18, 0.18, "初始构造", "K-means 起点预分组 +\n单点可行航线生成")
    _flow_box(ax, 0.77, 0.60, 0.18, 0.18, "候选评估", "插入现有任务或新开任务，\n重算时长、返航与能耗")

    _flow_box(ax, 0.09, 0.22, 0.18, 0.18, "可行性筛选", "保留满足能量、\n时长约束的动作")
    _flow_box(ax, 0.33, 0.22, 0.18, 0.18, "K≤3 主线", "构造式贪心选择\n最大完工时长最小的动作")
    _flow_box(ax, 0.57, 0.22, 0.18, 0.18, "K=4 主线", "带权聚类 + 滚动重构\n显式编码 sortie 结构")
    _flow_box(ax, 0.81, 0.22, 0.14, 0.18, "结果输出", "route summary、detail 与\n完工时长、能耗、负载离散度")

    _straight_arrow(ax, (0.23, 0.69), (0.29, 0.69))
    _straight_arrow(ax, (0.47, 0.69), (0.53, 0.69))
    _straight_arrow(ax, (0.71, 0.69), (0.77, 0.69))
    _orth_arrow(ax, (0.86, 0.60), (0.18, 0.40), via_y=0.46, text="统一筛选后进入分支", text_pos=(0.56, 0.49))
    _straight_arrow(ax, (0.27, 0.31), (0.33, 0.31))
    _straight_arrow(ax, (0.51, 0.31), (0.57, 0.31))
    _straight_arrow(ax, (0.75, 0.31), (0.81, 0.31))

    ax.text(0.44, 0.12, "低到中等资源采用构造式贪心，高资源配置采用层次滚动重构", ha="center", va="center", fontsize=9.5, color=TEXT_SUB)
    ax.text(0.84, 0.12, "统一输出并进入问题二节点级输入", ha="center", va="center", fontsize=9.5, color=TEXT_SUB)

    output_path = FIG_DIR / "c_problem_problem1_solution_flowchart.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_problem2_flowchart() -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("问题二求解流程图", fontsize=17, weight="bold")

    _flow_box(ax, 0.05, 0.60, 0.18, 0.18, "读取问题一结果", "映射所属无人机、\n所属任务与基础 node state")
    _flow_box(ax, 0.29, 0.60, 0.18, 0.18, "候选识别", "判断已确认节点与\n待补足阈值节点")
    _flow_box(ax, 0.53, 0.60, 0.18, 0.18, "边际收益计算", "估计补足缺口后减少的\n地面服务与通行成本")
    _flow_box(ax, 0.77, 0.60, 0.18, 0.18, "状态评分排序", "按 F(u,v,E) 比较\n优先级、松弛度与缺口")

    _flow_box(ax, 0.05, 0.22, 0.18, 0.18, "动作接受", "更新确认集合、\n候选集与节点状态")
    _flow_box(ax, 0.29, 0.22, 0.18, 0.18, "地面阶段重算", "每次接受动作后\n重算物业复核路径")
    _flow_box(ax, 0.53, 0.22, 0.18, 0.18, "主线方案", "全部无人机返航后\n串行完成剩余复核")
    _flow_box(ax, 0.77, 0.22, 0.18, 0.18, "扩展分析", "交换算子、阈值灵敏度与\nALNS 给出稳健性与上界")

    _straight_arrow(ax, (0.23, 0.69), (0.29, 0.69))
    _straight_arrow(ax, (0.47, 0.69), (0.53, 0.69))
    _straight_arrow(ax, (0.71, 0.69), (0.77, 0.69))
    _orth_arrow(ax, (0.86, 0.60), (0.14, 0.40), via_y=0.46, text="排序后依次接受动作", text_pos=(0.55, 0.49))
    _straight_arrow(ax, (0.23, 0.31), (0.29, 0.31))
    _straight_arrow(ax, (0.47, 0.31), (0.53, 0.31))
    _straight_arrow(ax, (0.71, 0.31), (0.77, 0.31))

    ax.text(0.38, 0.12, "每次接受动作都触发一次地面复核重算", ha="center", va="center", fontsize=9.5, color=TEXT_SUB)
    ax.text(0.73, 0.12, "统一以总闭环完成时间为最终判据", ha="center", va="center", fontsize=9.5, color=TEXT_SUB)

    output_path = FIG_DIR / "c_problem_problem2_solution_flowchart.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    outputs = [plot_problem1_flowchart(), plot_problem2_flowchart()]
    for output in outputs:
        print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()