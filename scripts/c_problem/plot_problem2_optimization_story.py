from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "outputs" / "tables"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _annotate_bars(ax, values: list[float], fmt: str = "{:.1f}") -> None:
    for index, value in enumerate(values):
        ax.text(index, value + max(values) * 0.02, fmt.format(value), ha="center", va="bottom", fontsize=9)


def _load_story_context() -> dict[str, pd.DataFrame | pd.Series | int]:
    journey = _read_csv(TABLE_DIR / "c_problem_full_optimization_journey_compare.csv")
    f_guidance = _read_csv(TABLE_DIR / "c_problem_problem2_f_guidance_compare.csv")
    joint = _read_csv(TABLE_DIR / "c_problem_problem2_joint_summary.csv")
    swap = _read_csv(TABLE_DIR / "c_problem_problem2_swap_summary.csv")
    threshold = _read_csv(TABLE_DIR / "c_problem_problem2_threshold_sensitivity_summary.csv")
    alns_compare = _read_csv(TABLE_DIR / "c_problem_problem2_alns_joint_compare.csv")
    target_detail = _read_csv(PROCESSED_DIR / "c_problem_problem2_target_detail.csv")
    alns_detail = _read_csv(PROCESSED_DIR / "c_problem_problem2_alns_joint_detail.csv")

    best_alns = alns_compare.loc[alns_compare["drone_count"] == 4].sort_values("optimized_total_closed_loop_s").iloc[0]
    best_threshold = threshold.loc[threshold["drone_count"] == 4].sort_values("closed_loop_time_s").iloc[0]
    swap_k4 = swap.loc[(swap["drone_count"] == 4) & (swap["method"] == "joint_with_swap")].iloc[0]
    joint_k4 = joint.loc[joint["drone_count"] == 4].iloc[0]
    f_k4 = f_guidance.loc[f_guidance["drone_count"] == 4].iloc[0]

    return {
        "journey_k4": journey.loc[journey["drone_count"] == 4].copy(),
        "f_k4": f_k4,
        "joint_k4": joint_k4,
        "swap_k4": swap_k4,
        "best_threshold": best_threshold,
        "best_alns": best_alns,
        "alns_seed_runs": alns_compare.loc[alns_compare["drone_count"] == 4].sort_values("seed").copy(),
        "default_target": target_detail.loc[target_detail["drone_count"] == 4].copy(),
        "best_alns_target": alns_detail.loc[
            (alns_detail["drone_count"] == 4) & (alns_detail["seed"] == int(best_alns["seed"]))
        ].copy(),
    }


def _build_scheme_summary(context: dict[str, pd.DataFrame | pd.Series | int]) -> pd.DataFrame:
    f_k4 = context["f_k4"]
    joint_k4 = context["joint_k4"]
    swap_k4 = context["swap_k4"]
    best_threshold = context["best_threshold"]
    best_alns = context["best_alns"]

    return pd.DataFrame(
        [
            {
                "scheme": "Default F",
                "closed_loop_time_s": float(f_k4["f_default_s"]),
                "direct_confirm_count": int(f_k4["f_default_direct_confirm_count"]),
                "manual_review_count": 16 - int(f_k4["f_default_direct_confirm_count"]),
                "route_count": 4,
                "extra_hover_time_s": float(joint_k4["extra_hover_time_s"]),
                "air_stage_time_s": float(joint_k4["air_stage_time_s"]),
                "ground_stage_time_s": float(joint_k4["ground_stage_time_s"]),
            },
            {
                "scheme": "Joint",
                "closed_loop_time_s": float(joint_k4["closed_loop_time_s"]),
                "direct_confirm_count": int(joint_k4["direct_confirm_count"]),
                "manual_review_count": int(joint_k4["manual_review_count"]),
                "route_count": 4,
                "extra_hover_time_s": float(joint_k4["extra_hover_time_s"]),
                "air_stage_time_s": float(joint_k4["air_stage_time_s"]),
                "ground_stage_time_s": float(joint_k4["ground_stage_time_s"]),
            },
            {
                "scheme": "Swap",
                "closed_loop_time_s": float(swap_k4["closed_loop_time_s"]),
                "direct_confirm_count": int(swap_k4["direct_confirm_count"]),
                "manual_review_count": int(swap_k4["manual_review_count"]),
                "route_count": 4,
                "extra_hover_time_s": float(swap_k4["extra_hover_time_s"]),
                "air_stage_time_s": float(swap_k4["air_stage_time_s"]),
                "ground_stage_time_s": float(swap_k4["ground_stage_time_s"]),
            },
            {
                "scheme": "Threshold x0.8",
                "closed_loop_time_s": float(best_threshold["closed_loop_time_s"]),
                "direct_confirm_count": int(best_threshold["direct_confirm_count"]),
                "manual_review_count": int(best_threshold["manual_review_count"]),
                "route_count": 4,
                "extra_hover_time_s": float(best_threshold["extra_hover_time_s"]),
                "air_stage_time_s": float(best_threshold["air_stage_time_s"]),
                "ground_stage_time_s": float(best_threshold["ground_stage_time_s"]),
            },
            {
                "scheme": "ALNS best",
                "closed_loop_time_s": float(best_alns["optimized_total_closed_loop_s"]),
                "direct_confirm_count": int(best_alns["optimized_direct_confirm_count"]),
                "manual_review_count": 16 - int(best_alns["optimized_direct_confirm_count"]),
                "route_count": int(best_alns["optimized_route_count"]),
                "extra_hover_time_s": float(best_threshold["extra_hover_time_s"]),
                "air_stage_time_s": float(best_alns["optimized_air_completion_time_s"]),
                "ground_stage_time_s": float(best_alns["optimized_ground_completion_time_s"]),
            },
        ]
    )


def _plot_status_map(ax, coords: pd.DataFrame, detail: pd.DataFrame, title: str) -> None:
    merged = coords.merge(detail, on="node_id", how="inner", suffixes=("_coord", "_detail"))
    merged["node_label"] = merged["node_id"].map(lambda value: f"B{int(value):02d}")
    direct_mask = merged["is_direct_confirmed"] if "is_direct_confirmed" in merged.columns else merged["direct_confirmed"]
    priority_column = "priority_weight_detail" if "priority_weight_detail" in merged.columns else "priority_weight_coord"
    priority = merged[priority_column].astype(float)
    sizes = 110 + priority * 45

    ax.scatter(
        merged.loc[~direct_mask, "x_m"],
        merged.loc[~direct_mask, "y_m"],
        s=sizes.loc[~direct_mask],
        c="#d95f02",
        marker="X",
        edgecolors="white",
        linewidths=0.9,
        label="Manual review",
        alpha=0.95,
    )
    ax.scatter(
        merged.loc[direct_mask, "x_m"],
        merged.loc[direct_mask, "y_m"],
        s=sizes.loc[direct_mask],
        c="#1b9e77",
        marker="o",
        edgecolors="white",
        linewidths=0.9,
        label="Direct confirm",
        alpha=0.95,
    )
    for row in merged.itertuples(index=False):
        ax.text(float(row.x_m) + 2.5, float(row.y_m) + 2.5, str(int(row.node_id)), fontsize=8, color="#333333")

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def plot_storyboard(context: dict[str, pd.DataFrame | pd.Series | int], data) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    journey_k4 = context["journey_k4"].copy()
    summary = _build_scheme_summary(context)
    coords = data.nodes.loc[data.nodes["node_id"].notna(), ["node_id", "x_m", "y_m", "priority_weight"]].copy()
    coords["node_id"] = coords["node_id"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("K=4 Optimization Storyboard", fontsize=16, weight="bold")

    stage_labels = [
        "GA",
        "F",
        "Joint",
        "Swap",
        "Thres",
        "ALNS",
    ]
    stage_values = journey_k4["closed_loop_time_s"].tolist()
    stage_colors = ["#9e9e9e", "#7570b3", "#1f78b4", "#66a61e", "#e6ab02", "#d95f02"]
    axes[0, 0].bar(stage_labels, stage_values, color=stage_colors)
    axes[0, 0].set_title("Journey of closed-loop time", fontsize=12, weight="bold")
    axes[0, 0].set_ylabel("seconds")
    axes[0, 0].grid(axis="y", alpha=0.2, linestyle="--")
    _annotate_bars(axes[0, 0], stage_values)
    axes[0, 0].text(4.7, min(stage_values) + 55, "Best observed\n2997.6 s", ha="center", va="bottom", fontsize=10, color="#d95f02")

    scheme_positions = list(range(len(summary)))
    axes[0, 1].bar(scheme_positions, summary["direct_confirm_count"], color="#1b9e77", label="Direct")
    axes[0, 1].bar(
        scheme_positions,
        summary["manual_review_count"],
        bottom=summary["direct_confirm_count"],
        color="#d95f02",
        label="Manual",
    )
    axes[0, 1].set_xticks(scheme_positions, summary["scheme"], rotation=12)
    axes[0, 1].set_title("Direct-confirm vs manual-review count", fontsize=12, weight="bold")
    axes[0, 1].set_ylabel("node count")
    axes[0, 1].set_ylim(0, 18)
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.2, linestyle="--")

    _plot_status_map(axes[1, 0], coords, context["default_target"], "Default F node status")
    alns_target = context["best_alns_target"].rename(columns={"direct_confirmed": "is_direct_confirmed"})
    _plot_status_map(axes[1, 1], coords, alns_target, "Best ALNS node status")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = FIG_DIR / "c_problem_problem2_k4_best_solution_storyboard.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_diagnostics(context: dict[str, pd.DataFrame | pd.Series | int]) -> Path:
    summary = _build_scheme_summary(context)
    alns_runs = context["alns_seed_runs"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("K=4 Best-solution diagnostics", fontsize=16, weight="bold")

    selected = summary.loc[summary["scheme"].isin(["Joint", "Swap", "Threshold x0.8", "ALNS best"])].reset_index(drop=True)
    x_pos = list(range(len(selected)))
    width = 0.36
    axes[0, 0].bar([value - width / 2 for value in x_pos], selected["air_stage_time_s"], width=width, color="#1f78b4", label="Air")
    axes[0, 0].bar([value + width / 2 for value in x_pos], selected["ground_stage_time_s"], width=width, color="#ff7f00", label="Ground")
    axes[0, 0].set_xticks(x_pos, selected["scheme"], rotation=10)
    axes[0, 0].set_title("Air vs ground completion time", fontsize=12, weight="bold")
    axes[0, 0].set_ylabel("seconds")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.2, linestyle="--")

    axes[0, 1].bar(summary["scheme"], summary["closed_loop_time_s"], color=["#7570b3", "#1f78b4", "#66a61e", "#e6ab02", "#d95f02"])
    axes[0, 1].set_title("Closed-loop time by scheme", fontsize=12, weight="bold")
    axes[0, 1].set_ylabel("seconds")
    axes[0, 1].tick_params(axis="x", rotation=12)
    axes[0, 1].grid(axis="y", alpha=0.2, linestyle="--")
    for idx, value in enumerate(summary["closed_loop_time_s"]):
        axes[0, 1].text(idx, value + 25, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    axes[1, 0].plot(alns_runs["seed"].astype(int), alns_runs["optimized_total_closed_loop_s"], marker="o", color="#d95f02", linewidth=2)
    axes[1, 0].axhline(float(context["joint_k4"]["closed_loop_time_s"]), color="#1f78b4", linestyle="--", linewidth=1.5, label="Joint baseline")
    axes[1, 0].set_title("ALNS seed sensitivity", fontsize=12, weight="bold")
    axes[1, 0].set_xlabel("seed")
    axes[1, 0].set_ylabel("optimized closed-loop time / s")
    axes[1, 0].grid(alpha=0.2, linestyle="--")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(alns_runs["seed"].astype(str), alns_runs["optimized_route_count"], color="#66a61e", label="Route count")
    twin = axes[1, 1].twinx()
    twin.plot(alns_runs["seed"].astype(str), alns_runs["accepted_move_count"], color="#7570b3", marker="s", linewidth=2, label="Accepted moves")
    axes[1, 1].set_title("ALNS route split and accepted moves", fontsize=12, weight="bold")
    axes[1, 1].set_ylabel("route count")
    twin.set_ylabel("accepted moves")
    axes[1, 1].grid(axis="y", alpha=0.2, linestyle="--")
    axes[1, 1].legend(frameon=False, loc="upper left")
    twin.legend(frameon=False, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = FIG_DIR / "c_problem_problem2_k4_best_solution_diagnostics.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _flow_box(ax, x: float, y: float, width: float, height: float, title: str, body: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color,
        alpha=0.16,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", fontsize=11, weight="bold", color="#222222")
    ax.text(x + width / 2, y + height * 0.33, body, ha="center", va="center", fontsize=9, color="#333333")


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#555555")
    ax.add_patch(patch)


def plot_flowchart(context: dict[str, pd.DataFrame | pd.Series | int]) -> Path:
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Problem 2 optimization journey flowchart", fontsize=16, weight="bold")

    _flow_box(ax, 0.04, 0.58, 0.18, 0.22, "Problem 1 routes", "K-means start\nroute grouping", "#8da0cb")
    _flow_box(ax, 0.28, 0.58, 0.18, 0.22, "Unified F baseline", "K=4 closed loop\n3540.7 s", "#7570b3")
    _flow_box(ax, 0.52, 0.58, 0.18, 0.22, "Joint solver", "direct-confirm +\nground reroute\n3469.5 s", "#1f78b4")
    _flow_box(ax, 0.76, 0.58, 0.18, 0.22, "Swap local search", "cross-route swap\n3442.8 s", "#66a61e")
    _flow_box(ax, 0.52, 0.18, 0.18, 0.22, "Threshold branch", "multiplier = 0.8\n3349.8 s", "#e6ab02")
    _flow_box(ax, 0.76, 0.18, 0.18, 0.22, "ALNS best", "seed 23, 8 routes\n2997.6 s\n12 direct / 4 manual", "#d95f02")

    _arrow(ax, (0.22, 0.69), (0.28, 0.69))
    _arrow(ax, (0.46, 0.69), (0.52, 0.69))
    _arrow(ax, (0.70, 0.69), (0.76, 0.69))
    _arrow(ax, (0.61, 0.58), (0.61, 0.40))
    _arrow(ax, (0.85, 0.58), (0.85, 0.40))
    _arrow(ax, (0.70, 0.29), (0.76, 0.29))

    ax.text(0.61, 0.46, "parameter branch", ha="center", va="center", fontsize=9, color="#666666")
    ax.text(0.85, 0.46, "global search branch", ha="center", va="center", fontsize=9, color="#666666")

    output_path = FIG_DIR / "c_problem_problem2_optimization_journey_flowchart.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    context = _load_story_context()
    outputs = [
        plot_storyboard(context, data),
        plot_diagnostics(context),
        plot_flowchart(context),
    ]
    for output in outputs:
        print("Saved figure:", output)


if __name__ == "__main__":
    main()