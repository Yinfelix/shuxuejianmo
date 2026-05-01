from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "tables"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _rounded(value: float | int | None, digits: int = 1) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _table_payload(frame: pd.DataFrame, digits: int = 1) -> dict[str, list]:
    payload = frame.copy()
    for column in payload.columns:
        if pd.api.types.is_numeric_dtype(payload[column]):
            payload[column] = payload[column].map(lambda value: _rounded(value, digits))
    return {
        "headers": payload.columns.tolist(),
        "rows": payload.astype(object).where(pd.notna(payload), None).values.tolist(),
    }


def _build_problem2_journey_rows() -> pd.DataFrame:
    round1 = _read_csv(OUTPUT_DIR / "c_problem_problem2_algorithm_compare.csv")
    serial_compare = _read_csv(OUTPUT_DIR / "c_problem_problem2_serial_greedy_compare.csv")
    joint = _read_csv(OUTPUT_DIR / "c_problem_problem2_joint_summary.csv")
    swap = _read_csv(OUTPUT_DIR / "c_problem_problem2_swap_summary.csv")
    threshold = _read_csv(OUTPUT_DIR / "c_problem_problem2_threshold_sensitivity_summary.csv")
    alns = _read_csv(OUTPUT_DIR / "c_problem_problem2_alns_joint_compare.csv")
    deeper_alns_path = OUTPUT_DIR / "c_problem_problem2_alns_joint_compare_adaptive_deeper_seed37.csv"
    deeper_alns = _read_csv(deeper_alns_path) if deeper_alns_path.exists() else None

    rows: list[dict[str, object]] = []
    target_count = 16

    for drone_count in [3, 4]:
        round1_slice = round1.loc[
            (round1["drone_count"] == drone_count) & (round1["solution_type"] == "ga_hover")
        ].iloc[0]
        serial_slice = serial_compare.loc[serial_compare["drone_count"] == drone_count].iloc[0]
        joint_slice = joint.loc[joint["drone_count"] == drone_count].iloc[0]
        swap_slice = swap.loc[
            (swap["drone_count"] == drone_count) & (swap["method"] == "joint_with_swap")
        ].iloc[0]
        threshold_slice = threshold.loc[threshold["drone_count"] == drone_count].sort_values("closed_loop_time_s").iloc[0]
        alns_slice = alns.loc[alns["drone_count"] == drone_count].sort_values("optimized_total_closed_loop_s").iloc[0]

        stage_rows = [
            {
                "drone_count": drone_count,
                "stage_order": 1,
                "stage_group": "legacy",
                "stage_label": "首轮 GA 选点闭环",
                "closed_loop_time_s": float(round1_slice["total_closed_loop_time_s"]),
                "direct_confirm_count": int(round1_slice["direct_confirm_count"]),
                "manual_review_count": int(round1_slice["manual_review_count"]),
                "aux_param": "seed=11",
                "source_file": "c_problem_problem2_algorithm_compare.csv",
            },
            {
                "drone_count": drone_count,
                "stage_order": 2,
                "stage_group": "mainline",
                "stage_label": "旧默认串行策略",
                "closed_loop_time_s": float(serial_slice["old_default_total_closed_loop_s"]),
                "direct_confirm_count": int(serial_slice["old_default_direct_confirm_count"]),
                "manual_review_count": int(serial_slice["old_default_manual_review_count"]),
                "aux_param": "serial_default",
                "source_file": "c_problem_problem2_serial_greedy_compare.csv",
            },
            {
                "drone_count": drone_count,
                "stage_order": 3,
                "stage_group": "contrast",
                "stage_label": "串行贪心耗尽对照",
                "closed_loop_time_s": float(serial_slice["serial_greedy_total_closed_loop_s"]),
                "direct_confirm_count": int(serial_slice["serial_greedy_direct_confirm_count"]),
                "manual_review_count": int(serial_slice["serial_greedy_manual_review_count"]),
                "aux_param": "resource_exhaustion",
                "source_file": "c_problem_problem2_serial_greedy_compare.csv",
            },
            {
                "drone_count": drone_count,
                "stage_order": 4,
                "stage_group": "mainline",
                "stage_label": "当前联合求解器",
                "closed_loop_time_s": float(joint_slice["closed_loop_time_s"]),
                "direct_confirm_count": int(joint_slice["direct_confirm_count"]),
                "manual_review_count": int(joint_slice["manual_review_count"]),
                "aux_param": "threshold=1.0",
                "source_file": "c_problem_problem2_joint_summary.csv",
            },
            {
                "drone_count": drone_count,
                "stage_order": 5,
                "stage_group": "local_search",
                "stage_label": "交换算子增强",
                "closed_loop_time_s": float(swap_slice["closed_loop_time_s"]),
                "direct_confirm_count": int(swap_slice["direct_confirm_count"]),
                "manual_review_count": int(swap_slice["manual_review_count"]),
                "aux_param": f"swap_moves={int(swap_slice['swap_move_count'])}",
                "source_file": "c_problem_problem2_swap_summary.csv",
            },
            {
                "drone_count": drone_count,
                "stage_order": 6,
                "stage_group": "sensitivity_branch",
                "stage_label": "阈值敏感性最优",
                "closed_loop_time_s": float(threshold_slice["closed_loop_time_s"]),
                "direct_confirm_count": int(threshold_slice["direct_confirm_count"]),
                "manual_review_count": int(threshold_slice["manual_review_count"]),
                "aux_param": f"multiplier={float(threshold_slice['threshold_multiplier']):.1f}",
                "source_file": "c_problem_problem2_threshold_sensitivity_summary.csv",
            },
            {
                "drone_count": drone_count,
                "stage_order": 7,
                "stage_group": "alns_branch",
                "stage_label": "ALNS 当前最优",
                "closed_loop_time_s": float(alns_slice["optimized_total_closed_loop_s"]),
                "direct_confirm_count": int(alns_slice["optimized_direct_confirm_count"]),
                "manual_review_count": target_count - int(alns_slice["optimized_direct_confirm_count"]),
                "aux_param": f"seed={int(alns_slice['seed'])}",
                "source_file": "c_problem_problem2_alns_joint_compare.csv",
            },
        ]

        if deeper_alns is not None and (deeper_alns["drone_count"] == drone_count).any():
            deeper_slice = deeper_alns.loc[
                deeper_alns["drone_count"] == drone_count
            ].sort_values("optimized_total_closed_loop_s").iloc[0]
            stage_rows.append(
                {
                    "drone_count": drone_count,
                    "stage_order": 8,
                    "stage_group": "alns_validation",
                    "stage_label": "ALNS deeper复验",
                    "closed_loop_time_s": float(deeper_slice["optimized_total_closed_loop_s"]),
                    "direct_confirm_count": int(deeper_slice["optimized_direct_confirm_count"]),
                    "manual_review_count": target_count - int(deeper_slice["optimized_direct_confirm_count"]),
                    "aux_param": f"seed={int(deeper_slice['seed'])},iter=12",
                    "source_file": "c_problem_problem2_alns_joint_compare_adaptive_deeper_seed37.csv",
                }
            )

        first_value = stage_rows[0]["closed_loop_time_s"]
        previous_value: float | None = None
        for row in stage_rows:
            current_value = float(row["closed_loop_time_s"])
            row["gain_vs_first_s"] = float(first_value) - current_value
            row["gain_vs_first_pct"] = (float(first_value) - current_value) / float(first_value) * 100.0
            row["gain_vs_previous_s"] = None if previous_value is None else previous_value - current_value
            previous_value = current_value
            rows.append(row)

    return pd.DataFrame(rows)


def build_outputs() -> tuple[Path, Path]:
    problem1 = _read_csv(OUTPUT_DIR / "c_problem_problem1_method_comparison.csv")
    round1 = _read_csv(OUTPUT_DIR / "c_problem_problem2_algorithm_compare.csv")
    serial_compare = _read_csv(OUTPUT_DIR / "c_problem_problem2_serial_greedy_compare.csv")
    joint = _read_csv(OUTPUT_DIR / "c_problem_problem2_joint_summary.csv")
    threshold = _read_csv(OUTPUT_DIR / "c_problem_problem2_threshold_sensitivity_summary.csv")
    swap_compare = _read_csv(OUTPUT_DIR / "c_problem_problem2_swap_comparison.csv")
    alns = _read_csv(OUTPUT_DIR / "c_problem_problem2_alns_joint_compare.csv")
    deeper_alns_path = OUTPUT_DIR / "c_problem_problem2_alns_joint_compare_adaptive_deeper_seed37.csv"
    deeper_alns = _read_csv(deeper_alns_path) if deeper_alns_path.exists() else None
    journey = _build_problem2_journey_rows()

    best_alns = (
        alns.sort_values(["drone_count", "optimized_total_closed_loop_s"]).groupby("drone_count", as_index=False).first()
    )
    best_threshold = (
        threshold.sort_values(["drone_count", "closed_loop_time_s"]).groupby("drone_count", as_index=False).first()
    )

    structured_payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "workspace": str(ROOT),
            "description": "C 题全优化流程实验汇总，按 TSX 面板式分节组织。",
            "source_files": [
                "c_problem_problem1_method_comparison.csv",
                "c_problem_problem2_algorithm_compare.csv",
                "c_problem_problem2_serial_greedy_compare.csv",
                "c_problem_problem2_joint_summary.csv",
                "c_problem_problem2_threshold_sensitivity_summary.csv",
                "c_problem_problem2_swap_comparison.csv",
                "c_problem_problem2_alns_joint_compare.csv",
                "c_problem_problem2_alns_joint_compare_adaptive_deeper_seed37.csv",
            ],
        },
        "summary_stats": [
            {
                "label": "K=3 当前联合求解",
                "value": _rounded(joint.loc[joint["drone_count"] == 3, "closed_loop_time_s"].iloc[0]),
                "unit": "s",
            },
            {
                "label": "K=4 当前联合求解",
                "value": _rounded(joint.loc[joint["drone_count"] == 4, "closed_loop_time_s"].iloc[0]),
                "unit": "s",
            },
            {
                "label": "K=4 交换增强收益",
                "value": _rounded(swap_compare.loc[swap_compare["drone_count"] == 4, "closed_loop_gain_s"].iloc[0]),
                "unit": "s",
            },
            {
                "label": "K=4 ALNS 当前最优",
                "value": _rounded(best_alns.loc[best_alns["drone_count"] == 4, "optimized_total_closed_loop_s"].iloc[0]),
                "unit": "s",
            },
            {
                "label": "K=4 ALNS deeper复验",
                "value": _rounded(
                    deeper_alns.loc[deeper_alns["drone_count"] == 4, "optimized_total_closed_loop_s"].iloc[0]
                ) if deeper_alns is not None and (deeper_alns["drone_count"] == 4).any() else None,
                "unit": "s",
            },
        ],
        "tables": {
            "problem1_method_compare": _table_payload(problem1, digits=2),
            "problem2_round1_compare": _table_payload(round1[[
                "drone_count",
                "solution_type",
                "total_closed_loop_time_s",
                "direct_confirm_count",
                "manual_review_count",
            ]], digits=2),
            "problem2_serial_compare": _table_payload(serial_compare, digits=2),
            "problem2_current_joint": _table_payload(joint, digits=2),
            "problem2_threshold_best": _table_payload(best_threshold, digits=2),
            "problem2_swap_compare": _table_payload(swap_compare, digits=2),
            "problem2_alns_best": _table_payload(best_alns[[
                "seed",
                "drone_count",
                "optimized_total_closed_loop_s",
                "gain_vs_original_s",
                "optimized_direct_confirm_count",
                "optimized_ground_completion_time_s",
                "accepted_move_count",
                "final_move_tag",
            ]], digits=2),
            "journey_compare_k3": _table_payload(journey.loc[journey["drone_count"] == 3].drop(columns=["drone_count"]), digits=2),
            "journey_compare_k4": _table_payload(journey.loc[journey["drone_count"] == 4].drop(columns=["drone_count"]), digits=2),
        },
    }

    json_path = OUTPUT_DIR / "c_problem_full_optimization_report.json"
    csv_path = OUTPUT_DIR / "c_problem_full_optimization_journey_compare.csv"

    json_path.write_text(json.dumps(structured_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    journey.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path


def main() -> None:
    json_path, csv_path = build_outputs()
    print("Full optimization report saved to:", json_path)
    print("Optimization journey comparison saved to:", csv_path)


if __name__ == "__main__":
    main()