from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_problem2_joint import solve_problem2


def _tag_result(summary_df: pd.DataFrame, target_df: pd.DataFrame, route_df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = summary_df.copy()
    target = target_df.copy()
    route = route_df.copy()
    summary.insert(0, "method", method)
    target.insert(0, "method", method)
    route.insert(0, "method", method)
    return summary, target, route


def main() -> None:
    baseline = _tag_result(*solve_problem2(enable_swap=False), method="joint_baseline")
    swap_enhanced = _tag_result(*solve_problem2(enable_swap=True), method="joint_with_swap")

    summary_df = pd.concat([baseline[0], swap_enhanced[0]], ignore_index=True)
    target_df = pd.concat([baseline[1], swap_enhanced[1]], ignore_index=True)
    route_df = pd.concat([baseline[2], swap_enhanced[2]], ignore_index=True)

    compare_df = (
        summary_df.pivot_table(
            index=["threshold_multiplier", "drone_count"],
            columns="method",
            values=["closed_loop_time_s", "air_stage_time_s", "swap_move_count", "swap_improvement_s"],
            aggfunc="first",
        )
        .sort_index()
    )
    compare_df.columns = ["_".join(str(part) for part in column if part).strip("_") for column in compare_df.columns.to_flat_index()]
    compare_df = compare_df.reset_index()
    compare_df["closed_loop_gain_s"] = (
        compare_df["closed_loop_time_s_joint_baseline"] - compare_df["closed_loop_time_s_joint_with_swap"]
    )
    compare_df["air_stage_gain_s"] = compare_df["air_stage_time_s_joint_baseline"] - compare_df["air_stage_time_s_joint_with_swap"]

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c_problem_problem2_swap_summary.csv"
    compare_path = output_dir / "c_problem_problem2_swap_comparison.csv"
    target_path = processed_dir / "c_problem_problem2_swap_target_detail.csv"
    route_path = processed_dir / "c_problem_problem2_swap_route_detail.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    target_df.to_csv(target_path, index=False, encoding="utf-8-sig")
    route_df.to_csv(route_path, index=False, encoding="utf-8-sig")

    print("Problem 2 swap experiment summary saved to:", summary_path)
    print(summary_df.to_string(index=False))
    print("\nProblem 2 swap comparison saved to:", compare_path)
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()