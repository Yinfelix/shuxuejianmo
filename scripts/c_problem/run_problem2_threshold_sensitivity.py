from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_problem2_joint import solve_problem2


THRESHOLD_MULTIPLIERS = [0.8, 1.0, 1.2, 1.4]


def main() -> None:
    summary_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    route_frames: list[pd.DataFrame] = []

    for threshold_multiplier in THRESHOLD_MULTIPLIERS:
        summary_table, target_table, route_table = solve_problem2(threshold_multiplier=threshold_multiplier)
        summary_frames.append(summary_table)
        target_frames.append(target_table)
        route_frames.append(route_table)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    target_df = pd.concat(target_frames, ignore_index=True)
    route_df = pd.concat(route_frames, ignore_index=True)

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c_problem_problem2_threshold_sensitivity_summary.csv"
    target_path = processed_dir / "c_problem_problem2_threshold_sensitivity_target_detail.csv"
    route_path = processed_dir / "c_problem_problem2_threshold_sensitivity_route_detail.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    target_df.to_csv(target_path, index=False, encoding="utf-8-sig")
    route_df.to_csv(route_path, index=False, encoding="utf-8-sig")

    print("Problem 2 threshold sensitivity summary saved to:", summary_path)
    print(summary_df.to_string(index=False))
    print("\nProblem 2 threshold sensitivity target detail saved to:", target_path)
    print(target_df.head(20).to_string(index=False))
    print("\nProblem 2 threshold sensitivity route detail saved to:", route_path)
    print(route_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()