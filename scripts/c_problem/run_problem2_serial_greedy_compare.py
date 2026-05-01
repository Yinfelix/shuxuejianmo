from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import DEFAULT_WEIGHT_VECTOR, build_guidance_base_state, evaluate_guidance_weights
from run_problem2_joint import _summarize_node_state


K_VALUES = [1, 2, 3, 4]


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    battery_swap_time = float(parameter_value(data.params, "battery_swap_time_s"))

    compare_rows: list[dict[str, float | int | str]] = []
    detail_rows: list[pd.DataFrame] = []

    for drone_count in K_VALUES:
        _, _, base_node_state = build_guidance_base_state(drone_count, data)
        baseline_summary, _ = _summarize_node_state(
            node_state=base_node_state,
            manual_points=data.manual_points,
            ground_time=data.ground_time,
            drone_count=drone_count,
            solution_type="baseline_serial_rule",
            selected_nodes=[],
        )
        old_summary, old_detail, _ = evaluate_guidance_weights(
            drone_count=drone_count,
            data=data,
            base_node_state=base_node_state,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            weight_vector=DEFAULT_WEIGHT_VECTOR,
            solution_type="old_default_serial",
            ground_mode="serial",
            battery_swap_time_s=battery_swap_time,
            prefer_resource_exhaustion=False,
        )
        greedy_summary, greedy_detail, _ = evaluate_guidance_weights(
            drone_count=drone_count,
            data=data,
            base_node_state=base_node_state,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            weight_vector=DEFAULT_WEIGHT_VECTOR,
            solution_type="serial_greedy_exhaust",
            ground_mode="serial",
            battery_swap_time_s=battery_swap_time,
            prefer_resource_exhaustion=True,
        )

        compare_rows.append(
            {
                "drone_count": drone_count,
                "baseline_total_closed_loop_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                "baseline_direct_confirm_count": int(baseline_summary.loc[0, "direct_confirm_count"]),
                "old_default_total_closed_loop_s": float(old_summary.loc[0, "total_closed_loop_time_s"]),
                "old_default_direct_confirm_count": int(old_summary.loc[0, "direct_confirm_count"]),
                "old_default_manual_review_count": int(old_summary.loc[0, "manual_review_count"]),
                "old_default_selected_nodes": str(old_summary.loc[0, "selected_direct_confirm_nodes"]),
                "serial_greedy_total_closed_loop_s": float(greedy_summary.loc[0, "total_closed_loop_time_s"]),
                "serial_greedy_direct_confirm_count": int(greedy_summary.loc[0, "direct_confirm_count"]),
                "serial_greedy_manual_review_count": int(greedy_summary.loc[0, "manual_review_count"]),
                "serial_greedy_selected_nodes": str(greedy_summary.loc[0, "selected_direct_confirm_nodes"]),
                "greedy_vs_old_delta_s": float(old_summary.loc[0, "total_closed_loop_time_s"]) - float(greedy_summary.loc[0, "total_closed_loop_time_s"]),
                "ground_mode": str(greedy_summary.loc[0, "ground_mode"]),
            }
        )

        old_detail = old_detail.copy()
        old_detail["compare_profile"] = "old_default_serial"
        detail_rows.append(old_detail)
        greedy_detail = greedy_detail.copy()
        greedy_detail["compare_profile"] = "serial_greedy_exhaust"
        detail_rows.append(greedy_detail)

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    compare_path = output_dir / "c_problem_problem2_serial_greedy_compare.csv"
    detail_path = processed_dir / "c_problem_problem2_serial_greedy_detail.csv"
    pd.DataFrame(compare_rows).to_csv(compare_path, index=False, encoding="utf-8-sig")
    pd.concat(detail_rows, ignore_index=True).to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Problem 2 serial-rule comparison saved to:", compare_path)
    print(pd.DataFrame(compare_rows).to_string(index=False))
    print("\nProblem 2 serial-rule detail saved to:", detail_path)


if __name__ == "__main__":
    main()