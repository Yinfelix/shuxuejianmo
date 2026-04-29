from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import DEFAULT_WEIGHT_VECTOR, build_guidance_base_state, evaluate_guidance_weights
from run_problem2_ga import SEEDS, run_ga_with_base_state
from run_problem2_joint import _summarize_node_state


DRONE_COUNT = 4


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))

    _, _, base_node_state = build_guidance_base_state(DRONE_COUNT, data)
    baseline_summary, _ = _summarize_node_state(
        node_state=base_node_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        drone_count=DRONE_COUNT,
        solution_type="baseline",
        selected_nodes=[],
    )
    default_summary, _, _ = evaluate_guidance_weights(
        drone_count=DRONE_COUNT,
        data=data,
        base_node_state=base_node_state,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
        weight_vector=DEFAULT_WEIGHT_VECTOR,
        solution_type="f_default",
    )

    run_summaries: list[pd.DataFrame] = []
    run_details: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        summary, detail, _ = run_ga_with_base_state(
            drone_count=DRONE_COUNT,
            seed=seed,
            data=data,
            base_node_state=base_node_state,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
        )
        run_summaries.append(summary)
        run_details[int(seed)] = detail

    runs_df = pd.concat(run_summaries, ignore_index=True)
    best_index = runs_df["total_closed_loop_time_s"].astype(float).idxmin()
    best_summary = runs_df.loc[[best_index]].copy()
    best_seed = int(best_summary.iloc[0]["seed"])
    best_detail = run_details[best_seed].copy()

    compare_df = pd.DataFrame(
        [
            {
                "drone_count": DRONE_COUNT,
                "baseline_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                "f_default_s": float(default_summary.loc[0, "total_closed_loop_time_s"]),
                "f_default_direct_confirm_count": int(default_summary.loc[0, "direct_confirm_count"]),
                "f_ga_best_s": float(best_summary.iloc[0]["total_closed_loop_time_s"]),
                "f_ga_best_direct_confirm_count": int(best_summary.iloc[0]["direct_confirm_count"]),
                "f_ga_best_seed": best_seed,
                "f_ga_best_runtime_s": float(best_summary.iloc[0]["runtime_s"]),
                "f_ga_gain_vs_default_s": float(default_summary.loc[0, "total_closed_loop_time_s"]) - float(best_summary.iloc[0]["total_closed_loop_time_s"]),
                "best_selected_nodes": str(best_summary.iloc[0]["selected_direct_confirm_nodes"]),
                "best_manual_route": str(best_summary.iloc[0]["manual_route"]),
                "ground_route_method": str(best_summary.iloc[0]["ground_route_method"]),
                "weight_vector": str(best_summary.iloc[0]["weight_vector"]),
            }
        ]
    )

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    compare_path = output_dir / "c_problem_problem2_f_ga_focus_k4_compare.csv"
    runs_path = output_dir / "c_problem_problem2_f_ga_focus_k4_runs.csv"
    detail_path = processed_dir / "c_problem_problem2_f_ga_focus_k4_detail.csv"

    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    runs_df.to_csv(runs_path, index=False, encoding="utf-8-sig")
    best_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(compare_df.to_string(index=False))
    print("\nSaved compare to:", compare_path)
    print("Saved runs to:", runs_path)
    print("Saved best detail to:", detail_path)


if __name__ == "__main__":
    main()