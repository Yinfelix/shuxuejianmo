from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import DEFAULT_WEIGHT_VECTOR, build_guidance_base_state, evaluate_guidance_weights
from run_problem2_aco import SEEDS as ACO_SEEDS, run_aco_with_base_state
from run_problem2_ga import SEEDS as GA_SEEDS, run_ga_with_base_state
from run_problem2_joint import _summarize_node_state
from run_problem2_pso import SEEDS as PSO_SEEDS, run_pso_with_base_state


def _best_of_runs(run_tables: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(run_tables, ignore_index=True)
    best_index = combined["total_closed_loop_time_s"].astype(float).idxmin()
    return combined.loc[[best_index]].copy()


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    k_values = [3, 4]

    compare_rows: list[dict[str, float | int | str]] = []
    default_rows: list[pd.DataFrame] = []
    ga_rows: list[pd.DataFrame] = []
    pso_rows: list[pd.DataFrame] = []
    aco_rows: list[pd.DataFrame] = []

    for drone_count in k_values:
        _, _, base_node_state = build_guidance_base_state(drone_count, data)
        baseline_summary, _ = _summarize_node_state(
            node_state=base_node_state,
            manual_points=data.manual_points,
            ground_time=data.ground_time,
            drone_count=drone_count,
            solution_type="baseline",
            selected_nodes=[],
        )
        default_summary, _, _ = evaluate_guidance_weights(
            drone_count=drone_count,
            data=data,
            base_node_state=base_node_state,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            weight_vector=DEFAULT_WEIGHT_VECTOR,
            solution_type="f_default",
        )
        default_rows.append(default_summary)

        ga_runs = [run_ga_with_base_state(drone_count, seed, data, base_node_state, horizon, energy_limit, hover_power)[0] for seed in GA_SEEDS]
        pso_runs = [run_pso_with_base_state(drone_count, seed, data, base_node_state, horizon, energy_limit, hover_power)[0] for seed in PSO_SEEDS]
        aco_runs = [run_aco_with_base_state(drone_count, seed, data, base_node_state, horizon, energy_limit, hover_power)[0] for seed in ACO_SEEDS]

        best_ga = _best_of_runs(ga_runs)
        best_pso = _best_of_runs(pso_runs)
        best_aco = _best_of_runs(aco_runs)

        ga_rows.extend(ga_runs)
        pso_rows.extend(pso_runs)
        aco_rows.extend(aco_runs)

        compare_rows.append(
            {
                "drone_count": drone_count,
                "baseline_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                "f_default_s": float(default_summary.loc[0, "total_closed_loop_time_s"]),
                "f_default_direct_confirm_count": int(default_summary.loc[0, "direct_confirm_count"]),
                "f_ga_s": float(best_ga.loc[best_ga.index[0], "total_closed_loop_time_s"]),
                "f_ga_direct_confirm_count": int(best_ga.loc[best_ga.index[0], "direct_confirm_count"]),
                "f_pso_s": float(best_pso.loc[best_pso.index[0], "total_closed_loop_time_s"]),
                "f_pso_direct_confirm_count": int(best_pso.loc[best_pso.index[0], "direct_confirm_count"]),
                "f_aco_s": float(best_aco.loc[best_aco.index[0], "total_closed_loop_time_s"]),
                "f_aco_direct_confirm_count": int(best_aco.loc[best_aco.index[0], "direct_confirm_count"]),
                "best_algorithm": min(
                    {
                        "GA": float(best_ga.loc[best_ga.index[0], "total_closed_loop_time_s"]),
                        "PSO": float(best_pso.loc[best_pso.index[0], "total_closed_loop_time_s"]),
                        "ACO": float(best_aco.loc[best_aco.index[0], "total_closed_loop_time_s"]),
                    }.items(),
                    key=lambda item: item[1],
                )[0],
            }
        )

    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(compare_rows).to_csv(output_dir / "c_problem_problem2_f_guidance_compare.csv", index=False, encoding="utf-8-sig")
    pd.concat(default_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_default_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat(ga_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_guidance_ga_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat(pso_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_guidance_pso_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat(aco_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_guidance_aco_runs.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()