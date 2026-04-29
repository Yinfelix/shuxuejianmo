from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_problem2_aco as aco_module
import run_problem2_ga as ga_module
import run_problem2_pso as pso_module
from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import DEFAULT_WEIGHT_VECTOR, build_guidance_base_state, evaluate_guidance_weights
from run_problem2_joint import _summarize_node_state


LIGHT_SEED = 11
LIGHT_K_VALUES = [3, 4]


def _best_of_runs(run_tables: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(run_tables, ignore_index=True)
    best_index = combined["total_closed_loop_time_s"].astype(float).idxmin()
    return combined.loc[[best_index]].copy()


def _configure_light_search() -> None:
    ga_module.POPULATION_SIZE = 8
    ga_module.GENERATIONS = 8
    ga_module.ELITE_COUNT = 2
    ga_module.MUTATION_RATE = 0.24
    ga_module.MUTATION_STEP = 0.25

    pso_module.PARTICLE_COUNT = 8
    pso_module.ITERATIONS = 10
    pso_module.INERTIA = 0.68
    pso_module.COGNITIVE = 1.2
    pso_module.SOCIAL = 1.2
    pso_module.VELOCITY_CLAMP = 0.45

    aco_module.ANT_COUNT = 8
    aco_module.ITERATIONS = 8
    aco_module.TOP_ANTS = 2
    aco_module.EVAPORATION = 0.28


def main() -> None:
    _configure_light_search()

    data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))

    compare_rows: list[dict[str, float | int | str]] = []
    default_rows: list[pd.DataFrame] = []
    ga_rows: list[pd.DataFrame] = []
    pso_rows: list[pd.DataFrame] = []
    aco_rows: list[pd.DataFrame] = []

    for drone_count in LIGHT_K_VALUES:
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
            solution_type="f_default_light",
        )
        default_rows.append(default_summary)

        ga_runs = [
            ga_module.run_ga_with_base_state(drone_count, LIGHT_SEED, data, base_node_state, horizon, energy_limit, hover_power)[0]
        ]
        pso_runs = [
            pso_module.run_pso_with_base_state(drone_count, LIGHT_SEED, data, base_node_state, horizon, energy_limit, hover_power)[0]
        ]
        aco_runs = [
            aco_module.run_aco_with_base_state(drone_count, LIGHT_SEED, data, base_node_state, horizon, energy_limit, hover_power)[0]
        ]

        best_ga = _best_of_runs(ga_runs)
        best_pso = _best_of_runs(pso_runs)
        best_aco = _best_of_runs(aco_runs)

        ga_rows.extend(ga_runs)
        pso_rows.extend(pso_runs)
        aco_rows.extend(aco_runs)

        compare_rows.append(
            {
                "drone_count": drone_count,
                "profile": "light",
                "seed": LIGHT_SEED,
                "baseline_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                "f_default_s": float(default_summary.loc[0, "total_closed_loop_time_s"]),
                "f_default_direct_confirm_count": int(default_summary.loc[0, "direct_confirm_count"]),
                "f_ga_s": float(best_ga.loc[best_ga.index[0], "total_closed_loop_time_s"]),
                "f_ga_direct_confirm_count": int(best_ga.loc[best_ga.index[0], "direct_confirm_count"]),
                "f_ga_runtime_s": float(best_ga.loc[best_ga.index[0], "runtime_s"]),
                "f_pso_s": float(best_pso.loc[best_pso.index[0], "total_closed_loop_time_s"]),
                "f_pso_direct_confirm_count": int(best_pso.loc[best_pso.index[0], "direct_confirm_count"]),
                "f_pso_runtime_s": float(best_pso.loc[best_pso.index[0], "runtime_s"]),
                "f_aco_s": float(best_aco.loc[best_aco.index[0], "total_closed_loop_time_s"]),
                "f_aco_direct_confirm_count": int(best_aco.loc[best_aco.index[0], "direct_confirm_count"]),
                "f_aco_runtime_s": float(best_aco.loc[best_aco.index[0], "runtime_s"]),
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
    pd.DataFrame(compare_rows).to_csv(output_dir / "c_problem_problem2_f_guidance_compare_light.csv", index=False, encoding="utf-8-sig")
    pd.concat(default_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_default_runs_light.csv", index=False, encoding="utf-8-sig")
    pd.concat(ga_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_guidance_ga_runs_light.csv", index=False, encoding="utf-8-sig")
    pd.concat(pso_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_guidance_pso_runs_light.csv", index=False, encoding="utf-8-sig")
    pd.concat(aco_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_guidance_aco_runs_light.csv", index=False, encoding="utf-8-sig")

    print("Problem 2 light F-guidance comparison saved to:", output_dir / "c_problem_problem2_f_guidance_compare_light.csv")
    print(pd.DataFrame(compare_rows).to_string(index=False))


if __name__ == "__main__":
    main()