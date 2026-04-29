from __future__ import annotations

import os
from pathlib import Path
from random import Random
from time import perf_counter

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import DEFAULT_WEIGHT_VECTOR, WEIGHT_KEYS, build_guidance_base_state, clamp_weight_vector, evaluate_guidance_weights
from run_problem2_joint import _summarize_node_state


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_seeds(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


ANT_COUNT = _env_int("C_PROBLEM_ACO_ANTS", 20)
ITERATIONS = _env_int("C_PROBLEM_ACO_ITERATIONS", 26)
EVAPORATION = _env_float("C_PROBLEM_ACO_EVAPORATION", 0.24)
TOP_ANTS = _env_int("C_PROBLEM_ACO_TOP", 4)
WEIGHT_LEVELS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
SEEDS = _env_seeds("C_PROBLEM_ACO_SEEDS", [11, 23])


def _sample_level_indices(pheromone: list[list[float]], rng: Random) -> list[int]:
    level_indices: list[int] = []
    for row in pheromone:
        total = sum(row)
        threshold = rng.random() * total
        cumulative = 0.0
        chosen = 0
        for idx, value in enumerate(row):
            cumulative += value
            if cumulative >= threshold:
                chosen = idx
                break
        level_indices.append(chosen)
    return level_indices


def _indices_to_vector(level_indices: list[int]) -> list[float]:
    return clamp_weight_vector([WEIGHT_LEVELS[index] for index in level_indices])


def _evaluate_vector(
    weight_vector: list[float],
    drone_count: int,
    data,
    base_node_state: pd.DataFrame,
    horizon: float,
    energy_limit: float,
    hover_power: float,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    summary, detail, _ = evaluate_guidance_weights(
        drone_count=drone_count,
        data=data,
        base_node_state=base_node_state,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
        weight_vector=weight_vector,
        solution_type="f_aco",
    )
    return float(summary.loc[0, "total_closed_loop_time_s"]), summary, detail


def run_aco_with_base_state(
    drone_count: int,
    seed: int,
    data,
    base_node_state: pd.DataFrame,
    horizon: float,
    energy_limit: float,
    hover_power: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = Random(seed)
    pheromone = [[1.0 for _ in WEIGHT_LEVELS] for _ in WEIGHT_KEYS]
    default_indices = [min(range(len(WEIGHT_LEVELS)), key=lambda idx: abs(WEIGHT_LEVELS[idx] - value)) for value in DEFAULT_WEIGHT_VECTOR]

    best_objective = float("inf")
    best_summary: pd.DataFrame | None = None
    best_detail: pd.DataFrame | None = None
    best_indices = default_indices[:]
    trace_rows: list[dict[str, float | int]] = []

    start_time = perf_counter()
    for iteration in range(ITERATIONS):
        ant_results: list[tuple[float, pd.DataFrame, pd.DataFrame, list[int]]] = []
        for ant_id in range(ANT_COUNT):
            indices = default_indices[:] if ant_id == 0 and iteration == 0 else _sample_level_indices(pheromone, rng)
            objective, summary, detail = _evaluate_vector(
                weight_vector=_indices_to_vector(indices),
                drone_count=drone_count,
                data=data,
                base_node_state=base_node_state,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
            )
            ant_results.append((objective, summary, detail, indices[:]))
            if objective < best_objective:
                best_objective = objective
                best_summary = summary.copy()
                best_detail = detail.copy()
                best_indices = indices[:]

        ant_results.sort(key=lambda item: item[0])
        for row in pheromone:
            for idx in range(len(row)):
                row[idx] = max(0.05, row[idx] * (1.0 - EVAPORATION))

        for objective, _, _, indices in ant_results[:TOP_ANTS]:
            deposit = 5000.0 / max(objective, 1.0)
            for dimension, level_index in enumerate(indices):
                pheromone[dimension][level_index] += deposit

        trace_rows.append(
            {
                "drone_count": drone_count,
                "seed": seed,
                "iteration": iteration,
                "best_objective_s": best_objective,
                **{f"weight_{key}": WEIGHT_LEVELS[index] for key, index in zip(WEIGHT_KEYS, best_indices, strict=False)},
            }
        )

    runtime_s = perf_counter() - start_time
    if best_summary is None or best_detail is None:
        raise RuntimeError(f"F-guided ACO failed for K={drone_count}, seed={seed}")

    best_vector = _indices_to_vector(best_indices)
    best_summary = best_summary.copy()
    best_summary["solution_type"] = "f_aco"
    best_summary["seed"] = seed
    best_summary["runtime_s"] = runtime_s
    best_summary["weight_vector"] = ",".join(f"{value:.4f}" for value in best_vector)
    best_detail = best_detail.copy()
    best_detail["seed"] = seed
    return best_summary, best_detail, pd.DataFrame(trace_rows)


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    k_values = [3, 4]

    compare_rows: list[pd.DataFrame] = []
    run_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    trace_rows: list[pd.DataFrame] = []

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

        aco_runs: list[pd.DataFrame] = []
        best_run: pd.DataFrame | None = None
        for seed in SEEDS:
            aco_summary, aco_detail, aco_trace = run_aco_with_base_state(
                drone_count=drone_count,
                seed=seed,
                data=data,
                base_node_state=base_node_state,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
            )
            aco_runs.append(aco_summary)
            detail_rows.append(aco_detail)
            trace_rows.append(aco_trace)
            if best_run is None or float(aco_summary.loc[0, "total_closed_loop_time_s"]) < float(best_run.loc[0, "total_closed_loop_time_s"]):
                best_run = aco_summary

        run_rows.append(pd.concat(aco_runs, ignore_index=True))
        if best_run is None:
            raise RuntimeError(f"No ACO run completed for K={drone_count}")

        compare_rows.append(
            pd.DataFrame(
                [
                    {
                        "drone_count": drone_count,
                        "baseline_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                        "f_default_s": float(default_summary.loc[0, "total_closed_loop_time_s"]),
                        "f_aco_s": float(best_run.loc[0, "total_closed_loop_time_s"]),
                        "f_aco_direct_confirm_count": int(best_run.loc[0, "direct_confirm_count"]),
                        "f_aco_runtime_s": float(best_run.loc[0, "runtime_s"]),
                    }
                ]
            )
        )

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(compare_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_aco_compare.csv", index=False, encoding="utf-8-sig")
    pd.concat(run_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_aco_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat(detail_rows, ignore_index=True).to_csv(processed_dir / "c_problem_problem2_f_aco_detail.csv", index=False, encoding="utf-8-sig")
    pd.concat(trace_rows, ignore_index=True).to_csv(processed_dir / "c_problem_problem2_f_aco_trace.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()