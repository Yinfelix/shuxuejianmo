from __future__ import annotations

import os
from pathlib import Path
from random import Random
from time import perf_counter

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import (
    DEFAULT_WEIGHT_VECTOR,
    WEIGHT_KEYS,
    build_guidance_base_state,
    clamp_weight_vector,
    evaluate_guidance_weights,
)
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


POPULATION_SIZE = _env_int("C_PROBLEM_GA_POPULATION", 18)
GENERATIONS = _env_int("C_PROBLEM_GA_GENERATIONS", 28)
ELITE_COUNT = _env_int("C_PROBLEM_GA_ELITE", 4)
TOURNAMENT_SIZE = 3
MUTATION_RATE = _env_float("C_PROBLEM_GA_MUTATION_RATE", 0.28)
MUTATION_STEP = _env_float("C_PROBLEM_GA_MUTATION_STEP", 0.35)
SEEDS = _env_seeds("C_PROBLEM_GA_SEEDS", [11, 23])


def _random_vector(rng: Random) -> list[float]:
    return clamp_weight_vector([value + rng.uniform(-0.65, 0.65) for value in DEFAULT_WEIGHT_VECTOR])


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
        solution_type="f_ga",
    )
    return float(summary.loc[0, "total_closed_loop_time_s"]), summary, detail


def _tournament_select(population: list[list[float]], fitness: list[float], rng: Random) -> list[float]:
    sampled_indices = [rng.randrange(len(population)) for _ in range(TOURNAMENT_SIZE)]
    best_index = min(sampled_indices, key=lambda idx: fitness[idx])
    return population[best_index][:]


def _crossover(parent_a: list[float], parent_b: list[float], rng: Random) -> tuple[list[float], list[float]]:
    alpha = rng.random()
    child_a = [alpha * value_a + (1.0 - alpha) * value_b for value_a, value_b in zip(parent_a, parent_b, strict=False)]
    child_b = [(1.0 - alpha) * value_a + alpha * value_b for value_a, value_b in zip(parent_a, parent_b, strict=False)]
    return clamp_weight_vector(child_a), clamp_weight_vector(child_b)


def _mutate(weight_vector: list[float], rng: Random) -> list[float]:
    mutated = weight_vector[:]
    for idx in range(len(mutated)):
        if rng.random() < MUTATION_RATE:
            mutated[idx] += rng.uniform(-MUTATION_STEP, MUTATION_STEP)
    return clamp_weight_vector(mutated)


def run_ga_with_base_state(
    drone_count: int,
    seed: int,
    data,
    base_node_state: pd.DataFrame,
    horizon: float,
    energy_limit: float,
    hover_power: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = Random(seed)
    population = [_random_vector(rng) for _ in range(POPULATION_SIZE)]
    if population:
        population[0] = clamp_weight_vector(DEFAULT_WEIGHT_VECTOR)

    best_objective = float("inf")
    best_summary: pd.DataFrame | None = None
    best_detail: pd.DataFrame | None = None
    best_vector = clamp_weight_vector(DEFAULT_WEIGHT_VECTOR)
    trace_rows: list[dict[str, float | int]] = []

    start_time = perf_counter()
    for generation in range(GENERATIONS):
        evaluated: list[tuple[float, pd.DataFrame, pd.DataFrame, list[float]]] = []
        for vector in population:
            objective, summary, detail = _evaluate_vector(
                weight_vector=vector,
                drone_count=drone_count,
                data=data,
                base_node_state=base_node_state,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
            )
            evaluated.append((objective, summary, detail, vector[:]))
            if objective < best_objective:
                best_objective = objective
                best_summary = summary.copy()
                best_detail = detail.copy()
                best_vector = vector[:]

        evaluated.sort(key=lambda item: item[0])
        trace_rows.append(
            {
                "drone_count": drone_count,
                "seed": seed,
                "generation": generation,
                "best_objective_s": float(evaluated[0][0]),
                **{f"weight_{key}": float(value) for key, value in zip(WEIGHT_KEYS, evaluated[0][3], strict=False)},
            }
        )

        ranked_population = [item[3][:] for item in evaluated]
        fitness = [item[0] for item in evaluated]
        next_population = ranked_population[:ELITE_COUNT]
        while len(next_population) < POPULATION_SIZE:
            parent_a = _tournament_select(ranked_population, fitness, rng)
            parent_b = _tournament_select(ranked_population, fitness, rng)
            child_a, child_b = _crossover(parent_a, parent_b, rng)
            next_population.append(_mutate(child_a, rng))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(_mutate(child_b, rng))
        population = next_population[:POPULATION_SIZE]

    runtime_s = perf_counter() - start_time
    if best_summary is None or best_detail is None:
        raise RuntimeError(f"F-guided GA failed for K={drone_count}, seed={seed}")

    best_summary = best_summary.copy()
    best_summary["solution_type"] = "f_ga"
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

        ga_runs: list[pd.DataFrame] = []
        best_run: pd.DataFrame | None = None
        for seed in SEEDS:
            ga_summary, ga_detail, ga_trace = run_ga_with_base_state(
                drone_count=drone_count,
                seed=seed,
                data=data,
                base_node_state=base_node_state,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
            )
            ga_runs.append(ga_summary)
            detail_rows.append(ga_detail)
            trace_rows.append(ga_trace)
            if best_run is None or float(ga_summary.loc[0, "total_closed_loop_time_s"]) < float(best_run.loc[0, "total_closed_loop_time_s"]):
                best_run = ga_summary

        run_rows.append(pd.concat(ga_runs, ignore_index=True))
        if best_run is None:
            raise RuntimeError(f"No GA run completed for K={drone_count}")

        compare_rows.append(
            pd.DataFrame(
                [
                    {
                        "drone_count": drone_count,
                        "baseline_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                        "f_default_s": float(default_summary.loc[0, "total_closed_loop_time_s"]),
                        "f_ga_s": float(best_run.loc[0, "total_closed_loop_time_s"]),
                        "f_ga_direct_confirm_count": int(best_run.loc[0, "direct_confirm_count"]),
                        "f_ga_runtime_s": float(best_run.loc[0, "runtime_s"]),
                    }
                ]
            )
        )

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(compare_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_ga_compare.csv", index=False, encoding="utf-8-sig")
    pd.concat(run_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_ga_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat(detail_rows, ignore_index=True).to_csv(processed_dir / "c_problem_problem2_f_ga_detail.csv", index=False, encoding="utf-8-sig")
    pd.concat(trace_rows, ignore_index=True).to_csv(processed_dir / "c_problem_problem2_f_ga_trace.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()