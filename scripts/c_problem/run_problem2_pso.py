from __future__ import annotations

from math import exp
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


PARTICLE_COUNT = _env_int("C_PROBLEM_PSO_PARTICLES", 18)
ITERATIONS = _env_int("C_PROBLEM_PSO_ITERATIONS", 32)
INERTIA = _env_float("C_PROBLEM_PSO_INERTIA", 0.72)
COGNITIVE = _env_float("C_PROBLEM_PSO_COGNITIVE", 1.45)
SOCIAL = _env_float("C_PROBLEM_PSO_SOCIAL", 1.45)
VELOCITY_CLAMP = _env_float("C_PROBLEM_PSO_VELOCITY_CLAMP", 0.55)
SEEDS = _env_seeds("C_PROBLEM_PSO_SEEDS", [11, 23])


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _random_particle(rng: Random) -> tuple[list[float], list[float]]:
    position = clamp_weight_vector([value + rng.uniform(-0.75, 0.75) for value in DEFAULT_WEIGHT_VECTOR])
    velocity = [rng.uniform(-0.25, 0.25) for _ in DEFAULT_WEIGHT_VECTOR]
    return position, velocity


def _evaluate_position(
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
        solution_type="f_pso",
    )
    return float(summary.loc[0, "total_closed_loop_time_s"]), summary, detail


def run_pso_with_base_state(
    drone_count: int,
    seed: int,
    data,
    base_node_state: pd.DataFrame,
    horizon: float,
    energy_limit: float,
    hover_power: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = Random(seed)
    particles = []
    for _ in range(PARTICLE_COUNT):
        position, velocity = _random_particle(rng)
        particles.append({"position": position, "velocity": velocity})
    if particles:
        particles[0]["position"] = clamp_weight_vector(DEFAULT_WEIGHT_VECTOR)

    best_objective = float("inf")
    best_summary: pd.DataFrame | None = None
    best_detail: pd.DataFrame | None = None
    best_position = clamp_weight_vector(DEFAULT_WEIGHT_VECTOR)
    trace_rows: list[dict[str, float | int]] = []

    start_time = perf_counter()
    for particle in particles:
        objective, summary, detail = _evaluate_position(
            weight_vector=particle["position"],
            drone_count=drone_count,
            data=data,
            base_node_state=base_node_state,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
        )
        particle["best_position"] = particle["position"][:]
        particle["best_value"] = objective
        if objective < best_objective:
            best_objective = objective
            best_summary = summary.copy()
            best_detail = detail.copy()
            best_position = particle["position"][:]

    for iteration in range(ITERATIONS):
        for particle in particles:
            for idx in range(len(DEFAULT_WEIGHT_VECTOR)):
                cognitive_term = COGNITIVE * rng.random() * (particle["best_position"][idx] - particle["position"][idx])
                social_term = SOCIAL * rng.random() * (best_position[idx] - particle["position"][idx])
                velocity = INERTIA * particle["velocity"][idx] + cognitive_term + social_term
                velocity = max(-VELOCITY_CLAMP, min(VELOCITY_CLAMP, velocity))
                particle["velocity"][idx] = velocity
                particle["position"][idx] += (2.0 * _sigmoid(velocity) - 1.0) * 0.22
            particle["position"] = clamp_weight_vector(particle["position"])

            objective, summary, detail = _evaluate_position(
                weight_vector=particle["position"],
                drone_count=drone_count,
                data=data,
                base_node_state=base_node_state,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
            )
            if objective < particle["best_value"]:
                particle["best_value"] = objective
                particle["best_position"] = particle["position"][:]
            if objective < best_objective:
                best_objective = objective
                best_summary = summary.copy()
                best_detail = detail.copy()
                best_position = particle["position"][:]

        trace_rows.append(
            {
                "drone_count": drone_count,
                "seed": seed,
                "iteration": iteration,
                "best_objective_s": best_objective,
                **{f"weight_{key}": float(value) for key, value in zip(WEIGHT_KEYS, best_position, strict=False)},
            }
        )

    runtime_s = perf_counter() - start_time
    if best_summary is None or best_detail is None:
        raise RuntimeError(f"F-guided PSO failed for K={drone_count}, seed={seed}")

    best_summary = best_summary.copy()
    best_summary["solution_type"] = "f_pso"
    best_summary["seed"] = seed
    best_summary["runtime_s"] = runtime_s
    best_summary["weight_vector"] = ",".join(f"{value:.4f}" for value in best_position)
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

        pso_runs: list[pd.DataFrame] = []
        best_run: pd.DataFrame | None = None
        for seed in SEEDS:
            pso_summary, pso_detail, pso_trace = run_pso_with_base_state(
                drone_count=drone_count,
                seed=seed,
                data=data,
                base_node_state=base_node_state,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
            )
            pso_runs.append(pso_summary)
            detail_rows.append(pso_detail)
            trace_rows.append(pso_trace)
            if best_run is None or float(pso_summary.loc[0, "total_closed_loop_time_s"]) < float(best_run.loc[0, "total_closed_loop_time_s"]):
                best_run = pso_summary

        run_rows.append(pd.concat(pso_runs, ignore_index=True))
        if best_run is None:
            raise RuntimeError(f"No PSO run completed for K={drone_count}")

        compare_rows.append(
            pd.DataFrame(
                [
                    {
                        "drone_count": drone_count,
                        "baseline_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                        "f_default_s": float(default_summary.loc[0, "total_closed_loop_time_s"]),
                        "f_pso_s": float(best_run.loc[0, "total_closed_loop_time_s"]),
                        "f_pso_direct_confirm_count": int(best_run.loc[0, "direct_confirm_count"]),
                        "f_pso_runtime_s": float(best_run.loc[0, "runtime_s"]),
                    }
                ]
            )
        )

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(compare_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_pso_compare.csv", index=False, encoding="utf-8-sig")
    pd.concat(run_rows, ignore_index=True).to_csv(output_dir / "c_problem_problem2_f_pso_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat(detail_rows, ignore_index=True).to_csv(processed_dir / "c_problem_problem2_f_pso_detail.csv", index=False, encoding="utf-8-sig")
    pd.concat(trace_rows, ignore_index=True).to_csv(processed_dir / "c_problem_problem2_f_pso_trace.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()