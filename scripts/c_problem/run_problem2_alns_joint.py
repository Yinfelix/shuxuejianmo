from __future__ import annotations

import math
import os
from pathlib import Path
from random import Random

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import build_guidance_base_state
from run_problem2_taskcount_joint import (
    _build_route_plan,
    _evaluate_route_plan,
    _normalize_route_plan,
)


ALNS_SEED = int(os.environ.get("ALNS_SEED", "11"))
ALNS_SEEDS = [int(value) for value in os.environ.get("ALNS_SEEDS", str(ALNS_SEED)).split(",") if value.strip()]
ALNS_ITERATIONS = int(os.environ.get("ALNS_ITERATIONS", "18"))
ALNS_COOLING = float(os.environ.get("ALNS_COOLING", "0.90"))
ALNS_MIN_TEMPERATURE = float(os.environ.get("ALNS_MIN_TEMPERATURE", "1.0"))
ALNS_DRONE_COUNTS = [int(value) for value in os.environ.get("ALNS_DRONE_COUNTS", "1,2,3,4").split(",") if value.strip()]
ALNS_OUTPUT_TAG = os.environ.get("ALNS_OUTPUT_TAG", "").strip()
ALNS_STAGE_SPLITS = (0.34, 0.67)
ALNS_STAGE_WEIGHT_PRESETS = {
    "explore": {
        "random_remove": 1.35,
        "priority_remove": 0.9,
        "route_remove": 1.15,
        "tail_remove": 0.8,
    },
    "balance": {
        "random_remove": 1.0,
        "priority_remove": 1.05,
        "route_remove": 1.0,
        "tail_remove": 1.1,
    },
    "intensify": {
        "random_remove": 0.75,
        "priority_remove": 1.2,
        "route_remove": 0.9,
        "tail_remove": 1.35,
    },
}


def _all_nodes(route_plan: dict[int, list[list[int]]]) -> list[int]:
    nodes: list[int] = []
    for routes in route_plan.values():
        for route in routes:
            nodes.extend(int(node_id) for node_id in route)
    return nodes


def _node_home_drone_map(route_plan: dict[int, list[list[int]]]) -> dict[int, int]:
    home_map: dict[int, int] = {}
    for drone_id, routes in route_plan.items():
        for route in routes:
            for node_id in route:
                home_map[int(node_id)] = int(drone_id)
    return home_map


def _remove_nodes(route_plan: dict[int, list[list[int]]], removed_nodes: list[int]) -> dict[int, list[list[int]]]:
    removed_set = set(int(node_id) for node_id in removed_nodes)
    updated = {
        int(drone_id): [[int(node_id) for node_id in route if int(node_id) not in removed_set] for route in routes]
        for drone_id, routes in route_plan.items()
    }
    return _normalize_route_plan(updated)


def _destroy_random_nodes(route_plan: dict[int, list[list[int]]], rng: Random, priority_map: dict[int, float]) -> tuple[str, dict[int, list[list[int]]], list[int]]:
    del priority_map
    nodes = _all_nodes(route_plan)
    if len(nodes) <= 2:
        return "random_remove", route_plan, []
    removal_count = min(3, max(2, len(nodes) // 6))
    removed_nodes = rng.sample(nodes, removal_count)
    return "random_remove", _remove_nodes(route_plan, removed_nodes), removed_nodes


def _destroy_priority_nodes(route_plan: dict[int, list[list[int]]], rng: Random, priority_map: dict[int, float]) -> tuple[str, dict[int, list[list[int]]], list[int]]:
    del rng
    nodes = sorted(_all_nodes(route_plan), key=lambda node_id: (priority_map.get(int(node_id), 0.0), int(node_id)), reverse=True)
    removed_nodes = nodes[: min(3, len(nodes))]
    return "priority_remove", _remove_nodes(route_plan, removed_nodes), removed_nodes


def _destroy_route(route_plan: dict[int, list[list[int]]], rng: Random, priority_map: dict[int, float]) -> tuple[str, dict[int, list[list[int]]], list[int]]:
    del priority_map
    populated_routes: list[tuple[int, int, list[int]]] = []
    for drone_id, routes in route_plan.items():
        for route_index, route in enumerate(routes):
            if route:
                populated_routes.append((int(drone_id), route_index, route))
    if len(populated_routes) <= 1:
        return "route_remove", route_plan, []
    drone_id, route_index, route = rng.choice(populated_routes)
    updated = {int(current_drone_id): [list(nodes) for nodes in routes] for current_drone_id, routes in route_plan.items()}
    updated[drone_id] = updated[drone_id][:route_index] + updated[drone_id][route_index + 1 :]
    return "route_remove", _normalize_route_plan(updated), [int(node_id) for node_id in route]


def _destroy_low_utilization_tail(route_plan: dict[int, list[list[int]]], rng: Random, priority_map: dict[int, float]) -> tuple[str, dict[int, list[list[int]]], list[int]]:
    del rng, priority_map
    best_route: tuple[int, int, list[int]] | None = None
    for drone_id, routes in route_plan.items():
        for route_index, route in enumerate(routes):
            if len(route) >= 4 and (best_route is None or len(route) > len(best_route[2])):
                best_route = (int(drone_id), route_index, route)
    if best_route is None:
        return "tail_remove", route_plan, []
    drone_id, route_index, route = best_route
    removed_nodes = [int(node_id) for node_id in route[-2:]]
    updated = {int(current_drone_id): [list(nodes) for nodes in routes] for current_drone_id, routes in route_plan.items()}
    updated[drone_id][route_index] = updated[drone_id][route_index][:-2]
    return "tail_remove", _normalize_route_plan(updated), removed_nodes


def _repair_removed_nodes(
    base_plan: dict[int, list[list[int]]],
    removed_nodes: list[int],
    hover_times: dict[int, float],
    data,
    drone_count: int,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    battery_swap_time: float,
    priority_map: dict[int, float],
    home_drone_map: dict[int, int],
) -> dict[int, list[list[int]]] | None:
    repaired_plan = _normalize_route_plan(base_plan)
    insertion_order = sorted(removed_nodes, key=lambda node_id: (priority_map.get(int(node_id), 0.0), int(node_id)), reverse=True)
    for insertion_index, node_id in enumerate(insertion_order):
        best_plan = None
        best_objective = math.inf
        remaining_nodes = insertion_order[insertion_index + 1 :]
        for target_drone_id in sorted(repaired_plan):
            for route_index, route in enumerate(repaired_plan[target_drone_id]):
                for insert_index in range(len(route) + 1):
                    candidate_plan = {int(drone_id): [list(nodes) for nodes in routes] for drone_id, routes in repaired_plan.items()}
                    candidate_plan[target_drone_id][route_index] = (
                        candidate_plan[target_drone_id][route_index][:insert_index]
                        + [int(node_id)]
                        + candidate_plan[target_drone_id][route_index][insert_index:]
                    )
                    completed_plan = {int(drone_id): [list(nodes) for nodes in routes] for drone_id, routes in candidate_plan.items()}
                    for remaining_node in remaining_nodes:
                        completed_plan[int(home_drone_map[int(remaining_node)])] = completed_plan[int(home_drone_map[int(remaining_node)])] + [[int(remaining_node)]]
                    candidate = _evaluate_route_plan(
                        drone_count=drone_count,
                        route_plan=completed_plan,
                        hover_times=hover_times,
                        data=data,
                        horizon=horizon,
                        energy_limit=energy_limit,
                        hover_power=hover_power,
                        battery_swap_time=battery_swap_time,
                        move_tag=f"repair_insert_{node_id}",
                        solution_type="alns_joint",
                    )
                    if candidate is not None and candidate["augmented_objective_s"] < best_objective:
                        best_plan = candidate_plan
                        best_objective = candidate["augmented_objective_s"]

            candidate_plan = {int(drone_id): [list(nodes) for nodes in routes] for drone_id, routes in repaired_plan.items()}
            candidate_plan[target_drone_id] = candidate_plan[target_drone_id] + [[int(node_id)]]
            completed_plan = {int(drone_id): [list(nodes) for nodes in routes] for drone_id, routes in candidate_plan.items()}
            for remaining_node in remaining_nodes:
                completed_plan[int(home_drone_map[int(remaining_node)])] = completed_plan[int(home_drone_map[int(remaining_node)])] + [[int(remaining_node)]]
            candidate = _evaluate_route_plan(
                drone_count=drone_count,
                route_plan=completed_plan,
                hover_times=hover_times,
                data=data,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
                battery_swap_time=battery_swap_time,
                move_tag=f"repair_newroute_{node_id}",
                solution_type="alns_joint",
            )
            if candidate is not None and candidate["augmented_objective_s"] < best_objective:
                best_plan = candidate_plan
                best_objective = candidate["augmented_objective_s"]

        if best_plan is None:
            return None
        repaired_plan = _normalize_route_plan(best_plan)
    return repaired_plan


def _select_destroy_method(weights: dict[str, float], rng: Random) -> str:
    total_weight = sum(weights.values())
    threshold = rng.random() * total_weight
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if cumulative >= threshold:
            return name
    return next(iter(weights))


def _alns_stage_name(iteration_index: int, total_iterations: int) -> str:
    if total_iterations <= 1:
        return "intensify"
    progress_ratio = (iteration_index + 1) / total_iterations
    if progress_ratio <= ALNS_STAGE_SPLITS[0]:
        return "explore"
    if progress_ratio <= ALNS_STAGE_SPLITS[1]:
        return "balance"
    return "intensify"


def _initialize_stage_weights(method_names: list[str]) -> dict[str, dict[str, float]]:
    stage_weights: dict[str, dict[str, float]] = {}
    for stage_name, preset in ALNS_STAGE_WEIGHT_PRESETS.items():
        stage_weights[stage_name] = {method_name: float(preset.get(method_name, 1.0)) for method_name in method_names}
    return stage_weights


def _run_alns(
    drone_count: int,
    initial_plan: dict[int, list[list[int]]],
    hover_times: dict[int, float],
    data,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    battery_swap_time: float,
    seed: int,
):
    rng = Random(seed + drone_count)
    priority_map = {int(row.node_id): float(row.priority_weight) for row in data.nodes.loc[data.nodes["node_id"] != 0, ["node_id", "priority_weight"]].itertuples(index=False)}
    destroy_methods = {
        "random_remove": _destroy_random_nodes,
        "priority_remove": _destroy_priority_nodes,
        "route_remove": _destroy_route,
        "tail_remove": _destroy_low_utilization_tail,
    }
    stage_weights = _initialize_stage_weights(list(destroy_methods))

    current = _evaluate_route_plan(
        drone_count=drone_count,
        route_plan=initial_plan,
        hover_times=hover_times,
        data=data,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
        battery_swap_time=battery_swap_time,
        move_tag="original",
        solution_type="alns_joint",
    )
    if current is None:
        raise RuntimeError(f"Initial route plan for drone_count={drone_count} is infeasible")
    best = current
    accepted_history = [current]
    meta_rows: list[dict[str, float | int | str]] = []
    temperature = max(ALNS_MIN_TEMPERATURE, current["augmented_objective_s"] * 0.02)

    for iteration_index in range(ALNS_ITERATIONS):
        stage_name = _alns_stage_name(iteration_index, ALNS_ITERATIONS)
        destroy_weights = stage_weights[stage_name]
        method_name = _select_destroy_method(destroy_weights, rng)
        home_drone_map = _node_home_drone_map(current["route_plan"])
        destroy_name, partial_plan, removed_nodes = destroy_methods[method_name](current["route_plan"], rng, priority_map)
        if not removed_nodes:
            continue
        repaired_plan = _repair_removed_nodes(
            base_plan=partial_plan,
            removed_nodes=removed_nodes,
            hover_times=hover_times,
            data=data,
            drone_count=drone_count,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            battery_swap_time=battery_swap_time,
            priority_map=priority_map,
            home_drone_map=home_drone_map,
        )
        if repaired_plan is None:
            destroy_weights[method_name] = max(0.2, destroy_weights[method_name] * 0.95)
            continue

        candidate = _evaluate_route_plan(
            drone_count=drone_count,
            route_plan=repaired_plan,
            hover_times=hover_times,
            data=data,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            battery_swap_time=battery_swap_time,
            move_tag=f"alns_{destroy_name}",
            solution_type="alns_joint",
        )
        if candidate is None:
            destroy_weights[method_name] = max(0.2, destroy_weights[method_name] * 0.95)
            continue

        delta = candidate["augmented_objective_s"] - current["augmented_objective_s"]
        accepted = delta < 0 or rng.random() < math.exp(-delta / max(temperature, ALNS_MIN_TEMPERATURE))
        if accepted:
            current = candidate
            accepted_history.append(current)
            destroy_weights[method_name] += 0.35
            if current["augmented_objective_s"] + 1e-9 < best["augmented_objective_s"]:
                best = current
                destroy_weights[method_name] += 1.25
        else:
            destroy_weights[method_name] = max(0.2, destroy_weights[method_name] * 0.97)

        meta_rows.append(
            {
                "seed": seed,
                "drone_count": drone_count,
                "iteration_index": iteration_index,
                "stage_name": stage_name,
                "destroy_method": destroy_name,
                "removed_nodes": ",".join(str(node_id) for node_id in removed_nodes),
                "candidate_augmented_objective_s": candidate["augmented_objective_s"],
                "current_augmented_objective_s": current["augmented_objective_s"],
                "best_augmented_objective_s": best["augmented_objective_s"],
                "accepted": accepted,
                "temperature": temperature,
                "stage_method_weight": destroy_weights[method_name],
                "route_count": current["route_count"],
                "selected_direct_confirm_nodes": str(current["summary"].loc[0, "selected_direct_confirm_nodes"]),
            }
        )
        temperature = max(ALNS_MIN_TEMPERATURE, temperature * ALNS_COOLING)

    return best, accepted_history, pd.DataFrame(meta_rows)


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    battery_swap_time = float(parameter_value(data.params, "battery_swap_time_s"))

    compare_rows: list[dict[str, float | int | str]] = []
    detail_tables: list[pd.DataFrame] = []
    meta_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for drone_count in ALNS_DRONE_COUNTS:
        _, problem1_detail, _ = build_guidance_base_state(drone_count, data)
        route_plan, hover_times = _build_route_plan(problem1_detail)
        original = _evaluate_route_plan(
            drone_count=drone_count,
            route_plan=route_plan,
            hover_times=hover_times,
            data=data,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            battery_swap_time=battery_swap_time,
            move_tag="original",
            solution_type="alns_joint",
        )
        if original is None:
            raise RuntimeError(f"Initial route plan for drone_count={drone_count} is infeasible")
        drone_run_rows: list[dict[str, float | int | str]] = []
        best_run_row: dict[str, float | int | str] | None = None

        for seed in ALNS_SEEDS:
            best, accepted_history, meta_df = _run_alns(
                drone_count=drone_count,
                initial_plan=route_plan,
                hover_times=hover_times,
                data=data,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
                battery_swap_time=battery_swap_time,
                seed=seed,
            )
            best_detail = best["detail"].copy()
            best_detail["seed"] = seed
            best_detail["move_tag"] = best["move_tag"]
            best_detail["route_count"] = best["route_count"]
            detail_tables.append(best_detail)
            meta_tables.append(meta_df)

            run_row = {
                "seed": seed,
                "drone_count": drone_count,
                "original_route_count": original["route_count"],
                "optimized_route_count": best["route_count"],
                "original_total_closed_loop_s": float(original["summary"].loc[0, "total_closed_loop_time_s"]),
                "optimized_total_closed_loop_s": float(best["summary"].loc[0, "total_closed_loop_time_s"]),
                "gain_vs_original_s": float(original["summary"].loc[0, "total_closed_loop_time_s"]) - float(best["summary"].loc[0, "total_closed_loop_time_s"]),
                "original_augmented_objective_s": original["augmented_objective_s"],
                "optimized_augmented_objective_s": best["augmented_objective_s"],
                "optimized_air_completion_time_s": float(best["summary"].loc[0, "air_completion_time_s"]),
                "optimized_ground_completion_time_s": float(best["summary"].loc[0, "ground_completion_time_s"]),
                "optimized_direct_confirm_count": int(best["summary"].loc[0, "direct_confirm_count"]),
                "optimized_avg_route_energy_utilization": best["avg_route_energy_utilization"],
                "optimized_min_route_energy_utilization": best["min_route_energy_utilization"],
                "energy_penalty": best["energy_penalty"],
                "final_move_tag": best["move_tag"],
                "accepted_move_count": len(accepted_history) - 1,
                "search_method": "alns",
            }
            compare_rows.append(run_row)
            drone_run_rows.append(run_row)
            if best_run_row is None or (
                float(run_row["optimized_augmented_objective_s"]),
                float(run_row["optimized_total_closed_loop_s"]),
                int(run_row["optimized_route_count"]),
            ) < (
                float(best_run_row["optimized_augmented_objective_s"]),
                float(best_run_row["optimized_total_closed_loop_s"]),
                int(best_run_row["optimized_route_count"]),
            ):
                best_run_row = run_row

        drone_runs = pd.DataFrame(drone_run_rows)
        improved_ratio = float((drone_runs["gain_vs_original_s"] > 1e-9).mean()) if not drone_runs.empty else 0.0
        summary_rows.append(
            {
                "drone_count": drone_count,
                "seed_count": len(drone_run_rows),
                "original_total_closed_loop_s": float(original["summary"].loc[0, "total_closed_loop_time_s"]),
                "best_optimized_total_closed_loop_s": float(best_run_row["optimized_total_closed_loop_s"]) if best_run_row is not None else float(original["summary"].loc[0, "total_closed_loop_time_s"]),
                "mean_optimized_total_closed_loop_s": float(drone_runs["optimized_total_closed_loop_s"].mean()) if not drone_runs.empty else float(original["summary"].loc[0, "total_closed_loop_time_s"]),
                "std_optimized_total_closed_loop_s": float(drone_runs["optimized_total_closed_loop_s"].std(ddof=0)) if not drone_runs.empty else 0.0,
                "best_gain_vs_original_s": float(best_run_row["gain_vs_original_s"]) if best_run_row is not None else 0.0,
                "mean_gain_vs_original_s": float(drone_runs["gain_vs_original_s"].mean()) if not drone_runs.empty else 0.0,
                "std_gain_vs_original_s": float(drone_runs["gain_vs_original_s"].std(ddof=0)) if not drone_runs.empty else 0.0,
                "improved_seed_ratio": improved_ratio,
                "best_seed": int(best_run_row["seed"]) if best_run_row is not None else -1,
                "best_move_tag": best_run_row["final_move_tag"] if best_run_row is not None else "original",
                "best_optimized_route_count": int(best_run_row["optimized_route_count"]) if best_run_row is not None else int(original["route_count"]),
                "best_direct_confirm_count": int(best_run_row["optimized_direct_confirm_count"]) if best_run_row is not None else int(original["summary"].loc[0, "direct_confirm_count"]),
                "search_method": "alns",
            }
        )

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{ALNS_OUTPUT_TAG}" if ALNS_OUTPUT_TAG else ""
    compare_path = output_dir / f"c_problem_problem2_alns_joint_compare{suffix}.csv"
    summary_path = output_dir / f"c_problem_problem2_alns_joint_summary{suffix}.csv"
    detail_path = processed_dir / f"c_problem_problem2_alns_joint_detail{suffix}.csv"
    meta_path = processed_dir / f"c_problem_problem2_alns_joint_meta{suffix}.csv"

    pd.DataFrame(compare_rows).to_csv(compare_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.concat(detail_tables, ignore_index=True).to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.concat(meta_tables, ignore_index=True).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("Problem 2 ALNS joint comparison saved to:", compare_path)
    print(pd.DataFrame(compare_rows).to_string(index=False))
    print("\nProblem 2 ALNS joint summary saved to:", summary_path)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nALNS joint detail saved to:", detail_path)
    print("ALNS joint metadata saved to:", meta_path)


if __name__ == "__main__":
    main()