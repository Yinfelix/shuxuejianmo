from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import DEFAULT_WEIGHT_VECTOR, build_guidance_base_state, enrich_node_state_with_route_context, evaluate_guidance_weights
from run_problem2_joint import _build_node_state


ENERGY_UTILIZATION_PENALTY_COEFF = 120.0
TIME_UTILIZATION_PENALTY_COEFF = 80.0
MAX_VNS_ITERATIONS = 8
MIN_SPLIT_ROUTE_LENGTH = 4
MAX_CANDIDATES_PER_NEIGHBORHOOD = 160


def _metric(matrix: pd.DataFrame, from_id: int, to_id: int) -> float:
    return float(matrix.loc[str(int(from_id)), str(int(to_id))])


def _build_route_plan(problem1_detail: pd.DataFrame) -> tuple[dict[int, list[list[int]]], dict[int, float]]:
    route_plan: dict[int, list[list[int]]] = {}
    hover_times: dict[int, float] = {}
    ordered = problem1_detail.sort_values(["drone_id", "route_order", "stop_order"])
    for row in ordered.itertuples(index=False):
        hover_times[int(row.node_id)] = float(row.hover_time_s)
    for drone_id, drone_group in ordered.groupby("drone_id"):
        route_plan[int(drone_id)] = [route_group.sort_values("stop_order")["node_id"].astype(int).tolist() for _, route_group in drone_group.groupby("route_id")]
    return route_plan, hover_times


def _normalize_route_plan(route_plan: dict[int, list[list[int]]]) -> dict[int, list[list[int]]]:
    normalized: dict[int, list[list[int]]] = {}
    for drone_id, routes in route_plan.items():
        normalized[int(drone_id)] = [list(route) for route in routes if route]
    return normalized


def _route_plan_signature(route_plan: dict[int, list[list[int]]]) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    normalized = _normalize_route_plan(route_plan)
    return tuple((int(drone_id), tuple(tuple(int(node_id) for node_id in route) for route in normalized[drone_id])) for drone_id in sorted(normalized))


def _append_candidate(
    candidates: list[tuple[str, dict[int, list[list[int]]]]],
    seen: set[tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]],
    move_tag: str,
    candidate_plan: dict[int, list[list[int]]],
) -> None:
    normalized_plan = _normalize_route_plan(candidate_plan)
    signature = _route_plan_signature(normalized_plan)
    if signature in seen:
        return
    seen.add(signature)
    candidates.append((move_tag, normalized_plan))
    if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
        return


def _rebuild_detail(
    drone_count: int,
    route_plan: dict[int, list[list[int]]],
    hover_times: dict[int, float],
    data,
    hover_power: float,
    battery_swap_time: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    drone_totals: dict[int, float] = {}

    for drone_id in sorted(route_plan):
        drone_total_time = 0.0
        for route_order, node_order in enumerate(route_plan[drone_id], start=1):
            flight_time = 0.0
            flight_energy = 0.0
            route_nodes = [0, *node_order, 0]
            for from_id, to_id in zip(route_nodes, route_nodes[1:], strict=False):
                flight_time += _metric(data.flight_time, from_id, to_id)
                flight_energy += _metric(data.flight_energy, from_id, to_id)

            hover_time = sum(hover_times[node_id] for node_id in node_order)
            route_duration = flight_time + hover_time
            route_energy = flight_energy + hover_time * hover_power
            if route_order > 1:
                drone_total_time += battery_swap_time
            drone_total_time += route_duration

            route_path = "0-" + "-".join(str(node_id) for node_id in node_order) + "-0"
            for stop_order, node_id in enumerate(node_order, start=1):
                rows.append(
                    {
                        "drone_count": drone_count,
                        "drone_id": drone_id,
                        "route_id": route_order,
                        "route_order": route_order,
                        "stop_order": stop_order,
                        "node_id": node_id,
                        "hover_time_s": hover_times[node_id],
                        "route_duration_s": route_duration,
                        "route_energy_j": route_energy,
                        "route_path": route_path,
                        "drone_total_time_s": 0.0,
                    }
                )

        drone_totals[drone_id] = drone_total_time

    detail = pd.DataFrame(rows)
    for drone_id, drone_total_time in drone_totals.items():
        detail.loc[detail["drone_id"] == drone_id, "drone_total_time_s"] = drone_total_time
    return detail.sort_values(["drone_id", "route_order", "stop_order"]).reset_index(drop=True)


def _evaluate_route_plan(
    drone_count: int,
    route_plan: dict[int, list[list[int]]],
    hover_times: dict[int, float],
    data,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    battery_swap_time: float,
    move_tag: str,
    solution_type: str = "taskcount_joint",
    ground_mode: str = "serial",
    prefer_resource_exhaustion: bool = False,
):
    problem1_detail = _rebuild_detail(drone_count, route_plan, hover_times, data, hover_power, battery_swap_time)
    expected_nodes = {int(node_id) for node_id in data.nodes.loc[data.nodes["node_id"] != 0, "node_id"].tolist()}
    actual_nodes = [int(node_id) for node_id in problem1_detail["node_id"].tolist()]
    if len(actual_nodes) != len(expected_nodes) or len(set(actual_nodes)) != len(expected_nodes) or set(actual_nodes) != expected_nodes:
        return None

    route_table = problem1_detail[["drone_id", "route_id", "route_duration_s", "route_energy_j"]].drop_duplicates().copy()
    route_table["energy_utilization_ratio"] = route_table["route_energy_j"] / energy_limit if energy_limit > 0 else 0.0
    route_table["unused_energy_ratio"] = 1.0 - route_table["energy_utilization_ratio"]
    is_feasible = bool(
        (route_table["route_duration_s"] <= horizon + 1e-9).all()
        and (route_table["route_energy_j"] <= energy_limit + 1e-9).all()
    )
    if not is_feasible:
        return None

    base_node_state = enrich_node_state_with_route_context(_build_node_state(problem1_detail, data.nodes), problem1_detail)
    if base_node_state[["drone_id", "route_id", "route_energy_j", "drone_total_time_s"]].isna().any().any():
        return None
    summary, detail, selected = evaluate_guidance_weights(
        drone_count=drone_count,
        data=data,
        base_node_state=base_node_state,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
        weight_vector=DEFAULT_WEIGHT_VECTOR,
        solution_type=solution_type,
        ground_mode=ground_mode,
        battery_swap_time_s=battery_swap_time,
        prefer_resource_exhaustion=prefer_resource_exhaustion,
    )

    energy_penalty = float((route_table["unused_energy_ratio"] ** 2).mean()) if not route_table.empty else 0.0
    drone_totals = problem1_detail[["drone_id", "drone_total_time_s"]].drop_duplicates().copy()
    drone_totals["unused_time_ratio"] = 1.0 - (drone_totals["drone_total_time_s"] / horizon if horizon > 0 else 0.0)
    time_penalty = float((drone_totals["unused_time_ratio"] ** 2).mean()) if not drone_totals.empty else 0.0
    augmented_objective = (
        float(summary.loc[0, "total_closed_loop_time_s"])
        + ENERGY_UTILIZATION_PENALTY_COEFF * energy_penalty
        + TIME_UTILIZATION_PENALTY_COEFF * time_penalty
    )

    return {
        "move_tag": move_tag,
        "route_plan": route_plan,
        "problem1_detail": problem1_detail,
        "summary": summary,
        "detail": detail,
        "selected": selected,
        "route_table": route_table,
        "route_count": int(route_table.shape[0]),
        "energy_penalty": energy_penalty,
        "time_penalty": time_penalty,
        "augmented_objective_s": augmented_objective,
        "avg_route_energy_utilization": float(route_table["energy_utilization_ratio"].mean()) if not route_table.empty else 0.0,
        "min_route_energy_utilization": float(route_table["energy_utilization_ratio"].min()) if not route_table.empty else 0.0,
    }


def _generate_split_candidates(route_plan: dict[int, list[list[int]]]) -> list[tuple[str, dict[int, list[list[int]]]]]:
    candidates: list[tuple[str, dict[int, list[list[int]]]]] = []
    seen: set[tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]] = set()
    drone_ids = sorted(route_plan)
    for drone_id in drone_ids:
        for route_index, node_order in enumerate(route_plan[drone_id]):
            if len(node_order) < MIN_SPLIT_ROUTE_LENGTH:
                continue
            for cut_position in range(2, len(node_order) - 1):
                head_nodes = node_order[:cut_position]
                tail_nodes = node_order[cut_position:]

                same_drone_plan = deepcopy(route_plan)
                same_drone_plan[drone_id] = (
                    same_drone_plan[drone_id][:route_index]
                    + [head_nodes, tail_nodes]
                    + same_drone_plan[drone_id][route_index + 1 :]
                )
                _append_candidate(candidates, seen, f"split_same_d{drone_id}_r{route_index + 1}_c{cut_position}", same_drone_plan)
                if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
                    return candidates

                for other_drone_id in drone_ids:
                    if other_drone_id == drone_id:
                        continue
                    transfer_plan = deepcopy(route_plan)
                    transfer_plan[drone_id] = (
                        transfer_plan[drone_id][:route_index]
                        + [head_nodes]
                        + transfer_plan[drone_id][route_index + 1 :]
                    )
                    transfer_plan[other_drone_id] = transfer_plan[other_drone_id] + [tail_nodes]
                    _append_candidate(
                        candidates,
                        seen,
                        f"split_transfer_d{drone_id}_to_d{other_drone_id}_r{route_index + 1}_c{cut_position}",
                        transfer_plan,
                    )
                    if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
                        return candidates
    return candidates


def _generate_merge_candidates(route_plan: dict[int, list[list[int]]]) -> list[tuple[str, dict[int, list[list[int]]]]]:
    candidates: list[tuple[str, dict[int, list[list[int]]]]] = []
    seen: set[tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]] = set()
    for drone_id in sorted(route_plan):
        if len(route_plan[drone_id]) < 2:
            continue
        for route_index in range(len(route_plan[drone_id]) - 1):
            merged_plan = deepcopy(route_plan)
            merged_route = merged_plan[drone_id][route_index] + merged_plan[drone_id][route_index + 1]
            merged_plan[drone_id] = (
                merged_plan[drone_id][:route_index]
                + [merged_route]
                + merged_plan[drone_id][route_index + 2 :]
            )
            _append_candidate(candidates, seen, f"merge_d{drone_id}_r{route_index + 1}_{route_index + 2}", merged_plan)
            if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
                return candidates
    return candidates


def _generate_intra_relocate_candidates(route_plan: dict[int, list[list[int]]]) -> list[tuple[str, dict[int, list[list[int]]]]]:
    candidates: list[tuple[str, dict[int, list[list[int]]]]] = []
    seen: set[tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]] = set()
    for drone_id in sorted(route_plan):
        for route_index, node_order in enumerate(route_plan[drone_id]):
            if len(node_order) < 3:
                continue
            for source_index in range(len(node_order)):
                for target_index in range(len(node_order)):
                    if source_index == target_index:
                        continue
                    relocated_nodes = node_order[:]
                    node_id = relocated_nodes.pop(source_index)
                    relocated_nodes.insert(target_index, node_id)
                    candidate_plan = deepcopy(route_plan)
                    candidate_plan[drone_id][route_index] = relocated_nodes
                    _append_candidate(
                        candidates,
                        seen,
                        f"intra_relocate_d{drone_id}_r{route_index + 1}_{source_index + 1}_{target_index + 1}",
                        candidate_plan,
                    )
                    if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
                        return candidates
    return candidates


def _generate_two_opt_candidates(route_plan: dict[int, list[list[int]]]) -> list[tuple[str, dict[int, list[list[int]]]]]:
    candidates: list[tuple[str, dict[int, list[list[int]]]]] = []
    seen: set[tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]] = set()
    for drone_id in sorted(route_plan):
        for route_index, node_order in enumerate(route_plan[drone_id]):
            if len(node_order) < 4:
                continue
            for left in range(len(node_order) - 2):
                for right in range(left + 2, len(node_order) + 1):
                    candidate_nodes = node_order[:left] + list(reversed(node_order[left:right])) + node_order[right:]
                    candidate_plan = deepcopy(route_plan)
                    candidate_plan[drone_id][route_index] = candidate_nodes
                    _append_candidate(candidates, seen, f"two_opt_d{drone_id}_r{route_index + 1}_{left + 1}_{right}", candidate_plan)
                    if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
                        return candidates
    return candidates


def _generate_cross_relocate_candidates(route_plan: dict[int, list[list[int]]]) -> list[tuple[str, dict[int, list[list[int]]]]]:
    candidates: list[tuple[str, dict[int, list[list[int]]]]] = []
    seen: set[tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]] = set()
    drone_ids = sorted(route_plan)
    for source_drone_id in drone_ids:
        for source_route_index, node_order in enumerate(route_plan[source_drone_id]):
            if len(node_order) <= 1:
                continue
            for node_index, node_id in enumerate(node_order):
                source_removed_plan = deepcopy(route_plan)
                source_removed_plan[source_drone_id][source_route_index] = (
                    source_removed_plan[source_drone_id][source_route_index][:node_index]
                    + source_removed_plan[source_drone_id][source_route_index][node_index + 1 :]
                )
                for target_drone_id in drone_ids:
                    for target_route_index, target_route in enumerate(source_removed_plan[target_drone_id]):
                        if source_drone_id == target_drone_id and source_route_index == target_route_index:
                            continue
                        for insert_index in range(len(target_route) + 1):
                            candidate_plan = deepcopy(source_removed_plan)
                            candidate_plan[target_drone_id][target_route_index] = (
                                candidate_plan[target_drone_id][target_route_index][:insert_index]
                                + [node_id]
                                + candidate_plan[target_drone_id][target_route_index][insert_index:]
                            )
                            _append_candidate(
                                candidates,
                                seen,
                                f"cross_relocate_d{source_drone_id}_r{source_route_index + 1}_n{node_id}_to_d{target_drone_id}_r{target_route_index + 1}_p{insert_index + 1}",
                                candidate_plan,
                            )
                            if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
                                return candidates
                    candidate_plan = deepcopy(source_removed_plan)
                    candidate_plan[target_drone_id] = candidate_plan[target_drone_id] + [[node_id]]
                    _append_candidate(
                        candidates,
                        seen,
                        f"cross_relocate_newroute_d{source_drone_id}_r{source_route_index + 1}_n{node_id}_to_d{target_drone_id}",
                        candidate_plan,
                    )
                    if len(candidates) >= MAX_CANDIDATES_PER_NEIGHBORHOOD:
                        return candidates
    return candidates


def _neighborhood_generators() -> list[tuple[str, callable]]:
    return [
        ("intra_relocate", _generate_intra_relocate_candidates),
        ("two_opt", _generate_two_opt_candidates),
        ("cross_relocate", _generate_cross_relocate_candidates),
        ("split", _generate_split_candidates),
        ("merge", _generate_merge_candidates),
    ]


def _select_best_candidate(
    drone_count: int,
    candidate_specs: list[tuple[str, dict[int, list[list[int]]]]],
    hover_times: dict[int, float],
    data,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    battery_swap_time: float,
    solution_type: str = "taskcount_joint",
):
    best_candidate = None
    for move_tag, candidate_plan in candidate_specs:
        candidate = _evaluate_route_plan(
            drone_count=drone_count,
            route_plan=candidate_plan,
            hover_times=hover_times,
            data=data,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            battery_swap_time=battery_swap_time,
            move_tag=move_tag,
            solution_type=solution_type,
        )
        if candidate is None:
            continue
        if best_candidate is None or (
            candidate["augmented_objective_s"],
            candidate["summary"].loc[0, "total_closed_loop_time_s"],
            candidate["route_count"],
        ) < (
            best_candidate["augmented_objective_s"],
            best_candidate["summary"].loc[0, "total_closed_loop_time_s"],
            best_candidate["route_count"],
        ):
            best_candidate = candidate
    return best_candidate


def _variable_neighborhood_search(
    drone_count: int,
    initial_plan: dict[int, list[list[int]]],
    hover_times: dict[int, float],
    data,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    battery_swap_time: float,
):
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
        solution_type="taskcount_joint",
    )
    if current is None:
        raise RuntimeError(f"Initial route plan for drone_count={drone_count} is infeasible")
    history = [current]
    neighborhoods = _neighborhood_generators()

    for _ in range(MAX_VNS_ITERATIONS):
        neighborhood_index = 0
        improved = False
        while neighborhood_index < len(neighborhoods):
            neighborhood_name, generator = neighborhoods[neighborhood_index]
            candidate_specs = generator(current["route_plan"])
            if not candidate_specs:
                neighborhood_index += 1
                continue

            best_candidate = _select_best_candidate(
                drone_count=drone_count,
                candidate_specs=candidate_specs,
                hover_times=hover_times,
                data=data,
                horizon=horizon,
                energy_limit=energy_limit,
                hover_power=hover_power,
                battery_swap_time=battery_swap_time,
                solution_type="taskcount_joint",
            )
            if best_candidate is None:
                neighborhood_index += 1
                continue
            if best_candidate["augmented_objective_s"] + 1e-9 < current["augmented_objective_s"]:
                best_candidate["move_tag"] = f"vns_{neighborhood_name}:{best_candidate['move_tag']}"
                current = best_candidate
                history.append(current)
                neighborhood_index = 0
                improved = True
                continue
            neighborhood_index += 1

        if not improved:
            break

    return history


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    battery_swap_time = float(parameter_value(data.params, "battery_swap_time_s"))

    compare_rows: list[dict[str, float | int | str]] = []
    detail_tables: list[pd.DataFrame] = []
    meta_rows: list[dict[str, float | int | str]] = []

    for drone_count in range(1, 5):
        _, problem1_detail, _ = build_guidance_base_state(drone_count, data)
        route_plan, hover_times = _build_route_plan(problem1_detail)
        history = _variable_neighborhood_search(
            drone_count=drone_count,
            initial_plan=route_plan,
            hover_times=hover_times,
            data=data,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            battery_swap_time=battery_swap_time,
        )

        original = history[0]
        best = history[-1]
        best_detail = best["detail"].copy()
        best_detail["move_tag"] = best["move_tag"]
        best_detail["route_count"] = best["route_count"]
        detail_tables.append(best_detail)

        compare_rows.append(
            {
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
                "accepted_move_count": len(history) - 1,
                "search_method": "vns",
            }
        )

        for iteration_index, record in enumerate(history):
            meta_rows.append(
                {
                    "drone_count": drone_count,
                    "iteration_index": iteration_index,
                    "move_tag": record["move_tag"],
                    "route_count": record["route_count"],
                    "total_closed_loop_time_s": float(record["summary"].loc[0, "total_closed_loop_time_s"]),
                    "air_completion_time_s": float(record["summary"].loc[0, "air_completion_time_s"]),
                    "ground_completion_time_s": float(record["summary"].loc[0, "ground_completion_time_s"]),
                    "direct_confirm_count": int(record["summary"].loc[0, "direct_confirm_count"]),
                    "avg_route_energy_utilization": record["avg_route_energy_utilization"],
                    "min_route_energy_utilization": record["min_route_energy_utilization"],
                    "energy_penalty": record["energy_penalty"],
                    "augmented_objective_s": record["augmented_objective_s"],
                    "selected_direct_confirm_nodes": str(record["summary"].loc[0, "selected_direct_confirm_nodes"]),
                    "manual_route": str(record["summary"].loc[0, "manual_route"]),
                    "search_method": "vns",
                }
            )

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    compare_path = output_dir / "c_problem_problem2_taskcount_joint_compare.csv"
    detail_path = processed_dir / "c_problem_problem2_taskcount_joint_detail.csv"
    meta_path = processed_dir / "c_problem_problem2_taskcount_joint_meta.csv"

    pd.DataFrame(compare_rows).to_csv(compare_path, index=False, encoding="utf-8-sig")
    pd.concat(detail_tables, ignore_index=True).to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("Problem 2 task-count joint comparison saved to:", compare_path)
    print(pd.DataFrame(compare_rows).to_string(index=False))
    print("\nTask-count joint detail saved to:", detail_path)
    print("Task-count joint metadata saved to:", meta_path)


if __name__ == "__main__":
    main()