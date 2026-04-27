from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from problem2_f_guidance import (
    DEFAULT_WEIGHT_VECTOR,
    build_guidance_base_state,
    clamp_weight_vector,
    enrich_node_state_with_route_context,
    vector_to_weight_dict,
    evaluate_guidance_weights,
)
from run_problem2_joint import _build_node_state, _summarize_node_state


DEPOT_ID = 0
DEPOT_MANUAL_ID = "P0"


def _metric(matrix: pd.DataFrame, from_id: int, to_id: int) -> float:
    return float(matrix.loc[str(from_id), str(to_id)])


def _finite_metric(matrix: pd.DataFrame, from_id: int, to_id: int) -> float | None:
    value = pd.to_numeric(pd.Series([matrix.loc[str(from_id), str(to_id)]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _recompute_route(
    drone_count: int,
    drone_id: int,
    route_id: int,
    route_order: int,
    node_order: list[int],
    hover_times: dict[int, float],
    data,
    hover_power: float,
) -> tuple[list[dict[str, float | int | str]], float, float]:
    current = DEPOT_ID
    flight_time_total = 0.0
    flight_energy_total = 0.0
    hover_time_total = 0.0
    for node_id in node_order:
        leg_time = _finite_metric(data.flight_time, current, node_id)
        leg_energy = _finite_metric(data.flight_energy, current, node_id)
        if leg_time is None or leg_energy is None:
            raise ValueError(f"Non-finite flight edge on reordered route: {current} -> {node_id}")
        flight_time_total += leg_time
        flight_energy_total += leg_energy
        hover_time_total += float(hover_times[node_id])
        current = node_id
    back_time = _finite_metric(data.flight_time, current, DEPOT_ID)
    back_energy = _finite_metric(data.flight_energy, current, DEPOT_ID)
    if back_time is None or back_energy is None:
        raise ValueError(f"Non-finite return edge on reordered route: {current} -> 0")
    flight_time_total += back_time
    flight_energy_total += back_energy

    route_duration = flight_time_total + hover_time_total
    route_energy = flight_energy_total + hover_time_total * hover_power
    route_path = "0-" + "-".join(str(node_id) for node_id in node_order) + "-0"

    route_rows: list[dict[str, float | int | str]] = []
    for stop_order, node_id in enumerate(node_order, start=1):
        route_rows.append(
            {
                "drone_count": drone_count,
                "drone_id": drone_id,
                "route_id": route_id,
                "route_order": route_order,
                "stop_order": stop_order,
                "node_id": node_id,
                "hover_time_s": float(hover_times[node_id]),
                "route_duration_s": route_duration,
                "route_energy_j": route_energy,
                "route_path": route_path,
                "drone_total_time_s": 0.0,
            }
        )
    return route_rows, route_duration, route_energy


def _route_score_order(
    route_detail: pd.DataFrame,
    data,
    hover_power: float,
    energy_limit: float,
    horizon: float,
    fixed_other_routes_time: float,
    battery_swap_time: float,
    weight_vector: list[float],
) -> list[int]:
    weights = vector_to_weight_dict(clamp_weight_vector(weight_vector))
    remaining_nodes = route_detail.sort_values("stop_order")["node_id"].astype(int).tolist()
    node_rows = route_detail.set_index("node_id")
    max_leg_time = 1.0
    static_ground_gain: dict[int, float] = {}
    max_ground_gain = 1.0
    max_priority = 1.0
    max_hover_gap = 1.0

    for node_id in remaining_nodes:
        node_info = node_rows.loc[node_id]
        manual_point_id = str(node_info.get("manual_point_id", ""))
        service_time = float(node_info.get("manual_service_time_s", 0.0) or 0.0)
        ground_proxy = service_time
        if manual_point_id and manual_point_id in data.ground_time.index:
            ground_proxy += 0.5 * (
                float(data.ground_time.loc[DEPOT_MANUAL_ID, manual_point_id])
                + float(data.ground_time.loc[manual_point_id, DEPOT_MANUAL_ID])
            )
        static_ground_gain[node_id] = ground_proxy
        max_ground_gain = max(max_ground_gain, ground_proxy)
        max_priority = max(max_priority, float(node_info["priority_weight"]))
        hover_gap = max(0.0, float(node_info["direct_confirm_time_s"] - node_info["hover_time_s"]))
        max_hover_gap = max(max_hover_gap, hover_gap)
        max_leg_time = max(max_leg_time, _metric(data.flight_time, DEPOT_ID, node_id))

    ordered_nodes: list[int] = []
    current_node = DEPOT_ID
    partial_route_time = 0.0
    partial_route_energy = 0.0

    while remaining_nodes:
        scored_candidates: list[tuple[float, int]] = []
        for node_id in remaining_nodes:
            node_info = node_rows.loc[node_id]
            hover_time = float(node_info["hover_time_s"])
            leg_time = _finite_metric(data.flight_time, current_node, node_id)
            leg_energy = _finite_metric(data.flight_energy, current_node, node_id)
            return_time = _finite_metric(data.flight_time, node_id, DEPOT_ID)
            return_energy = _finite_metric(data.flight_energy, node_id, DEPOT_ID)
            if leg_time is None or leg_energy is None or return_time is None or return_energy is None:
                continue
            projected_route_time = partial_route_time + leg_time + hover_time + return_time
            projected_drone_time = fixed_other_routes_time + battery_swap_time + projected_route_time
            residual_energy_ratio = max(0.0, energy_limit - (partial_route_energy + leg_energy + hover_time * hover_power + return_energy)) / energy_limit
            residual_time_ratio = max(0.0, horizon - projected_drone_time) / horizon
            hover_gap = max(0.0, float(node_info["direct_confirm_time_s"] - hover_time))
            travel_efficiency = max(0.0, 1.0 - leg_time / max_leg_time)
            score = (
                weights["ground_gain"] * (static_ground_gain[node_id] / max_ground_gain)
                + weights["priority"] * (float(node_info["priority_weight"]) / max_priority)
                + weights["energy_slack"] * residual_energy_ratio
                + weights["time_slack"] * residual_time_ratio
                + weights["route_progress"] * travel_efficiency
                - weights["hover_gap_penalty"] * (hover_gap / max_hover_gap)
            )
            scored_candidates.append((score, node_id))

        if not scored_candidates:
            return route_detail.sort_values("stop_order")["node_id"].astype(int).tolist()

        scored_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        chosen_node = scored_candidates[0][1]
        chosen_hover = float(node_rows.loc[chosen_node, "hover_time_s"])
        partial_route_time += float(_finite_metric(data.flight_time, current_node, chosen_node)) + chosen_hover
        partial_route_energy += float(_finite_metric(data.flight_energy, current_node, chosen_node)) + chosen_hover * hover_power
        ordered_nodes.append(chosen_node)
        remaining_nodes.remove(chosen_node)
        current_node = chosen_node

    return ordered_nodes


def _reorder_problem1_detail(
    problem1_detail: pd.DataFrame,
    data,
    drone_count: int,
    hover_power: float,
    energy_limit: float,
    horizon: float,
    battery_swap_time: float,
    weight_vector: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = _build_node_state(problem1_detail, data.nodes)
    enriched = enriched.merge(
        data.manual_points[["mapped_target_id", "manual_point_id", "manual_service_time_s"]].rename(columns={"mapped_target_id": "node_id"}),
        on="node_id",
        how="left",
    )

    route_key_to_order = {
        (int(row.drone_id), int(row.route_id)): int(row.route_order)
        for row in problem1_detail[["drone_id", "route_id", "route_order"]].drop_duplicates().itertuples(index=False)
    }
    original_by_drone = {
        int(drone_id): group.sort_values(["route_order", "stop_order"]).copy()
        for drone_id, group in problem1_detail.groupby("drone_id")
    }

    reordered_rows: list[dict[str, float | int | str]] = []
    meta_rows: list[dict[str, float | int]] = []

    for drone_id, drone_detail in original_by_drone.items():
        route_groups = list(drone_detail.groupby("route_id"))
        candidate_rows: list[dict[str, float | int | str]] = []
        route_change_count = 0
        route_durations: list[float] = []
        route_energies: list[float] = []
        fallback_used = False

        original_route_durations = {
            int(route_id): float(route_group["route_duration_s"].iloc[0])
            for route_id, route_group in route_groups
        }

        for route_id, route_group in sorted(route_groups, key=lambda item: route_key_to_order[(drone_id, int(item[0]))]):
            route_order = route_key_to_order[(drone_id, int(route_id))]
            fixed_other_routes_time = sum(duration for rid, duration in original_route_durations.items() if rid != int(route_id))
            route_nodes = route_group.sort_values("stop_order")["node_id"].astype(int).tolist()
            route_node_state = route_group[["node_id", "stop_order", "hover_time_s"]].copy()
            route_node_state = route_node_state.merge(
                enriched.drop(columns=["allocated_hover_time_s", "extra_hover_added_s", "direct_confirmed", "confirmed_by_extra_hover"], errors="ignore"),
                on="node_id",
                how="left",
            )
            proposed_order = _route_score_order(
                route_detail=route_node_state,
                data=data,
                hover_power=hover_power,
                energy_limit=energy_limit,
                horizon=horizon,
                fixed_other_routes_time=fixed_other_routes_time,
                battery_swap_time=battery_swap_time * max(route_order - 1, 0),
                weight_vector=weight_vector,
            )
            if proposed_order != route_nodes:
                route_change_count += 1
            hover_times = {
                int(row.node_id): float(row.hover_time_s)
                for row in route_group[["node_id", "hover_time_s"]].itertuples(index=False)
            }
            try:
                route_rows, route_duration, route_energy = _recompute_route(
                    drone_count=drone_count,
                    drone_id=drone_id,
                    route_id=int(route_id),
                    route_order=route_order,
                    node_order=proposed_order,
                    hover_times=hover_times,
                    data=data,
                    hover_power=hover_power,
                )
            except ValueError:
                route_rows = route_group.to_dict("records")
                route_duration = float(route_group["route_duration_s"].iloc[0])
                route_energy = float(route_group["route_energy_j"].iloc[0])
            candidate_rows.extend(route_rows)
            route_durations.append(route_duration)
            route_energies.append(route_energy)

        drone_total_time = sum(route_durations) + battery_swap_time * max(len(route_durations) - 1, 0)
        if any(route_energy > energy_limit for route_energy in route_energies) or drone_total_time > horizon:
            fallback_used = True
            candidate_rows = original_by_drone[drone_id].to_dict("records")
            drone_total_time = float(original_by_drone[drone_id]["drone_total_time_s"].iloc[0])

        for row in candidate_rows:
            row["drone_total_time_s"] = drone_total_time
            reordered_rows.append(row)

        meta_rows.append(
            {
                "drone_count": drone_count,
                "drone_id": drone_id,
                "route_change_count": route_change_count,
                "fallback_used": int(fallback_used),
                "drone_total_time_s": drone_total_time,
            }
        )

    reordered_detail = pd.DataFrame(reordered_rows).sort_values(["drone_id", "route_order", "stop_order"]).reset_index(drop=True)
    meta_table = pd.DataFrame(meta_rows)
    return reordered_detail, meta_table


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    battery_swap_time = float(parameter_value(data.params, "battery_swap_time_s"))

    compare_rows: list[dict[str, float | int | str]] = []
    detail_tables: list[pd.DataFrame] = []
    meta_tables: list[pd.DataFrame] = []

    for drone_count in range(1, 5):
        _, problem1_detail, base_node_state = build_guidance_base_state(drone_count, data)
        baseline_summary, _ = _summarize_node_state(
            node_state=base_node_state,
            manual_points=data.manual_points,
            ground_time=data.ground_time,
            drone_count=drone_count,
            solution_type="baseline",
            selected_nodes=[],
        )
        fixed_f_summary, _, fixed_selected = evaluate_guidance_weights(
            drone_count=drone_count,
            data=data,
            base_node_state=base_node_state,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            weight_vector=DEFAULT_WEIGHT_VECTOR,
            solution_type="f_default",
        )

        reordered_detail, reorder_meta = _reorder_problem1_detail(
            problem1_detail=problem1_detail,
            data=data,
            drone_count=drone_count,
            hover_power=hover_power,
            energy_limit=energy_limit,
            horizon=horizon,
            battery_swap_time=battery_swap_time,
            weight_vector=DEFAULT_WEIGHT_VECTOR,
        )
        reordered_state = _build_node_state(reordered_detail, data.nodes)
        reordered_state = enrich_node_state_with_route_context(reordered_state, reordered_detail)
        joint_summary, joint_detail, joint_selected = evaluate_guidance_weights(
            drone_count=drone_count,
            data=data,
            base_node_state=reordered_state,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            weight_vector=DEFAULT_WEIGHT_VECTOR,
            solution_type="joint_reorder_f",
        )

        joint_detail = joint_detail.copy()
        joint_detail["route_reordered"] = True
        detail_tables.append(joint_detail)
        meta_tables.append(reorder_meta)

        compare_rows.append(
            {
                "drone_count": drone_count,
                "baseline_s": float(baseline_summary.loc[0, "total_closed_loop_time_s"]),
                "f_default_s": float(fixed_f_summary.loc[0, "total_closed_loop_time_s"]),
                "joint_reorder_f_s": float(joint_summary.loc[0, "total_closed_loop_time_s"]),
                "joint_gain_vs_default_s": float(fixed_f_summary.loc[0, "total_closed_loop_time_s"]) - float(joint_summary.loc[0, "total_closed_loop_time_s"]),
                "joint_direct_confirm_count": int(joint_summary.loc[0, "direct_confirm_count"]),
                "joint_air_completion_time_s": float(joint_summary.loc[0, "air_completion_time_s"]),
                "joint_selected_nodes": ",".join(str(record["node_id"]) for record in joint_selected),
                "route_change_count": int(reorder_meta["route_change_count"].sum()),
                "fallback_drone_count": int(reorder_meta["fallback_used"].sum()),
            }
        )

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    compare_path = output_dir / "c_problem_problem2_joint_reorder_f_compare.csv"
    detail_path = processed_dir / "c_problem_problem2_joint_reorder_f_detail.csv"
    meta_path = processed_dir / "c_problem_problem2_joint_reorder_f_meta.csv"
    pd.DataFrame(compare_rows).to_csv(compare_path, index=False, encoding="utf-8-sig")
    pd.concat(detail_tables, ignore_index=True).to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.concat(meta_tables, ignore_index=True).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("Problem 2 joint reorder F comparison saved to:", compare_path)
    print(pd.DataFrame(compare_rows).to_string(index=False))
    print("\nJoint reorder detail saved to:", detail_path)
    print("Joint reorder metadata saved to:", meta_path)


if __name__ == "__main__":
    main()