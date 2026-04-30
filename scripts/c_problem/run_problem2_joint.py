from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value


DEPOT_ID = 0
MANUAL_DEPOT_ID = "P0"
PROBLEM1_SUMMARY_PATH = Path("outputs/tables/c_problem_problem1_heuristic_summary.csv")
PROBLEM1_DETAIL_PATH = Path("data/processed/c_problem_problem1_route_detail.csv")


@dataclass
class RouteState:
    drone_id: int
    route_id: int
    route_order: int
    stops: list[int]
    hover_times_s: dict[int, float]
    duration_s: float
    energy_j: float
    mode: str = "base"


@dataclass
class DroneState:
    drone_id: int
    routes: list[RouteState] = field(default_factory=list)
    total_time_s: float = 0.0


@dataclass
class CandidateAction:
    node_id: int
    action_type: str
    drone_id: int
    route_id: int | None
    added_air_time_s: float
    added_energy_j: float
    route_duration_s: float
    route_energy_j: float
    new_closed_loop_s: float
    improvement_s: float


@dataclass
class SwapCandidate:
    left_drone_id: int
    left_route_id: int
    left_node_id: int
    right_drone_id: int
    right_route_id: int
    right_node_id: int
    left_route_stops: list[int]
    right_route_stops: list[int]
    left_hover_times_s: dict[int, float]
    right_hover_times_s: dict[int, float]
    left_route_duration_s: float
    left_route_energy_j: float
    right_route_duration_s: float
    right_route_energy_j: float
    new_closed_loop_s: float
    improvement_s: float


def _metric(matrix: pd.DataFrame, from_id: int | str, to_id: int | str) -> float:
    return float(matrix.loc[str(from_id), str(to_id)])


def _load_problem1_solution() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PROBLEM1_SUMMARY_PATH.exists() or not PROBLEM1_DETAIL_PATH.exists():
        raise FileNotFoundError(
            "Problem 1 outputs are missing. Run scripts/c_problem/run_problem1_heuristic.py first."
        )
    return pd.read_csv(PROBLEM1_SUMMARY_PATH), pd.read_csv(PROBLEM1_DETAIL_PATH)


def _build_target_table(data, threshold_multiplier: float = 1.0) -> pd.DataFrame:
    merged = data.nodes.loc[data.nodes["node_id"] != 0].copy()
    for column in ["base_hover_time_s", "direct_confirm_time_s", "priority_weight", "manual_service_time_s"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["manual_point_id"] = merged["manual_point_id"].astype(str)
    merged["direct_confirm_time_s"] = merged["direct_confirm_time_s"] * threshold_multiplier
    merged["threshold_multiplier"] = threshold_multiplier
    merged["hover_gap_s"] = merged["direct_confirm_time_s"] - merged["base_hover_time_s"]
    return merged


def _build_drone_states(detail_df: pd.DataFrame, drone_count: int) -> dict[int, DroneState]:
    states: dict[int, DroneState] = {}
    slice_df = detail_df.loc[detail_df["drone_count"] == drone_count].copy()
    for (drone_id, route_id), route_group in slice_df.groupby(["drone_id", "route_id"], sort=True):
        route_group = route_group.sort_values("stop_order")
        route = RouteState(
            drone_id=int(drone_id),
            route_id=int(route_id),
            route_order=int(route_group["route_order"].iloc[0]),
            stops=[int(value) for value in route_group["node_id"].tolist()],
            hover_times_s={
                int(node_id): float(hover_time)
                for node_id, hover_time in zip(route_group["node_id"], route_group["hover_time_s"], strict=False)
            },
            duration_s=float(route_group["route_duration_s"].iloc[0]),
            energy_j=float(route_group["route_energy_j"].iloc[0]),
            mode="base",
        )
        state = states.setdefault(int(drone_id), DroneState(drone_id=int(drone_id)))
        state.routes.append(route)
        state.total_time_s = float(route_group["drone_total_time_s"].iloc[0])
    for state in states.values():
        state.routes.sort(key=lambda item: item.route_order)
    return states


def _build_node_state(problem1_detail: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    route_context = problem1_detail.copy()
    route_context["node_id"] = route_context["node_id"].astype(int)
    route_context["hover_time_s"] = pd.to_numeric(route_context["hover_time_s"], errors="coerce")
    merged = nodes.loc[nodes["node_id"] != 0].copy().merge(
        route_context[
            [
                "node_id",
                "drone_id",
                "route_id",
                "hover_time_s",
                "route_duration_s",
                "route_energy_j",
                "drone_total_time_s",
            ]
        ],
        on="node_id",
        how="left",
    )
    merged["allocated_hover_time_s"] = pd.to_numeric(merged["hover_time_s"], errors="coerce").fillna(0.0)
    merged["extra_hover_added_s"] = 0.0
    merged["direct_confirmed"] = merged["allocated_hover_time_s"] >= pd.to_numeric(merged["direct_confirm_time_s"], errors="coerce").fillna(0.0)
    merged["confirmed_by_extra_hover"] = False
    return merged


def _route_for_node(drone_states: dict[int, DroneState], node_id: int) -> tuple[DroneState, RouteState] | None:
    for drone_state in drone_states.values():
        for route in drone_state.routes:
            if node_id in route.stops:
                return drone_state, route
    return None


def _ground_travel_time(path: list[str], ground_time: pd.DataFrame) -> float:
    return sum(_metric(ground_time, from_id, to_id) for from_id, to_id in zip(path, path[1:], strict=False))


def _ground_path(manual_point_ids: list[str], ground_time: pd.DataFrame) -> list[str]:
    if not manual_point_ids:
        return [MANUAL_DEPOT_ID, MANUAL_DEPOT_ID]

    remaining = set(manual_point_ids)
    current = MANUAL_DEPOT_ID
    path = [MANUAL_DEPOT_ID]
    while remaining:
        next_point = min(remaining, key=lambda point_id: _metric(ground_time, current, point_id))
        path.append(next_point)
        remaining.remove(next_point)
        current = next_point
    path.append(MANUAL_DEPOT_ID)

    improved = True
    best = path
    while improved:
        improved = False
        best_cost = _ground_travel_time(best, ground_time)
        for left in range(1, len(best) - 2):
            for right in range(left + 1, len(best) - 1):
                candidate = best[:left] + best[left : right + 1][::-1] + best[right + 1 :]
                candidate_cost = _ground_travel_time(candidate, ground_time)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    return best


def _ground_stage_metrics(manual_points: pd.DataFrame, manual_point_ids: list[str], ground_time: pd.DataFrame) -> tuple[float, float, float, list[str]]:
    if not manual_point_ids:
        return 0.0, 0.0, 0.0, [MANUAL_DEPOT_ID, MANUAL_DEPOT_ID]
    path = _ground_path(manual_point_ids, ground_time)
    travel_time_s = _ground_travel_time(path, ground_time)
    service_time_s = float(
        manual_points.loc[manual_points["manual_point_id"].isin(manual_point_ids), "manual_service_time_s"].sum()
    )
    return travel_time_s + service_time_s, travel_time_s, service_time_s, path


def optimize_ground_review_path(ground_time: pd.DataFrame, manual_point_ids: list[str], method: str = "two_opt") -> tuple[list[str], float]:
    del method
    path = _ground_path([str(point_id) for point_id in manual_point_ids], ground_time)
    return path, _ground_travel_time(path, ground_time)


def _route_travel_savings(ground_time: pd.DataFrame, manual_route: list[str]) -> dict[str, float]:
    savings: dict[str, float] = {}
    if len(manual_route) <= 2:
        return savings
    for left, current, right in zip(manual_route, manual_route[1:], manual_route[2:], strict=False):
        if current == MANUAL_DEPOT_ID:
            continue
        savings[str(current)] = _metric(ground_time, left, current) + _metric(ground_time, current, right) - _metric(ground_time, left, right)
    return savings


def _build_manual_review_table(node_state: pd.DataFrame, manual_points: pd.DataFrame) -> pd.DataFrame:
    del manual_points
    pending = node_state.loc[~node_state["direct_confirmed"]].copy()
    return pending[["node_id", "node_name", "manual_point_id", "manual_service_time_s"]].reset_index(drop=True)


def _summarize_node_state(
    node_state: pd.DataFrame,
    manual_points: pd.DataFrame,
    ground_time: pd.DataFrame,
    drone_count: int,
    solution_type: str,
    selected_nodes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = node_state.copy().reset_index(drop=True)
    detail["solution_type"] = solution_type
    detail["selected_by_guidance"] = detail["node_id"].astype(str).isin(set(str(node_id) for node_id in selected_nodes))

    manual_point_ids = detail.loc[~detail["direct_confirmed"], "manual_point_id"].dropna().astype(str).tolist()
    ground_stage_time_s, ground_travel_time_s, ground_service_time_s, ground_path = _ground_stage_metrics(
        manual_points,
        manual_point_ids,
        ground_time,
    )
    air_completion_time_s = float(detail[["drone_id", "drone_total_time_s"]].drop_duplicates()["drone_total_time_s"].max()) if not detail.empty else 0.0
    route_energy_table = detail[["drone_id", "route_id", "route_energy_j"]].drop_duplicates()
    total_air_energy_j = float(route_energy_table["route_energy_j"].sum()) if not route_energy_table.empty else 0.0
    total_hover_time_s = float(detail["allocated_hover_time_s"].sum()) if not detail.empty else 0.0
    base_hover_sum = float(pd.to_numeric(detail["base_hover_time_s"], errors="coerce").fillna(0.0).sum()) if "base_hover_time_s" in detail.columns else total_hover_time_s
    direct_confirm_count = int(detail["direct_confirmed"].sum()) if not detail.empty else 0
    summary = pd.DataFrame(
        [
            {
                "drone_count": drone_count,
                "solution_type": solution_type,
                "air_completion_time_s": air_completion_time_s,
                "ground_completion_time_s": ground_stage_time_s,
                "ground_travel_time_s": ground_travel_time_s,
                "ground_service_time_s": ground_service_time_s,
                "total_closed_loop_time_s": air_completion_time_s + ground_stage_time_s,
                "direct_confirm_count": direct_confirm_count,
                "direct_confirm_ratio": direct_confirm_count / len(detail) if len(detail) else 0.0,
                "manual_review_count": int((~detail["direct_confirmed"]).sum()) if not detail.empty else 0,
                "extra_hover_time_s": total_hover_time_s - base_hover_sum,
                "total_hover_time_s": total_hover_time_s,
                "total_air_energy_j": total_air_energy_j,
                "confirmed_priority_weight": float(detail.loc[detail["direct_confirmed"], "priority_weight"].sum()) if not detail.empty else 0.0,
                "selected_direct_confirm_nodes": ",".join(str(node_id) for node_id in sorted(detail.loc[detail["direct_confirmed"], "node_id"].astype(int).tolist())),
                "manual_route": "-".join(ground_path),
            }
        ]
    )
    return summary, detail


def _route_metrics(
    stops: list[int],
    hover_times_s: dict[int, float],
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
    hover_power: float,
) -> tuple[float, float] | None:
    if not stops:
        return 0.0, 0.0

    route_nodes = [DEPOT_ID, *stops, DEPOT_ID]
    total_flight_time_s = 0.0
    total_flight_energy_j = 0.0
    for from_id, to_id in zip(route_nodes, route_nodes[1:], strict=False):
        total_flight_time_s += _metric(flight_time, from_id, to_id)
        total_flight_energy_j += _metric(flight_energy, from_id, to_id)

    total_hover_time_s = sum(float(hover_times_s[node_id]) for node_id in stops)
    total_duration_s = total_flight_time_s + total_hover_time_s
    total_energy_j = total_flight_energy_j + total_hover_time_s * hover_power
    return total_duration_s, total_energy_j


def _swap_route_state(route: RouteState, old_node_id: int, new_node_id: int, new_hover_time_s: float) -> tuple[list[int], dict[int, float]]:
    swapped_stops = route.stops.copy()
    swapped_stops[swapped_stops.index(old_node_id)] = new_node_id
    swapped_hover_times_s = route.hover_times_s.copy()
    swapped_hover_times_s.pop(old_node_id)
    swapped_hover_times_s[new_node_id] = new_hover_time_s
    return swapped_stops, swapped_hover_times_s


def _evaluate_swap_action(
    left_drone: DroneState,
    left_route: RouteState,
    left_node_id: int,
    right_drone: DroneState,
    right_route: RouteState,
    right_node_id: int,
    hover_power: float,
    energy_limit: float,
    horizon: float,
    ground_stage_time_s: float,
    current_closed_loop_s: float,
    drone_states: dict[int, DroneState],
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
) -> SwapCandidate | None:
    left_stops, left_hover_times_s = _swap_route_state(
        left_route,
        left_node_id,
        right_node_id,
        right_route.hover_times_s[right_node_id],
    )
    right_stops, right_hover_times_s = _swap_route_state(
        right_route,
        right_node_id,
        left_node_id,
        left_route.hover_times_s[left_node_id],
    )
    left_metrics = _route_metrics(left_stops, left_hover_times_s, flight_time, flight_energy, hover_power)
    right_metrics = _route_metrics(right_stops, right_hover_times_s, flight_time, flight_energy, hover_power)
    if left_metrics is None or right_metrics is None:
        return None

    left_route_duration_s, left_route_energy_j = left_metrics
    right_route_duration_s, right_route_energy_j = right_metrics
    if left_route_energy_j > energy_limit or right_route_energy_j > energy_limit:
        return None

    drone_totals = {drone_id: state.total_time_s for drone_id, state in drone_states.items()}
    drone_totals[left_drone.drone_id] += left_route_duration_s - left_route.duration_s
    drone_totals[right_drone.drone_id] += right_route_duration_s - right_route.duration_s

    if drone_totals[left_drone.drone_id] > horizon or drone_totals[right_drone.drone_id] > horizon:
        return None

    new_air_makespan_s = max(drone_totals.values())
    new_closed_loop_s = new_air_makespan_s + ground_stage_time_s
    improvement_s = current_closed_loop_s - new_closed_loop_s
    if improvement_s <= 1e-9:
        return None

    return SwapCandidate(
        left_drone_id=left_drone.drone_id,
        left_route_id=left_route.route_id,
        left_node_id=left_node_id,
        right_drone_id=right_drone.drone_id,
        right_route_id=right_route.route_id,
        right_node_id=right_node_id,
        left_route_stops=left_stops,
        right_route_stops=right_stops,
        left_hover_times_s=left_hover_times_s,
        right_hover_times_s=right_hover_times_s,
        left_route_duration_s=left_route_duration_s,
        left_route_energy_j=left_route_energy_j,
        right_route_duration_s=right_route_duration_s,
        right_route_energy_j=right_route_energy_j,
        new_closed_loop_s=new_closed_loop_s,
        improvement_s=improvement_s,
    )


def _apply_swap_action(swap: SwapCandidate, drone_states: dict[int, DroneState]) -> None:
    left_drone = drone_states[swap.left_drone_id]
    right_drone = drone_states[swap.right_drone_id]
    left_route = next(route for route in left_drone.routes if route.route_id == swap.left_route_id)
    right_route = next(route for route in right_drone.routes if route.route_id == swap.right_route_id)

    left_drone.total_time_s += swap.left_route_duration_s - left_route.duration_s
    right_drone.total_time_s += swap.right_route_duration_s - right_route.duration_s

    left_route.stops = swap.left_route_stops
    left_route.hover_times_s = swap.left_hover_times_s
    left_route.duration_s = swap.left_route_duration_s
    left_route.energy_j = swap.left_route_energy_j

    right_route.stops = swap.right_route_stops
    right_route.hover_times_s = swap.right_hover_times_s
    right_route.duration_s = swap.right_route_duration_s
    right_route.energy_j = swap.right_route_energy_j


def _swap_local_search(
    drone_states: dict[int, DroneState],
    hover_power: float,
    energy_limit: float,
    horizon: float,
    ground_stage_time_s: float,
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
) -> tuple[int, float]:
    accepted_swap_count = 0
    total_improvement_s = 0.0

    while True:
        current_air_makespan_s = max(state.total_time_s for state in drone_states.values())
        current_closed_loop_s = current_air_makespan_s + ground_stage_time_s
        route_pairs = [(drone_state, route) for drone_state in drone_states.values() for route in drone_state.routes]
        best_swap: SwapCandidate | None = None

        for left_index, (left_drone, left_route) in enumerate(route_pairs):
            for right_drone, right_route in route_pairs[left_index + 1 :]:
                for left_node_id in left_route.stops:
                    for right_node_id in right_route.stops:
                        candidate = _evaluate_swap_action(
                            left_drone,
                            left_route,
                            left_node_id,
                            right_drone,
                            right_route,
                            right_node_id,
                            hover_power,
                            energy_limit,
                            horizon,
                            ground_stage_time_s,
                            current_closed_loop_s,
                            drone_states,
                            flight_time,
                            flight_energy,
                        )
                        if candidate is not None and (
                            best_swap is None
                            or (candidate.improvement_s, -candidate.new_closed_loop_s)
                            > (best_swap.improvement_s, -best_swap.new_closed_loop_s)
                        ):
                            best_swap = candidate

        if best_swap is None:
            break

        _apply_swap_action(best_swap, drone_states)
        accepted_swap_count += 1
        total_improvement_s += best_swap.improvement_s

    return accepted_swap_count, total_improvement_s


def _evaluate_existing_route_action(
    node_id: int,
    hover_gap_s: float,
    drone_state: DroneState,
    route: RouteState,
    hover_power: float,
    energy_limit: float,
    horizon: float,
    new_ground_time_s: float,
    current_closed_loop_s: float,
    drone_states: dict[int, DroneState],
) -> CandidateAction | None:
    added_energy_j = hover_gap_s * hover_power
    route_energy_j = route.energy_j + added_energy_j
    if route_energy_j > energy_limit:
        return None

    projected_drone_time_s = drone_state.total_time_s + hover_gap_s
    if projected_drone_time_s > horizon:
        return None

    other_times = [state.total_time_s for state in drone_states.values() if state.drone_id != drone_state.drone_id]
    new_air_makespan_s = max([projected_drone_time_s, *other_times]) if other_times else projected_drone_time_s
    new_closed_loop_s = new_air_makespan_s + new_ground_time_s
    improvement_s = current_closed_loop_s - new_closed_loop_s
    if improvement_s <= 1e-9:
        return None

    return CandidateAction(
        node_id=node_id,
        action_type="extend_existing",
        drone_id=drone_state.drone_id,
        route_id=route.route_id,
        added_air_time_s=hover_gap_s,
        added_energy_j=added_energy_j,
        route_duration_s=route.duration_s + hover_gap_s,
        route_energy_j=route_energy_j,
        new_closed_loop_s=new_closed_loop_s,
        improvement_s=improvement_s,
    )


def _evaluate_new_route_action(
    node_id: int,
    hover_gap_s: float,
    drone_state: DroneState,
    hover_power: float,
    battery_swap_time_s: float,
    energy_limit: float,
    horizon: float,
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
    new_ground_time_s: float,
    current_closed_loop_s: float,
    drone_states: dict[int, DroneState],
) -> CandidateAction | None:
    flight_time_s = _metric(flight_time, DEPOT_ID, node_id) + _metric(flight_time, node_id, DEPOT_ID)
    flight_energy_j = _metric(flight_energy, DEPOT_ID, node_id) + _metric(flight_energy, node_id, DEPOT_ID)
    route_energy_j = flight_energy_j + hover_gap_s * hover_power
    if route_energy_j > energy_limit:
        return None

    added_air_time_s = flight_time_s + hover_gap_s + (battery_swap_time_s if drone_state.routes else 0.0)
    projected_drone_time_s = drone_state.total_time_s + added_air_time_s
    if projected_drone_time_s > horizon:
        return None

    other_times = [state.total_time_s for state in drone_states.values() if state.drone_id != drone_state.drone_id]
    new_air_makespan_s = max([projected_drone_time_s, *other_times]) if other_times else projected_drone_time_s
    new_closed_loop_s = new_air_makespan_s + new_ground_time_s
    improvement_s = current_closed_loop_s - new_closed_loop_s
    if improvement_s <= 1e-9:
        return None

    return CandidateAction(
        node_id=node_id,
        action_type="new_route",
        drone_id=drone_state.drone_id,
        route_id=None,
        added_air_time_s=added_air_time_s,
        added_energy_j=route_energy_j,
        route_duration_s=flight_time_s + hover_gap_s,
        route_energy_j=route_energy_j,
        new_closed_loop_s=new_closed_loop_s,
        improvement_s=improvement_s,
    )


def _apply_action(action: CandidateAction, drone_states: dict[int, DroneState], hover_gap_s: float) -> None:
    drone_state = drone_states[action.drone_id]
    if action.action_type == "extend_existing":
        for route in drone_state.routes:
            if route.route_id == action.route_id:
                route.hover_times_s[action.node_id] += hover_gap_s
                route.duration_s = action.route_duration_s
                route.energy_j = action.route_energy_j
                drone_state.total_time_s += action.added_air_time_s
                return
        raise RuntimeError(f"Route {action.route_id} not found for drone {action.drone_id}.")

    next_route_id = max(route.route_id for state in drone_states.values() for route in state.routes) + 1
    drone_state.routes.append(
        RouteState(
            drone_id=action.drone_id,
            route_id=next_route_id,
            route_order=len(drone_state.routes) + 1,
            stops=[action.node_id],
            hover_times_s={action.node_id: hover_gap_s},
            duration_s=action.route_duration_s,
            energy_j=action.route_energy_j,
            mode="confirm_revisit",
        )
    )
    drone_state.total_time_s += action.added_air_time_s


def solve_problem2(threshold_multiplier: float = 1.0, enable_swap: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_c_problem_data(WORKBOOK_PATH)
    summary_df, detail_df = _load_problem1_solution()
    target_df = _build_target_table(data, threshold_multiplier=threshold_multiplier)
    base_hover_sum = float(target_df["base_hover_time_s"].sum())

    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    battery_swap_time_s = float(parameter_value(data.params, "battery_swap_time_s"))

    summary_rows: list[dict[str, float | int | str | bool]] = []
    target_rows: list[dict[str, float | int | str | bool]] = []
    route_rows: list[dict[str, float | int | str]] = []

    for drone_count in sorted(summary_df["drone_count"].unique()):
        drone_states = _build_drone_states(detail_df, int(drone_count))
        current_hover = {
            int(node_id): float(hover_time)
            for node_id, hover_time in target_df[["node_id", "base_hover_time_s"]].itertuples(index=False)
        }
        confirmed_nodes: set[int] = set()

        while True:
            remaining_nodes = sorted(set(int(node_id) for node_id in target_df["node_id"]) - confirmed_nodes)
            remaining_manual_points = target_df.loc[
                target_df["node_id"].isin(remaining_nodes), "manual_point_id"
            ].dropna().astype(str).tolist()
            current_ground_time_s, _, _, _ = _ground_stage_metrics(data.manual_points, remaining_manual_points, data.ground_time)
            current_air_makespan_s = max(state.total_time_s for state in drone_states.values())
            current_closed_loop_s = current_air_makespan_s + current_ground_time_s
            best_action: CandidateAction | None = None

            for row in target_df.itertuples(index=False):
                node_id = int(row.node_id)
                if node_id in confirmed_nodes:
                    continue
                hover_gap_s = float(row.direct_confirm_time_s) - float(current_hover[node_id])
                if hover_gap_s <= 1e-9:
                    confirmed_nodes.add(node_id)
                    continue

                new_manual_points = [point_id for point_id in remaining_manual_points if point_id != str(row.manual_point_id)]
                new_ground_time_s, _, _, _ = _ground_stage_metrics(data.manual_points, new_manual_points, data.ground_time)

                located = _route_for_node(drone_states, node_id)
                if located is not None:
                    drone_state, route = located
                    candidate = _evaluate_existing_route_action(
                        node_id,
                        hover_gap_s,
                        drone_state,
                        route,
                        hover_power,
                        energy_limit,
                        horizon,
                        new_ground_time_s,
                        current_closed_loop_s,
                        drone_states,
                    )
                    if candidate is not None and (best_action is None or (candidate.improvement_s, -candidate.added_air_time_s) > (best_action.improvement_s, -best_action.added_air_time_s)):
                        best_action = candidate

                for drone_state in drone_states.values():
                    candidate = _evaluate_new_route_action(
                        node_id,
                        hover_gap_s,
                        drone_state,
                        hover_power,
                        battery_swap_time_s,
                        energy_limit,
                        horizon,
                        data.flight_time,
                        data.flight_energy,
                        new_ground_time_s,
                        current_closed_loop_s,
                        drone_states,
                    )
                    if candidate is not None and (best_action is None or (candidate.improvement_s, -candidate.added_air_time_s) > (best_action.improvement_s, -best_action.added_air_time_s)):
                        best_action = candidate

            if best_action is None:
                break

            row = target_df.loc[target_df["node_id"] == best_action.node_id].iloc[0]
            hover_gap_s = float(row["direct_confirm_time_s"]) - float(current_hover[best_action.node_id])
            _apply_action(best_action, drone_states, hover_gap_s)
            current_hover[best_action.node_id] = float(row["direct_confirm_time_s"])
            confirmed_nodes.add(best_action.node_id)

        manual_nodes = sorted(set(int(node_id) for node_id in target_df["node_id"]) - confirmed_nodes)
        manual_point_ids = target_df.loc[target_df["node_id"].isin(manual_nodes), "manual_point_id"].dropna().astype(str).tolist()
        ground_stage_time_s, ground_travel_time_s, ground_service_time_s, ground_path = _ground_stage_metrics(
            data.manual_points,
            manual_point_ids,
            data.ground_time,
        )
        swap_move_count = 0
        swap_improvement_s = 0.0
        if enable_swap:
            swap_move_count, swap_improvement_s = _swap_local_search(
                drone_states,
                hover_power,
                energy_limit,
                horizon,
                ground_stage_time_s,
                data.flight_time,
                data.flight_energy,
            )
        air_stage_time_s = max(state.total_time_s for state in drone_states.values())
        closed_loop_time_s = air_stage_time_s + ground_stage_time_s
        total_hover_time_s = sum(sum(route.hover_times_s.values()) for state in drone_states.values() for route in state.routes)
        total_air_energy_j = sum(route.energy_j for state in drone_states.values() for route in state.routes)

        summary_rows.append(
            {
                "threshold_multiplier": threshold_multiplier,
                "swap_enabled": enable_swap,
                "drone_count": int(drone_count),
                "air_stage_time_s": air_stage_time_s,
                "air_stage_time_min": air_stage_time_s / 60.0,
                "ground_stage_time_s": ground_stage_time_s,
                "ground_travel_time_s": ground_travel_time_s,
                "ground_service_time_s": ground_service_time_s,
                "closed_loop_time_s": closed_loop_time_s,
                "closed_loop_time_min": closed_loop_time_s / 60.0,
                "direct_confirm_count": len(confirmed_nodes),
                "direct_confirm_ratio": len(confirmed_nodes) / len(target_df),
                "manual_review_count": len(manual_nodes),
                "extra_hover_time_s": total_hover_time_s - base_hover_sum,
                "total_hover_time_s": total_hover_time_s,
                "total_air_energy_j": total_air_energy_j,
                "confirmed_priority_weight": float(target_df.loc[target_df["node_id"].isin(confirmed_nodes), "priority_weight"].sum()),
                "swap_move_count": swap_move_count,
                "swap_improvement_s": swap_improvement_s,
                "ground_path": "-".join(ground_path),
                "feasible_within_horizon": air_stage_time_s <= horizon,
            }
        )

        for row in target_df.itertuples(index=False):
            node_id = int(row.node_id)
            target_rows.append(
                {
                    "threshold_multiplier": threshold_multiplier,
                    "swap_enabled": enable_swap,
                    "drone_count": int(drone_count),
                    "node_id": node_id,
                    "node_name": row.node_name,
                    "priority_weight": float(row.priority_weight),
                    "base_hover_time_s": float(row.base_hover_time_s),
                    "direct_confirm_time_s": float(row.direct_confirm_time_s),
                    "final_hover_time_s": float(current_hover[node_id]),
                    "extra_hover_time_s": float(current_hover[node_id]) - float(row.base_hover_time_s),
                    "is_direct_confirmed": node_id in confirmed_nodes,
                    "manual_point_id": str(row.manual_point_id),
                    "manual_service_time_s": float(row.manual_service_time_s),
                }
            )

        for drone_state in drone_states.values():
            for route in drone_state.routes:
                route_rows.append(
                    {
                        "threshold_multiplier": threshold_multiplier,
                        "swap_enabled": enable_swap,
                        "drone_count": int(drone_count),
                        "drone_id": drone_state.drone_id,
                        "route_id": route.route_id,
                        "route_order": route.route_order,
                        "route_mode": route.mode,
                        "route_path": "0-" + "-".join(str(stop) for stop in route.stops) + "-0",
                        "route_duration_s": route.duration_s,
                        "route_energy_j": route.energy_j,
                        "drone_total_time_s": drone_state.total_time_s,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(target_rows), pd.DataFrame(route_rows)


def main() -> None:
    summary_table, target_table, route_table = solve_problem2()
    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c_problem_problem2_joint_summary.csv"
    target_path = processed_dir / "c_problem_problem2_target_detail.csv"
    route_path = processed_dir / "c_problem_problem2_route_detail.csv"

    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    target_table.to_csv(target_path, index=False, encoding="utf-8-sig")
    route_table.to_csv(route_path, index=False, encoding="utf-8-sig")

    print("Problem 2 joint summary saved to:", summary_path)
    print(summary_table.to_string(index=False))
    print("\nProblem 2 target detail saved to:", target_path)
    print(target_table.head(20).to_string(index=False))
    print("\nProblem 2 route detail saved to:", route_path)
    print(route_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

