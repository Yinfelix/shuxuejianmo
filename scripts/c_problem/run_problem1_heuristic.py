from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import pstdev

import pandas as pd
from sklearn.cluster import KMeans

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value


DEPOT_ID = 0
PROBLEM1_DETAIL_PATH = Path("data/processed/c_problem_problem1_route_detail.csv")


@dataclass
class Target:
    node_id: int
    node_name: str
    x_m: float
    y_m: float
    z_m: float
    priority_weight: float
    base_hover_time_s: float
    direct_confirm_time_s: float
    manual_service_time_s: float
    direct_hover_gap_s: float
    downstream_gain_s: float
    construct_score: float


@dataclass
class Route:
    route_id: int
    stops: list[int] = field(default_factory=list)
    hover_times_s: dict[int, float] = field(default_factory=dict)
    flight_time_s: float = 0.0
    hover_time_s: float = 0.0
    total_duration_s: float = 0.0
    flight_energy_j: float = 0.0
    hover_energy_j: float = 0.0
    total_energy_j: float = 0.0


@dataclass
class DronePlan:
    drone_id: int
    routes: list[Route] = field(default_factory=list)
    total_time_s: float = 0.0


def _metric(matrix: pd.DataFrame, from_id: int, to_id: int) -> float:
    return float(matrix.loc[str(from_id), str(to_id)])


def _metric_or_none(matrix: pd.DataFrame, from_id: int, to_id: int) -> float | None:
    try:
        value = float(matrix.loc[str(from_id), str(to_id)])
    except KeyError:
        return None
    return value if math.isfinite(value) else None


def _other_drone_max_time(drone_plans: list[DronePlan], excluded_index: int) -> float:
    other_times = [other.total_time_s for idx, other in enumerate(drone_plans) if idx != excluded_index]
    return max(other_times) if other_times else 0.0


def _solo_route(target: Target, flight_time: pd.DataFrame, flight_energy: pd.DataFrame, hover_power: float, route_id: int) -> Route | None:
    out_time = _metric_or_none(flight_time, DEPOT_ID, target.node_id)
    back_time = _metric_or_none(flight_time, target.node_id, DEPOT_ID)
    out_energy = _metric_or_none(flight_energy, DEPOT_ID, target.node_id)
    back_energy = _metric_or_none(flight_energy, target.node_id, DEPOT_ID)
    if None in (out_time, back_time, out_energy, back_energy):
        return None
    hover_energy = target.base_hover_time_s * hover_power
    return Route(
        route_id=route_id,
        stops=[target.node_id],
        hover_times_s={target.node_id: target.base_hover_time_s},
        flight_time_s=out_time + back_time,
        hover_time_s=target.base_hover_time_s,
        total_duration_s=out_time + back_time + target.base_hover_time_s,
        flight_energy_j=out_energy + back_energy,
        hover_energy_j=hover_energy,
        total_energy_j=out_energy + back_energy + hover_energy,
    )


def _route_metrics(
    stops: list[int],
    hover_times_s: dict[int, float],
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
    hover_power: float,
) -> tuple[float, float, float, float, float, float] | None:
    if not stops:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    route_nodes = [DEPOT_ID, *stops, DEPOT_ID]
    total_flight_time = 0.0
    total_flight_energy = 0.0
    for from_id, to_id in zip(route_nodes, route_nodes[1:], strict=False):
        leg_time = _metric_or_none(flight_time, from_id, to_id)
        leg_energy = _metric_or_none(flight_energy, from_id, to_id)
        if leg_time is None or leg_energy is None:
            return None
        total_flight_time += leg_time
        total_flight_energy += leg_energy

    total_hover_time = sum(float(hover_times_s[node_id]) for node_id in stops)
    total_hover_energy = total_hover_time * hover_power
    total_duration = total_flight_time + total_hover_time
    total_energy = total_flight_energy + total_hover_energy
    return total_flight_time, total_hover_time, total_duration, total_flight_energy, total_hover_energy, total_energy


def _insertion_delta(
    route: Route,
    target: Target,
    insert_position: int,
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
    hover_power: float,
) -> tuple[float, float] | None:
    new_stops = route.stops.copy()
    new_stops.insert(insert_position, target.node_id)
    new_hover_times = route.hover_times_s.copy()
    new_hover_times[target.node_id] = target.base_hover_time_s
    metrics = _route_metrics(new_stops, new_hover_times, flight_time, flight_energy, hover_power)
    if metrics is None:
        return None
    _, _, new_total_duration, _, _, new_total_energy = metrics
    return new_total_duration - route.total_duration_s, new_total_energy - route.total_energy_j


def _insert_target(
    route: Route,
    target: Target,
    insert_position: int,
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
    hover_power: float,
) -> None:
    route.stops.insert(insert_position, target.node_id)
    route.hover_times_s[target.node_id] = target.base_hover_time_s
    metrics = _route_metrics(route.stops, route.hover_times_s, flight_time, flight_energy, hover_power)
    if metrics is None:
        raise RuntimeError(f"Inserted route became infeasible for node {target.node_id}.")
    (
        route.flight_time_s,
        route.hover_time_s,
        route.total_duration_s,
        route.flight_energy_j,
        route.hover_energy_j,
        route.total_energy_j,
    ) = metrics


def _normalized_series(values: pd.Series) -> pd.Series:
    max_value = float(values.max()) if not values.empty else 0.0
    if max_value <= 0:
        return pd.Series(0.0, index=values.index)
    return values.astype(float) / max_value


def _candidate_action_score(
    projected_makespan: float,
    projected_finish: float,
    projected_energy: float,
    energy_limit: float,
    horizon: float,
    target: Target,
) -> float:
    residual_energy_ratio = max(0.0, energy_limit - projected_energy) / energy_limit if energy_limit > 0 else 0.0
    residual_time_ratio = max(0.0, horizon - projected_finish) / horizon if horizon > 0 else 0.0
    return (
        projected_makespan
        - 0.12 * target.downstream_gain_s
        - 16.0 * target.construct_score
        - 10.0 * residual_energy_ratio
        - 8.0 * residual_time_ratio
    )


def _target_distance_sq(target_a: Target, target_b: Target) -> float:
    return (
        (target_a.x_m - target_b.x_m) ** 2
        + (target_a.y_m - target_b.y_m) ** 2
        + (target_a.z_m - target_b.z_m) ** 2
    )


def _build_kmeans_seed_order(base_targets: list[Target], drone_count: int) -> list[Target]:
    cluster_count = min(drone_count, len(base_targets))
    if cluster_count <= 0:
        return []
    if cluster_count == 1:
        centroid_like = Target(
            node_id=-1,
            node_name="cluster_center",
            x_m=sum(target.x_m for target in base_targets) / len(base_targets),
            y_m=sum(target.y_m for target in base_targets) / len(base_targets),
            z_m=sum(target.z_m for target in base_targets) / len(base_targets),
            priority_weight=0.0,
            base_hover_time_s=0.0,
            direct_confirm_time_s=0.0,
            manual_service_time_s=0.0,
            direct_hover_gap_s=0.0,
            downstream_gain_s=0.0,
            construct_score=0.0,
        )
        representative = min(
            base_targets,
            key=lambda target: (_target_distance_sq(target, centroid_like), -target.construct_score, -target.priority_weight),
        )
        remaining = [target for target in base_targets if target.node_id != representative.node_id]
        remaining.sort(key=lambda target: (_target_distance_sq(target, representative), -target.construct_score, -target.priority_weight))
        return [representative, *remaining]

    coord_frame = pd.DataFrame(
        [{"x_m": target.x_m, "y_m": target.y_m, "z_m": target.z_m, "node_id": target.node_id} for target in base_targets]
    )
    model = KMeans(n_clusters=cluster_count, n_init=10, random_state=0)
    labels = model.fit_predict(coord_frame[["x_m", "y_m", "z_m"]])

    cluster_members: dict[int, list[Target]] = {cluster_id: [] for cluster_id in range(cluster_count)}
    for target, label in zip(base_targets, labels, strict=False):
        cluster_members[int(label)].append(target)

    cluster_rows: list[tuple[int, Target]] = []
    for cluster_id in range(cluster_count):
        center_x, center_y, center_z = model.cluster_centers_[cluster_id]
        center_probe = Target(
            node_id=-1,
            node_name="cluster_center",
            x_m=float(center_x),
            y_m=float(center_y),
            z_m=float(center_z),
            priority_weight=0.0,
            base_hover_time_s=0.0,
            direct_confirm_time_s=0.0,
            manual_service_time_s=0.0,
            direct_hover_gap_s=0.0,
            downstream_gain_s=0.0,
            construct_score=0.0,
        )
        representative = min(
            cluster_members[cluster_id],
            key=lambda target: (_target_distance_sq(target, center_probe), -target.construct_score, -target.priority_weight),
        )
        cluster_rows.append((cluster_id, representative))

    cluster_rows.sort(key=lambda item: (-item[1].construct_score, -item[1].priority_weight, item[1].base_hover_time_s))
    ordered_targets: list[Target] = [representative for _, representative in cluster_rows]
    used_node_ids = {target.node_id for target in ordered_targets}

    for cluster_id, representative in cluster_rows:
        remaining = [target for target in cluster_members[cluster_id] if target.node_id not in used_node_ids]
        remaining.sort(
            key=lambda target: (
                _target_distance_sq(target, representative),
                -target.construct_score,
                -target.priority_weight,
                -target.downstream_gain_s,
            )
        )
        ordered_targets.extend(remaining)
        used_node_ids.update(target.node_id for target in remaining)

    leftovers = [target for target in base_targets if target.node_id not in used_node_ids]
    leftovers.sort(key=lambda target: (-target.construct_score, -target.priority_weight, -target.downstream_gain_s))
    ordered_targets.extend(leftovers)
    return ordered_targets


def _build_target_orders(base_targets: list[Target], flight_time: pd.DataFrame, drone_count: int, max_seed_count: int = 6) -> list[list[Target]]:
    if not base_targets:
        return []

    orders: list[list[Target]] = []
    seen_orders: set[tuple[int, ...]] = set()

    def append_order(seed: Target | None) -> None:
        if seed is None:
            order = list(base_targets)
        else:
            remaining = [target for target in base_targets if target.node_id != seed.node_id]

            def sort_key(candidate: Target) -> tuple[bool, float, float, float, float]:
                travel_time = _metric_or_none(flight_time, seed.node_id, candidate.node_id)
                return (
                    travel_time is None,
                    travel_time if travel_time is not None else float("inf"),
                    -candidate.construct_score,
                    -candidate.priority_weight,
                    -candidate.downstream_gain_s,
                )

            remaining.sort(key=sort_key)
            order = [seed, *remaining]

        signature = tuple(target.node_id for target in order)
        if signature not in seen_orders:
            seen_orders.add(signature)
            orders.append(order)

    kmeans_order = _build_kmeans_seed_order(base_targets, drone_count)
    if kmeans_order:
        signature = tuple(target.node_id for target in kmeans_order)
        if signature not in seen_orders:
            seen_orders.add(signature)
            orders.append(kmeans_order)

    append_order(seed=None)
    seed_pool = sorted(
        base_targets,
        key=lambda target: (-target.construct_score, -target.priority_weight, -target.downstream_gain_s, target.base_hover_time_s),
    )[:max_seed_count]
    for seed in seed_pool:
        append_order(seed=seed)
    return orders


def _finalize_solution(
    drone_plans: list[DronePlan],
    drone_count: int,
    targets: list[Target],
    horizon: float,
    use_f_guidance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_rows: list[dict[str, float | int | str]] = []
    total_energy = 0.0
    total_flight_energy = 0.0
    total_hover_time = 0.0
    total_route_count = 0
    for drone_plan in drone_plans:
        route_rows.extend(_route_detail_rows(drone_plan, drone_count))
        total_route_count += len(drone_plan.routes)
        for route in drone_plan.routes:
            total_energy += route.total_energy_j
            total_flight_energy += route.flight_energy_j
            total_hover_time += route.hover_time_s

    makespan = max(drone_plan.total_time_s for drone_plan in drone_plans)
    load_std = pstdev([drone_plan.total_time_s for drone_plan in drone_plans]) if drone_count > 1 else 0.0

    summary = pd.DataFrame(
        [
            {
                "drone_count": drone_count,
                "construct_mode": "f_guided" if use_f_guidance else "legacy_fallback",
                "target_count": len(targets),
                "route_count": total_route_count,
                "makespan_s": makespan,
                "makespan_min": makespan / 60.0,
                "total_hover_time_s": total_hover_time,
                "total_flight_energy_j": total_flight_energy,
                "total_energy_j": total_energy,
                "max_drone_time_s": makespan,
                "load_std_s": load_std,
                "feasible_within_horizon": makespan <= horizon,
                "horizon_s": horizon,
            }
        ]
    )
    detail = pd.DataFrame(route_rows)
    return summary, detail


def _try_construct_solution(
    targets: list[Target],
    drone_count: int,
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
    hover_power: float,
    horizon: float,
    energy_limit: float,
    battery_swap_time: float,
    use_f_guidance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    drone_plans = [DronePlan(drone_id=drone_id) for drone_id in range(1, drone_count + 1)]
    route_id = 1

    for target in targets:
        candidate_actions: list[tuple[float, float, float, float, str, int, int, int, float, float]] = []
        for drone_index, drone_plan in enumerate(drone_plans):
            solo_route = _solo_route(target, flight_time, flight_energy, hover_power, route_id)
            if solo_route is not None:
                added_if_new = solo_route.total_duration_s
                if drone_plan.routes:
                    added_if_new += battery_swap_time
                projected_if_new = drone_plan.total_time_s + added_if_new
                if projected_if_new <= horizon and solo_route.total_energy_j <= energy_limit:
                    projected_makespan = max(
                        projected_if_new,
                        _other_drone_max_time(drone_plans, drone_index),
                    )
                    action_score = (
                        _candidate_action_score(
                            projected_makespan=projected_makespan,
                            projected_finish=projected_if_new,
                            projected_energy=solo_route.total_energy_j,
                            energy_limit=energy_limit,
                            horizon=horizon,
                            target=target,
                        )
                        if use_f_guidance
                        else projected_makespan
                    )
                    candidate_actions.append(
                        (
                            action_score,
                            projected_makespan,
                            projected_if_new,
                            solo_route.total_energy_j,
                            "new",
                            drone_index,
                            -1,
                            -1,
                            added_if_new,
                            0.0,
                        )
                    )

            for current_route_index, current_route in enumerate(drone_plan.routes):
                for insert_position in range(len(current_route.stops) + 1):
                    delta = _insertion_delta(
                        current_route,
                        target,
                        insert_position,
                        flight_time,
                        flight_energy,
                        hover_power,
                    )
                    if delta is None:
                        continue
                    delta_time, delta_energy = delta
                    if current_route.total_energy_j + delta_energy > energy_limit:
                        continue
                    projected_if_insert = drone_plan.total_time_s + delta_time
                    if projected_if_insert > horizon:
                        continue
                    projected_makespan = max(
                        projected_if_insert,
                        _other_drone_max_time(drone_plans, drone_index),
                    )
                    action_score = (
                        _candidate_action_score(
                            projected_makespan=projected_makespan,
                            projected_finish=projected_if_insert,
                            projected_energy=current_route.total_energy_j + delta_energy,
                            energy_limit=energy_limit,
                            horizon=horizon,
                            target=target,
                        )
                        if use_f_guidance
                        else projected_makespan
                    )
                    candidate_actions.append(
                        (
                            action_score,
                            projected_makespan,
                            projected_if_insert,
                            current_route.total_energy_j + delta_energy,
                            "insert",
                            drone_index,
                            current_route_index,
                            insert_position,
                            delta_time,
                            delta_energy,
                        )
                    )

        if not candidate_actions:
            return None

        if use_f_guidance:
            min_makespan = min(item[1] for item in candidate_actions)
            shortlisted = [item for item in candidate_actions if item[1] <= min_makespan + 30.0]
            shortlisted.sort(key=lambda item: (item[0], item[1], item[4] == "new", item[2], item[3]))
            _, _, _, _, action, chosen_drone_index, chosen_route_index, insert_position, delta_time, delta_energy = shortlisted[0]
        else:
            candidate_actions.sort(key=lambda item: (item[1], item[4] == "new", item[2], item[3]))
            _, _, _, _, action, chosen_drone_index, chosen_route_index, insert_position, delta_time, delta_energy = candidate_actions[0]
        chosen_plan = drone_plans[chosen_drone_index]

        if action == "new":
            solo_route = _solo_route(target, flight_time, flight_energy, hover_power, route_id)
            if solo_route is None:
                return None
            chosen_plan.total_time_s += solo_route.total_duration_s
            if chosen_plan.routes:
                chosen_plan.total_time_s += battery_swap_time
            chosen_plan.routes.append(solo_route)
            route_id += 1
        else:
            current_route = chosen_plan.routes[chosen_route_index]
            _insert_target(
                current_route,
                target,
                insert_position,
                flight_time,
                flight_energy,
                hover_power,
            )
            chosen_plan.total_time_s += delta_time

    return _finalize_solution(drone_plans, drone_count, targets, horizon, use_f_guidance)


def _route_detail_rows(drone_plan: DronePlan, drone_count: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    running_time = 0.0
    for route_index, route in enumerate(drone_plan.routes, start=1):
        if route_index > 1:
            running_time += 0.0
        route_path = "0-" + "-".join(str(stop) for stop in route.stops) + "-0"
        for stop_order, node_id in enumerate(route.stops, start=1):
            rows.append(
                {
                    "drone_count": drone_count,
                    "drone_id": drone_plan.drone_id,
                    "route_id": route.route_id,
                    "route_order": route_index,
                    "stop_order": stop_order,
                    "node_id": node_id,
                    "hover_time_s": route.hover_times_s[node_id],
                    "route_duration_s": route.total_duration_s,
                    "route_energy_j": route.total_energy_j,
                    "route_path": route_path,
                    "drone_total_time_s": drone_plan.total_time_s,
                }
            )
    return rows


def _route_summary_from_detail(detail: pd.DataFrame, energy_limit: float, horizon: float) -> pd.DataFrame:
    route_rows: list[dict[str, float | int | str]] = []
    ordered = detail.sort_values(["drone_count", "drone_id", "route_order", "stop_order"]).copy()
    for (drone_count, drone_id, route_id, route_order), route_group in ordered.groupby(["drone_count", "drone_id", "route_id", "route_order"]):
        node_sequence = route_group["node_id"].astype(int).tolist()
        route_duration = float(route_group["route_duration_s"].iloc[0])
        route_energy = float(route_group["route_energy_j"].iloc[0])
        drone_total_time = float(route_group["drone_total_time_s"].iloc[0])
        route_rows.append(
            {
                "drone_count": int(drone_count),
                "drone_id": int(drone_id),
                "route_id": int(route_id),
                "route_order": int(route_order),
                "flight_task_count": 1,
                "stop_count": len(node_sequence),
                "start_node_id": int(node_sequence[0]),
                "end_node_id": int(node_sequence[-1]),
                "route_duration_s": route_duration,
                "route_duration_min": route_duration / 60.0,
                "route_energy_j": route_energy,
                "energy_utilization_ratio": route_energy / energy_limit if energy_limit > 0 else 0.0,
                "remaining_energy_j": max(0.0, energy_limit - route_energy),
                "drone_total_time_s": drone_total_time,
                "drone_time_utilization_ratio": drone_total_time / horizon if horizon > 0 else 0.0,
                "path_trace": str(route_group["route_path"].iloc[0]),
                "node_sequence": " -> ".join(str(node_id) for node_id in node_sequence),
            }
        )
    return pd.DataFrame(route_rows)


def _summary_from_cached_detail(detail: pd.DataFrame, drone_count: int, horizon: float, construct_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = detail.copy()
    route_table = detail[["drone_id", "route_id", "route_duration_s", "route_energy_j"]].drop_duplicates().copy()
    total_route_count = int(len(route_table))
    total_energy = float(route_table["route_energy_j"].sum())
    total_flight_energy = total_energy
    total_hover_time = float(detail["hover_time_s"].sum())
    makespan = float(detail["drone_total_time_s"].max())
    drone_totals = detail[["drone_id", "drone_total_time_s"]].drop_duplicates()["drone_total_time_s"].astype(float).tolist()
    load_std = pstdev(drone_totals) if len(drone_totals) > 1 else 0.0

    summary = pd.DataFrame(
        [
            {
                "drone_count": drone_count,
                "construct_mode": construct_mode,
                "target_count": int(detail["node_id"].nunique()),
                "route_count": total_route_count,
                "makespan_s": makespan,
                "makespan_min": makespan / 60.0,
                "total_hover_time_s": total_hover_time,
                "total_flight_energy_j": total_flight_energy,
                "total_energy_j": total_energy,
                "max_drone_time_s": makespan,
                "load_std_s": load_std,
                "feasible_within_horizon": makespan <= horizon,
                "horizon_s": horizon,
            }
        ]
    )
    return summary, detail


def _load_cached_problem1_solution(drone_count: int, horizon: float) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not PROBLEM1_DETAIL_PATH.exists():
        return None
    detail = pd.read_csv(PROBLEM1_DETAIL_PATH, encoding="utf-8-sig")
    filtered = detail.loc[detail["drone_count"] == drone_count].copy()
    if filtered.empty:
        return None
    return _summary_from_cached_detail(filtered, drone_count, horizon, construct_mode="cached_fallback")


def solve_for_drone_count(drone_count: int, use_f_guidance: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_c_problem_data(WORKBOOK_PATH)
    nodes = data.nodes.copy()
    targets_df = nodes.loc[nodes["node_id"] != 0].copy()

    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    battery_swap_time = float(parameter_value(data.params, "battery_swap_time_s"))

    targets_df["manual_service_time_s"] = pd.to_numeric(targets_df["manual_service_time_s"], errors="coerce").fillna(0.0)
    targets_df["direct_hover_gap_s"] = (
        pd.to_numeric(targets_df["direct_confirm_time_s"], errors="coerce").fillna(0.0)
        - pd.to_numeric(targets_df["base_hover_time_s"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    targets_df["downstream_gain_s"] = targets_df["manual_service_time_s"] + targets_df["direct_hover_gap_s"]
    targets_df["construct_score"] = (
        1.00 * _normalized_series(targets_df["priority_weight"])
        + 0.90 * _normalized_series(targets_df["manual_service_time_s"])
        + 0.80 * _normalized_series(targets_df["direct_hover_gap_s"])
        + 0.25 * _normalized_series(targets_df["base_hover_time_s"])
    )

    sort_columns = ["priority_weight", "base_hover_time_s", "direct_confirm_time_s"]
    ascending = [False, False, True]
    if use_f_guidance:
        sort_columns.append("construct_score")
        ascending.append(False)

    targets = [
        Target(
            node_id=int(row.node_id),
            node_name=str(row.node_name),
            x_m=float(row.x_m),
            y_m=float(row.y_m),
            z_m=float(row.z_m),
            priority_weight=float(row.priority_weight),
            base_hover_time_s=float(row.base_hover_time_s),
            direct_confirm_time_s=float(row.direct_confirm_time_s),
            manual_service_time_s=float(row.manual_service_time_s),
            direct_hover_gap_s=float(row.direct_hover_gap_s),
            downstream_gain_s=float(row.downstream_gain_s),
            construct_score=float(row.construct_score),
        )
        for row in targets_df.sort_values(by=sort_columns, ascending=ascending).itertuples(index=False)
    ]
    target_orders = _build_target_orders(targets, data.flight_time, drone_count)
    for target_order in target_orders:
        constructed = _try_construct_solution(
            targets=target_order,
            drone_count=drone_count,
            flight_time=data.flight_time,
            flight_energy=data.flight_energy,
            hover_power=hover_power,
            horizon=horizon,
            energy_limit=energy_limit,
            battery_swap_time=battery_swap_time,
            use_f_guidance=use_f_guidance,
        )
        if constructed is not None:
            return constructed

    if use_f_guidance:
        try:
            return solve_for_drone_count(drone_count, use_f_guidance=False)
        except RuntimeError:
            cached_solution = _load_cached_problem1_solution(drone_count, horizon)
            if cached_solution is not None:
                return cached_solution
    cached_solution = _load_cached_problem1_solution(drone_count, horizon)
    if cached_solution is not None:
        return cached_solution
    raise RuntimeError(f"No feasible assignment found after multistart construction with K={drone_count}.")


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    k_max = int(parameter_value(data.params, "K_max"))
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    horizon = float(parameter_value(data.params, "operating_horizon_s"))

    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for drone_count in range(1, k_max + 1):
        summary, detail = solve_for_drone_count(drone_count)
        summary_frames.append(summary)
        detail_frames.append(detail)

    summary_table = pd.concat(summary_frames, ignore_index=True)
    detail_table = pd.concat(detail_frames, ignore_index=True)
    route_summary_table = _route_summary_from_detail(detail_table, energy_limit=energy_limit, horizon=horizon)

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c_problem_problem1_heuristic_summary.csv"
    route_summary_path = output_dir / "c_problem_problem1_route_summary.csv"
    detail_path = processed_dir / "c_problem_problem1_route_detail.csv"
    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    route_summary_table.to_csv(route_summary_path, index=False, encoding="utf-8-sig")
    detail_table.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Problem 1 heuristic summary saved to:", summary_path)
    print(summary_table.to_string(index=False))
    print("\nProblem 1 route summary saved to:", route_summary_path)
    print(route_summary_table.head(20).to_string(index=False))
    print("\nProblem 1 route detail saved to:", detail_path)
    print(detail_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()