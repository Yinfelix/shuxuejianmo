from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import pstdev

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value


DEPOT_ID = 0
PROBLEM1_DETAIL_PATH = Path("data/processed/c_problem_problem1_route_detail.csv")


@dataclass
class Target:
    node_id: int
    node_name: str
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


def _other_drone_max_time(drone_plans: list[DronePlan], excluded_index: int) -> float:
    other_times = [other.total_time_s for idx, other in enumerate(drone_plans) if idx != excluded_index]
    return max(other_times) if other_times else 0.0


def _solo_route(target: Target, flight_time: pd.DataFrame, flight_energy: pd.DataFrame, hover_power: float, route_id: int) -> Route:
    out_time = _metric(flight_time, DEPOT_ID, target.node_id)
    back_time = _metric(flight_time, target.node_id, DEPOT_ID)
    out_energy = _metric(flight_energy, DEPOT_ID, target.node_id)
    back_energy = _metric(flight_energy, target.node_id, DEPOT_ID)
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


def _append_delta(route: Route, target: Target, flight_time: pd.DataFrame, flight_energy: pd.DataFrame, hover_power: float) -> tuple[float, float]:
    last_stop = route.stops[-1]
    delta_time = (
        _metric(flight_time, last_stop, target.node_id)
        + _metric(flight_time, target.node_id, DEPOT_ID)
        - _metric(flight_time, last_stop, DEPOT_ID)
        + target.base_hover_time_s
    )
    delta_energy = (
        _metric(flight_energy, last_stop, target.node_id)
        + _metric(flight_energy, target.node_id, DEPOT_ID)
        - _metric(flight_energy, last_stop, DEPOT_ID)
        + target.base_hover_time_s * hover_power
    )
    return delta_time, delta_energy


def _append_target(route: Route, target: Target, delta_time: float, delta_energy: float, flight_time: pd.DataFrame, flight_energy: pd.DataFrame, hover_power: float) -> None:
    last_stop = route.stops[-1]
    route.stops.append(target.node_id)
    route.hover_times_s[target.node_id] = target.base_hover_time_s
    route.flight_time_s += (
        _metric(flight_time, last_stop, target.node_id)
        + _metric(flight_time, target.node_id, DEPOT_ID)
        - _metric(flight_time, last_stop, DEPOT_ID)
    )
    route.flight_energy_j += (
        _metric(flight_energy, last_stop, target.node_id)
        + _metric(flight_energy, target.node_id, DEPOT_ID)
        - _metric(flight_energy, last_stop, DEPOT_ID)
    )
    route.hover_time_s += target.base_hover_time_s
    route.hover_energy_j += target.base_hover_time_s * hover_power
    route.total_duration_s += delta_time
    route.total_energy_j += delta_energy


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

    drone_plans = [DronePlan(drone_id=drone_id) for drone_id in range(1, drone_count + 1)]
    route_id = 1

    for target in targets:
        candidate_actions: list[tuple[float, float, float, float, str, int, float, float]] = []
        for drone_index, drone_plan in enumerate(drone_plans):
            solo_route = _solo_route(target, data.flight_time, data.flight_energy, hover_power, route_id)
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
                        added_if_new,
                        0.0,
                    )
                )

            if drone_plan.routes:
                current_route = drone_plan.routes[-1]
                delta_time, delta_energy = _append_delta(
                    current_route,
                    target,
                    data.flight_time,
                    data.flight_energy,
                    hover_power,
                )
                if current_route.total_energy_j + delta_energy <= energy_limit:
                    projected_if_append = drone_plan.total_time_s + delta_time
                    if projected_if_append <= horizon:
                        projected_makespan = max(
                            projected_if_append,
                            _other_drone_max_time(drone_plans, drone_index),
                        )
                        action_score = (
                            _candidate_action_score(
                                projected_makespan=projected_makespan,
                                projected_finish=projected_if_append,
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
                                projected_if_append,
                                current_route.total_energy_j + delta_energy,
                                "append",
                                drone_index,
                                delta_time,
                                delta_energy,
                            )
                        )

        if not candidate_actions:
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
            raise RuntimeError(f"No feasible assignment found for node {target.node_id} with K={drone_count}.")

        if use_f_guidance:
            min_makespan = min(item[1] for item in candidate_actions)
            shortlisted = [item for item in candidate_actions if item[1] <= min_makespan + 30.0]
            shortlisted.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            _, _, _, _, action, chosen_drone_index, delta_time, delta_energy = shortlisted[0]
        else:
            candidate_actions.sort(key=lambda item: (item[1], item[2], item[3]))
            _, _, _, _, action, chosen_drone_index, delta_time, delta_energy = candidate_actions[0]
        chosen_plan = drone_plans[chosen_drone_index]

        if action == "new":
            solo_route = _solo_route(target, data.flight_time, data.flight_energy, hover_power, route_id)
            chosen_plan.total_time_s += solo_route.total_duration_s
            if chosen_plan.routes:
                chosen_plan.total_time_s += battery_swap_time
            chosen_plan.routes.append(solo_route)
            route_id += 1
        else:
            current_route = chosen_plan.routes[-1]
            _append_target(
                current_route,
                target,
                delta_time,
                delta_energy,
                data.flight_time,
                data.flight_energy,
                hover_power,
            )
            chosen_plan.total_time_s += delta_time

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


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    k_max = int(parameter_value(data.params, "K_max"))

    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for drone_count in range(1, k_max + 1):
        summary, detail = solve_for_drone_count(drone_count)
        summary_frames.append(summary)
        detail_frames.append(detail)

    summary_table = pd.concat(summary_frames, ignore_index=True)
    detail_table = pd.concat(detail_frames, ignore_index=True)

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c_problem_problem1_heuristic_summary.csv"
    detail_path = processed_dir / "c_problem_problem1_route_detail.csv"
    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_table.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Problem 1 heuristic summary saved to:", summary_path)
    print(summary_table.to_string(index=False))
    print("\nProblem 1 route detail saved to:", detail_path)
    print(detail_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()