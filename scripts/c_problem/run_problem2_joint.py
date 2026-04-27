from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value
from run_problem1_heuristic import solve_for_drone_count


DEPOT_MANUAL_ID = "P0"


def _nearest_neighbor_path(ground_time: pd.DataFrame, points: list[str], start: str = DEPOT_MANUAL_ID) -> tuple[list[str], float]:
    if not points:
        return [start, start], 0.0

    unvisited = set(points)
    route = [start]
    current = start
    travel_time = 0.0

    while unvisited:
        candidates = []
        for point in unvisited:
            travel_value = pd.to_numeric(pd.Series([ground_time.loc[current, point]]), errors="coerce").iloc[0]
            if pd.notna(travel_value):
                candidates.append((float(travel_value), point))
        if not candidates:
            raise RuntimeError(f"No finite ground-time edge found from {current} to remaining points {sorted(unvisited)}")
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, next_point = candidates[0]
        travel_time += float(ground_time.loc[current, next_point])
        route.append(next_point)
        unvisited.remove(next_point)
        current = next_point

    return_value = pd.to_numeric(pd.Series([ground_time.loc[current, start]]), errors="coerce").iloc[0]
    if pd.isna(return_value):
        raise RuntimeError(f"No finite ground-time edge found from {current} back to {start}")
    travel_time += float(return_value)
    route.append(start)
    return route, travel_time


def _route_travel_savings(ground_time: pd.DataFrame, route: list[str]) -> dict[str, float]:
    savings: dict[str, float] = {}
    if len(route) <= 2:
        return savings

    for idx in range(1, len(route) - 1):
        current = route[idx]
        previous = route[idx - 1]
        following = route[idx + 1]
        savings[current] = (
            float(ground_time.loc[previous, current])
            + float(ground_time.loc[current, following])
            - float(ground_time.loc[previous, following])
        )
    return savings


def _build_node_state(problem1_detail: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    node_hover = (
        problem1_detail.groupby("node_id", as_index=False)["hover_time_s"]
        .sum()
        .rename(columns={"hover_time_s": "allocated_hover_time_s"})
    )
    assignment = problem1_detail.drop_duplicates(subset=["node_id"])[
        ["node_id", "drone_id", "route_id", "route_duration_s", "route_energy_j", "drone_total_time_s"]
    ].copy()

    node_state = nodes.loc[nodes["node_id"] != 0].copy()
    node_state = node_state.merge(node_hover, on="node_id", how="left")
    node_state = node_state.merge(assignment, on="node_id", how="left")
    node_state["allocated_hover_time_s"] = node_state["allocated_hover_time_s"].fillna(0.0)
    node_state["extra_hover_added_s"] = 0.0
    node_state["direct_confirmed"] = node_state["allocated_hover_time_s"] >= node_state["direct_confirm_time_s"]
    node_state["confirmed_by_extra_hover"] = False
    return node_state


def _build_manual_review_table(node_state: pd.DataFrame, manual_points: pd.DataFrame) -> pd.DataFrame:
    manual_candidates = manual_points.loc[manual_points["mapped_target_id"] != 0].copy()
    manual_candidates["mapped_target_id"] = manual_candidates["mapped_target_id"].astype("Int64")

    pending_nodes = node_state.loc[
        ~node_state["direct_confirmed"],
        ["node_id", "node_name", "priority_weight", "allocated_hover_time_s", "direct_confirm_time_s"],
    ].copy()
    pending_nodes["node_id"] = pending_nodes["node_id"].astype("Int64")
    pending_manual = pending_nodes.merge(
        manual_candidates,
        left_on="node_id",
        right_on="mapped_target_id",
        how="left",
    )
    missing_manual = pending_manual.loc[pending_manual["manual_point_id"].isna(), "node_id"].tolist()
    if missing_manual:
        raise RuntimeError(f"Missing manual point mapping for nodes: {missing_manual}")
    return pending_manual


def _summarize_node_state(
    node_state: pd.DataFrame,
    manual_points: pd.DataFrame,
    ground_time: pd.DataFrame,
    drone_count: int,
    solution_type: str,
    selected_nodes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pending_manual = _build_manual_review_table(node_state, manual_points)
    manual_point_ids = pending_manual["manual_point_id"].astype(str).tolist()
    manual_route, ground_travel_time = _nearest_neighbor_path(ground_time, manual_point_ids)
    ground_service_time = float(pending_manual["manual_service_time_s"].fillna(0.0).sum())
    ground_completion_time = ground_travel_time + ground_service_time

    air_completion_time = float(node_state["drone_total_time_s"].max())
    direct_confirm_count = int(node_state["direct_confirmed"].sum())
    manual_review_count = int((~node_state["direct_confirmed"]).sum())
    target_count = int(len(node_state))
    total_added_hover = float(node_state["extra_hover_added_s"].sum())

    joint_summary = pd.DataFrame(
        [
            {
                "drone_count": drone_count,
                "solution_type": solution_type,
                "air_completion_time_s": air_completion_time,
                "ground_travel_time_s": ground_travel_time,
                "ground_service_time_s": ground_service_time,
                "ground_completion_time_s": ground_completion_time,
                "total_closed_loop_time_s": air_completion_time + ground_completion_time,
                "direct_confirm_count": direct_confirm_count,
                "direct_confirm_ratio": direct_confirm_count / target_count if target_count else 0.0,
                "manual_review_count": manual_review_count,
                "added_hover_time_s": total_added_hover,
                "selected_direct_confirm_nodes": ",".join(selected_nodes or []),
                "manual_route": "-".join(manual_route),
            }
        ]
    )

    node_state = node_state.assign(
        drone_count=drone_count,
        solution_type=solution_type,
        manual_route="-".join(manual_route),
    )
    detail_columns = [
        "drone_count",
        "solution_type",
        "node_id",
        "node_name",
        "priority_weight",
        "allocated_hover_time_s",
        "extra_hover_added_s",
        "direct_confirm_time_s",
        "drone_id",
        "route_id",
        "manual_point_id",
        "manual_service_time_s",
        "direct_confirmed",
        "confirmed_by_extra_hover",
        "manual_route",
    ]
    joint_detail = node_state[detail_columns].copy()
    return joint_summary, joint_detail


def evaluate_selected_nodes(
    node_state: pd.DataFrame,
    manual_points: pd.DataFrame,
    ground_time: pd.DataFrame,
    drone_count: int,
    selected_node_ids: list[int],
    horizon: float,
    energy_limit: float,
    hover_power: float,
    solution_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    node_state = node_state.copy()
    route_energy = {
        (int(row.drone_id), int(row.route_id)): float(row.route_energy_j)
        for row in node_state[["drone_id", "route_id", "route_energy_j"]].drop_duplicates().itertuples(index=False)
    }
    drone_total = {
        int(row.drone_id): float(row.drone_total_time_s)
        for row in node_state[["drone_id", "drone_total_time_s"]].drop_duplicates().itertuples(index=False)
    }

    selected_node_ids = sorted(set(selected_node_ids))
    for node_id in selected_node_ids:
        node_mask = node_state["node_id"] == node_id
        if not node_mask.any():
            return pd.DataFrame(), pd.DataFrame(), False
        node_row = node_state.loc[node_mask].iloc[0]
        extra_hover_needed = max(0.0, float(node_row.direct_confirm_time_s - node_row.allocated_hover_time_s))
        if extra_hover_needed <= 0:
            continue

        route_key = (int(node_row.drone_id), int(node_row.route_id))
        drone_key = int(node_row.drone_id)
        new_route_energy = route_energy[route_key] + extra_hover_needed * hover_power
        new_drone_total = drone_total[drone_key] + extra_hover_needed
        if new_route_energy > energy_limit or new_drone_total > horizon:
            return pd.DataFrame(), pd.DataFrame(), False

        route_energy[route_key] = new_route_energy
        drone_total[drone_key] = new_drone_total
        node_state.loc[node_mask, "allocated_hover_time_s"] += extra_hover_needed
        node_state.loc[node_mask, "extra_hover_added_s"] += extra_hover_needed
        node_state.loc[node_mask, "direct_confirmed"] = True
        node_state.loc[node_mask, "confirmed_by_extra_hover"] = True

    for drone_id, total_time in drone_total.items():
        node_state.loc[node_state["drone_id"] == drone_id, "drone_total_time_s"] = total_time
    for (drone_id, route_id), total_energy in route_energy.items():
        node_state.loc[(node_state["drone_id"] == drone_id) & (node_state["route_id"] == route_id), "route_energy_j"] = total_energy

    summary, detail = _summarize_node_state(
        node_state=node_state,
        manual_points=manual_points,
        ground_time=ground_time,
        drone_count=drone_count,
        solution_type=solution_type,
        selected_nodes=[str(node_id) for node_id in selected_node_ids],
    )
    return summary, detail, True


def _greedy_extra_hover(
    node_state: pd.DataFrame,
    manual_points: pd.DataFrame,
    ground_time: pd.DataFrame,
    horizon: float,
    energy_limit: float,
    hover_power: float,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    node_state = node_state.copy()
    route_energy = {
        (int(row.drone_id), int(row.route_id)): float(row.route_energy_j)
        for row in node_state[["drone_id", "route_id", "route_energy_j"]].drop_duplicates().itertuples(index=False)
    }
    drone_total = {
        int(row.drone_id): float(row.drone_total_time_s)
        for row in node_state[["drone_id", "drone_total_time_s"]].drop_duplicates().itertuples(index=False)
    }
    current_air_completion = max(drone_total.values()) if drone_total else 0.0
    selected_records: list[dict[str, float | int | str]] = []

    while True:
        pending_manual = _build_manual_review_table(node_state, manual_points)
        manual_point_ids = pending_manual["manual_point_id"].astype(str).tolist()
        manual_route, _ = _nearest_neighbor_path(ground_time, manual_point_ids)
        travel_savings = _route_travel_savings(ground_time, manual_route)
        service_times = pending_manual.set_index("manual_point_id")["manual_service_time_s"].to_dict()

        best_choice: dict[str, float | int | str] | None = None
        for row in pending_manual.itertuples(index=False):
            node_row = node_state.loc[node_state["node_id"] == row.node_id].iloc[0]
            additional_hover_needed = float(node_row.direct_confirm_time_s - node_row.allocated_hover_time_s)
            if additional_hover_needed <= 0:
                continue

            route_key = (int(node_row.drone_id), int(node_row.route_id))
            drone_key = int(node_row.drone_id)
            new_route_energy = route_energy[route_key] + additional_hover_needed * hover_power
            new_drone_total = drone_total[drone_key] + additional_hover_needed
            if new_route_energy > energy_limit or new_drone_total > horizon:
                continue

            new_air_completion = max(current_air_completion, new_drone_total)
            air_delta = new_air_completion - current_air_completion
            ground_saving = float(service_times[str(row.manual_point_id)]) + float(travel_savings.get(str(row.manual_point_id), 0.0))
            total_improvement = ground_saving - air_delta
            score = total_improvement / additional_hover_needed

            candidate = {
                "node_id": int(row.node_id),
                "node_name": str(row.node_name),
                "manual_point_id": str(row.manual_point_id),
                "priority_weight": float(node_row.priority_weight),
                "additional_hover_needed_s": additional_hover_needed,
                "ground_saving_s": ground_saving,
                "air_delta_s": air_delta,
                "total_improvement_s": total_improvement,
                "score": score,
                "drone_id": drone_key,
                "route_id": int(node_row.route_id),
            }
            if best_choice is None or (
                candidate["total_improvement_s"],
                candidate["score"],
                candidate["priority_weight"],
            ) > (
                best_choice["total_improvement_s"],
                best_choice["score"],
                best_choice["priority_weight"],
            ):
                best_choice = candidate

        if best_choice is None or float(best_choice["total_improvement_s"]) <= 0:
            break

        node_id = int(best_choice["node_id"])
        drone_id = int(best_choice["drone_id"])
        route_id = int(best_choice["route_id"])
        gap = float(best_choice["additional_hover_needed_s"])
        route_key = (drone_id, route_id)

        node_state.loc[node_state["node_id"] == node_id, "allocated_hover_time_s"] += gap
        node_state.loc[node_state["node_id"] == node_id, "extra_hover_added_s"] += gap
        node_state.loc[node_state["node_id"] == node_id, "direct_confirmed"] = True
        node_state.loc[node_state["node_id"] == node_id, "confirmed_by_extra_hover"] = True

        route_energy[route_key] += gap * hover_power
        drone_total[drone_id] += gap
        current_air_completion = max(current_air_completion, drone_total[drone_id])

        node_state.loc[node_state["drone_id"] == drone_id, "drone_total_time_s"] = drone_total[drone_id]
        node_state.loc[(node_state["drone_id"] == drone_id) & (node_state["route_id"] == route_id), "route_energy_j"] = route_energy[route_key]
        selected_records.append(best_choice)

    return node_state, selected_records


def evaluate_joint_solution(drone_count: int, optimize_hover: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_c_problem_data(WORKBOOK_PATH)
    problem1_summary, problem1_detail = solve_for_drone_count(drone_count)

    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    horizon = float(parameter_value(data.params, "operating_horizon_s"))

    node_state = _build_node_state(problem1_detail, data.nodes)
    selected_records: list[dict[str, float | int | str]] = []
    if optimize_hover:
        node_state, selected_records = _greedy_extra_hover(
            node_state=node_state,
            manual_points=data.manual_points,
            ground_time=data.ground_time,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
        )

    joint_summary, joint_detail = _summarize_node_state(
        node_state=node_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        drone_count=drone_count,
        solution_type="optimize_hover" if optimize_hover else "baseline",
        selected_nodes=[str(record["node_id"]) for record in selected_records],
    )
    return joint_summary, joint_detail


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    k_max = int(data.params.loc[data.params["parameter"] == "K_max", "value"].iloc[0])

    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for drone_count in range(1, k_max + 1):
        for optimize_hover in (False, True):
            summary, detail = evaluate_joint_solution(drone_count, optimize_hover=optimize_hover)
            summary_frames.append(summary)
            detail_frames.append(detail)

    summary_table = pd.concat(summary_frames, ignore_index=True)
    detail_table = pd.concat(detail_frames, ignore_index=True)

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c_problem_problem2_joint_summary.csv"
    detail_path = processed_dir / "c_problem_problem2_manual_review_detail.csv"
    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_table.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Problem 2 joint summary saved to:", summary_path)
    print(summary_table.to_string(index=False))
    print("\nProblem 2 manual review detail saved to:", detail_path)
    print(detail_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()