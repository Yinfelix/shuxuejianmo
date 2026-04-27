from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data
from run_problem1_heuristic import solve_for_drone_count
from run_problem2_joint import (
    _build_manual_review_table,
    _build_node_state,
    _nearest_neighbor_path,
    _route_travel_savings,
    _summarize_node_state,
)


PROBLEM1_DETAIL_PATH = Path("data/processed/c_problem_problem1_route_detail.csv")
WEIGHT_KEYS = [
    "ground_gain",
    "priority",
    "energy_slack",
    "time_slack",
    "route_progress",
    "hover_gap_penalty",
]
WEIGHT_BOUNDS = {
    "ground_gain": (0.0, 3.0),
    "priority": (0.0, 2.0),
    "energy_slack": (0.0, 2.0),
    "time_slack": (0.0, 2.0),
    "route_progress": (0.0, 1.5),
    "hover_gap_penalty": (0.0, 2.5),
}
DEFAULT_WEIGHT_VECTOR = [1.35, 0.45, 0.55, 0.45, 0.25, 1.05]


def _safe_ratio(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return value / scale


def vector_to_weight_dict(weight_vector: list[float]) -> dict[str, float]:
    return {key: float(value) for key, value in zip(WEIGHT_KEYS, weight_vector, strict=False)}


def clamp_weight_vector(weight_vector: list[float]) -> list[float]:
    clamped: list[float] = []
    for key, value in zip(WEIGHT_KEYS, weight_vector, strict=False):
        lower, upper = WEIGHT_BOUNDS[key]
        clamped.append(min(upper, max(lower, float(value))))
    return clamped


def _load_problem1_detail(drone_count: int) -> pd.DataFrame:
    if PROBLEM1_DETAIL_PATH.exists():
        detail = pd.read_csv(PROBLEM1_DETAIL_PATH, encoding="utf-8-sig")
        filtered = detail.loc[detail["drone_count"] == drone_count].copy()
        if not filtered.empty:
            return filtered
    _, problem1_detail = solve_for_drone_count(drone_count)
    return problem1_detail


def enrich_node_state_with_route_context(node_state: pd.DataFrame, problem1_detail: pd.DataFrame) -> pd.DataFrame:
    route_context_rows: list[dict[str, float | int]] = []
    ordered = problem1_detail.sort_values(by=["drone_id", "route_id", "stop_order"])
    for (_, _), route_detail in ordered.groupby(["drone_id", "route_id"]):
        previous_node_id = 0
        stop_count = len(route_detail)
        for row in route_detail.itertuples(index=False):
            stop_order = int(row.stop_order)
            route_context_rows.append(
                {
                    "node_id": int(row.node_id),
                    "stop_order": stop_order,
                    "route_stop_count": stop_count,
                    "previous_node_id": previous_node_id,
                    "route_progress_ratio": stop_order / stop_count if stop_count else 0.0,
                }
            )
            previous_node_id = int(row.node_id)

    route_context = pd.DataFrame(route_context_rows)
    enriched = node_state.merge(route_context, on="node_id", how="left")
    enriched["stop_order"] = pd.to_numeric(enriched["stop_order"], errors="coerce").fillna(0).astype(int)
    enriched["route_stop_count"] = pd.to_numeric(enriched["route_stop_count"], errors="coerce").fillna(0).astype(int)
    enriched["previous_node_id"] = pd.to_numeric(enriched["previous_node_id"], errors="coerce").fillna(0).astype(int)
    enriched["route_progress_ratio"] = pd.to_numeric(enriched["route_progress_ratio"], errors="coerce").fillna(0.0)
    return enriched


def build_guidance_base_state(drone_count: int, data=None) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    data = data or load_c_problem_data(WORKBOOK_PATH)
    problem1_detail = _load_problem1_detail(drone_count)
    node_state = _build_node_state(problem1_detail, data.nodes)
    node_state = enrich_node_state_with_route_context(node_state, problem1_detail)
    return data, problem1_detail, node_state


def _ensure_guidance_columns(node_state: pd.DataFrame) -> pd.DataFrame:
    ensured = node_state.copy()
    defaults = {
        "stop_order": 0,
        "route_stop_count": 0,
        "previous_node_id": 0,
        "route_progress_ratio": 0.0,
    }
    for column, default_value in defaults.items():
        if column not in ensured.columns:
            ensured[column] = default_value
    ensured["stop_order"] = pd.to_numeric(ensured["stop_order"], errors="coerce").fillna(0).astype(int)
    ensured["route_stop_count"] = pd.to_numeric(ensured["route_stop_count"], errors="coerce").fillna(0).astype(int)
    ensured["previous_node_id"] = pd.to_numeric(ensured["previous_node_id"], errors="coerce").fillna(0).astype(int)
    ensured["route_progress_ratio"] = pd.to_numeric(ensured["route_progress_ratio"], errors="coerce").fillna(0.0)
    return ensured


def _scored_candidates(
    node_state: pd.DataFrame,
    manual_points: pd.DataFrame,
    ground_time: pd.DataFrame,
    route_energy: dict[tuple[int, int], float],
    drone_total: dict[int, float],
    current_air_completion: float,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    weight_vector: list[float],
) -> list[dict[str, float | int | str]]:
    pending_manual = _build_manual_review_table(node_state, manual_points)
    if pending_manual.empty:
        return []

    manual_point_ids = pending_manual["manual_point_id"].astype(str).tolist()
    manual_route, _ = _nearest_neighbor_path(ground_time, manual_point_ids)
    travel_savings = _route_travel_savings(ground_time, manual_route)

    raw_candidates: list[dict[str, float | int | str]] = []
    for row in pending_manual.itertuples(index=False):
        node_row = node_state.loc[node_state["node_id"] == int(row.node_id)].iloc[0]
        hover_gap = max(0.0, float(node_row.direct_confirm_time_s - node_row.allocated_hover_time_s))
        if hover_gap <= 0:
            continue

        route_key = (int(node_row.drone_id), int(node_row.route_id))
        drone_key = int(node_row.drone_id)
        new_route_energy = route_energy[route_key] + hover_gap * hover_power
        new_drone_total = drone_total[drone_key] + hover_gap
        if new_route_energy > energy_limit or new_drone_total > horizon:
            continue

        new_air_completion = max(current_air_completion, new_drone_total)
        air_delta = new_air_completion - current_air_completion
        service_gain = float(row.manual_service_time_s) if pd.notna(row.manual_service_time_s) else 0.0
        travel_gain = float(travel_savings.get(str(row.manual_point_id), 0.0))
        ground_gain = service_gain + travel_gain

        raw_candidates.append(
            {
                "node_id": int(row.node_id),
                "node_name": str(row.node_name),
                "manual_point_id": str(row.manual_point_id),
                "priority_weight": float(node_row.priority_weight),
                "hover_gap_s": hover_gap,
                "ground_gain_s": ground_gain,
                "service_gain_s": service_gain,
                "travel_gain_s": travel_gain,
                "air_delta_s": air_delta,
                "total_improvement_s": ground_gain - air_delta,
                "drone_id": drone_key,
                "route_id": int(node_row.route_id),
                "new_route_energy_j": new_route_energy,
                "new_drone_total_s": new_drone_total,
                "residual_energy_ratio": _safe_ratio(energy_limit - new_route_energy, energy_limit),
                "residual_time_ratio": _safe_ratio(horizon - new_drone_total, horizon),
                "route_progress_ratio": float(node_row.route_progress_ratio),
                "previous_node_id": int(node_row.previous_node_id),
            }
        )

    if not raw_candidates:
        return []

    weights = vector_to_weight_dict(clamp_weight_vector(weight_vector))
    max_ground_gain = max(float(candidate["ground_gain_s"]) for candidate in raw_candidates)
    max_priority = max(float(candidate["priority_weight"]) for candidate in raw_candidates)
    max_hover_gap = max(float(candidate["hover_gap_s"]) for candidate in raw_candidates)

    for candidate in raw_candidates:
        candidate["f_value"] = (
            weights["ground_gain"] * _safe_ratio(float(candidate["ground_gain_s"]), max_ground_gain)
            + weights["priority"] * _safe_ratio(float(candidate["priority_weight"]), max_priority)
            + weights["energy_slack"] * float(candidate["residual_energy_ratio"])
            + weights["time_slack"] * float(candidate["residual_time_ratio"])
            + weights["route_progress"] * float(candidate["route_progress_ratio"])
            - weights["hover_gap_penalty"] * _safe_ratio(float(candidate["hover_gap_s"]), max_hover_gap)
        )

    raw_candidates.sort(
        key=lambda candidate: (
            float(candidate["f_value"]),
            float(candidate["total_improvement_s"]),
            float(candidate["priority_weight"]),
            -float(candidate["hover_gap_s"]),
        ),
        reverse=True,
    )
    return raw_candidates


def apply_f_guidance_policy(
    node_state: pd.DataFrame,
    manual_points: pd.DataFrame,
    ground_time: pd.DataFrame,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    weight_vector: list[float],
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    guided_state = _ensure_guidance_columns(node_state)
    route_energy = {
        (int(row.drone_id), int(row.route_id)): float(row.route_energy_j)
        for row in guided_state[["drone_id", "route_id", "route_energy_j"]].drop_duplicates().itertuples(index=False)
    }
    drone_total = {
        int(row.drone_id): float(row.drone_total_time_s)
        for row in guided_state[["drone_id", "drone_total_time_s"]].drop_duplicates().itertuples(index=False)
    }
    current_air_completion = max(drone_total.values()) if drone_total else 0.0
    selected_records: list[dict[str, float | int | str]] = []

    while True:
        candidates = _scored_candidates(
            node_state=guided_state,
            manual_points=manual_points,
            ground_time=ground_time,
            route_energy=route_energy,
            drone_total=drone_total,
            current_air_completion=current_air_completion,
            horizon=horizon,
            energy_limit=energy_limit,
            hover_power=hover_power,
            weight_vector=weight_vector,
        )
        if not candidates:
            break

        best_choice = candidates[0]
        if float(best_choice["f_value"]) <= 0.0:
            break

        node_id = int(best_choice["node_id"])
        drone_id = int(best_choice["drone_id"])
        route_id = int(best_choice["route_id"])
        hover_gap = float(best_choice["hover_gap_s"])
        route_key = (drone_id, route_id)

        guided_state.loc[guided_state["node_id"] == node_id, "allocated_hover_time_s"] += hover_gap
        guided_state.loc[guided_state["node_id"] == node_id, "extra_hover_added_s"] += hover_gap
        guided_state.loc[guided_state["node_id"] == node_id, "direct_confirmed"] = True
        guided_state.loc[guided_state["node_id"] == node_id, "confirmed_by_extra_hover"] = True

        route_energy[route_key] += hover_gap * hover_power
        drone_total[drone_id] += hover_gap
        current_air_completion = max(current_air_completion, drone_total[drone_id])

        guided_state.loc[guided_state["drone_id"] == drone_id, "drone_total_time_s"] = drone_total[drone_id]
        guided_state.loc[
            (guided_state["drone_id"] == drone_id) & (guided_state["route_id"] == route_id),
            "route_energy_j",
        ] = route_energy[route_key]
        selected_records.append(best_choice)

    return guided_state, selected_records


def evaluate_guidance_weights(
    drone_count: int,
    data,
    base_node_state: pd.DataFrame,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    weight_vector: list[float],
    solution_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float | int | str]]]:
    guided_state, selected_records = apply_f_guidance_policy(
        node_state=base_node_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
        weight_vector=weight_vector,
    )
    summary, detail = _summarize_node_state(
        node_state=guided_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        drone_count=drone_count,
        solution_type=solution_type,
        selected_nodes=[str(record["node_id"]) for record in selected_records],
    )
    weight_dict = vector_to_weight_dict(clamp_weight_vector(weight_vector))
    for key, value in weight_dict.items():
        summary[f"f_{key}_weight"] = value
    return summary, detail, selected_records