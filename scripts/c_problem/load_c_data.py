from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import hypot

import pandas as pd


WORKBOOK_PATH = Path("2026同济数学建模竞赛赛题/2026C数据.xlsx")


@dataclass
class CProblemData:
    params: pd.DataFrame
    nodes: pd.DataFrame
    manual_points: pd.DataFrame
    flight_time: pd.DataFrame
    flight_energy: pd.DataFrame
    ground_time: pd.DataFrame


def _load_named_table(workbook: Path, sheet_name: str, header_row: int = 2) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row)
    unnamed = [column for column in df.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def _load_matrix(workbook: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=2)
    raw = raw.dropna(how="all")
    raw = raw.rename(columns={raw.columns[0]: "from_id", raw.columns[1]: "from_name"})
    matrix = raw.drop(columns=["from_name"]).copy()
    matrix["from_id"] = matrix["from_id"].astype(str)
    matrix = matrix.set_index("from_id")
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix.apply(pd.to_numeric, errors="coerce")


def _coerce_numeric_value(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _first_numeric_param_value(params: pd.DataFrame, parameter_name: str) -> float | None:
    matches = params.loc[params["parameter"] == parameter_name, "value"]
    for value in matches:
        numeric = _coerce_numeric_value(value)
        if numeric is not None:
            return numeric
    return None


def _fill_effective_energy_limit(params: pd.DataFrame) -> pd.DataFrame:
    params = params.copy()
    effective_mask = params["parameter"] == "effective_energy_limit_J"
    if effective_mask.any():
        explicit_values = params.loc[effective_mask, "value"].map(_coerce_numeric_value)
        if explicit_values.notna().any():
            params.loc[effective_mask, "value"] = explicit_values
            return params

    battery_capacity = _first_numeric_param_value(params, "battery_capacity_J")
    safety_reserve = _first_numeric_param_value(params, "safety_reserve_J")
    if battery_capacity is None or safety_reserve is None:
        return params

    fallback_value = battery_capacity - safety_reserve
    if effective_mask.any():
        params.loc[effective_mask, "value"] = fallback_value
    else:
        params.loc[len(params)] = {
            "parameter": "effective_energy_limit_J",
            "value": fallback_value,
        }
    return params


def _build_ground_time_fallback(manual_points: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    points = manual_points.copy()
    points["manual_point_id"] = points["manual_point_id"].astype(str)
    points["x_m"] = pd.to_numeric(points["x_m"], errors="coerce")
    points["y_m"] = pd.to_numeric(points["y_m"], errors="coerce")
    walking_speed_match = params.loc[params["parameter"] == "walking_speed_mps", "value"]
    detour_match = params.loc[params["parameter"] == "walking_detour_factor", "value"]
    walking_speed = float(walking_speed_match.iloc[0]) if not walking_speed_match.empty else 1.35
    detour_factor = float(detour_match.iloc[0]) if not detour_match.empty else 1.25

    point_ids = points["manual_point_id"].tolist()
    matrix = pd.DataFrame(index=point_ids, columns=point_ids, dtype=float)
    coords = {
        str(row.manual_point_id): (float(row.x_m), float(row.y_m))
        for row in points.itertuples(index=False)
    }
    for from_id in point_ids:
        from_x, from_y = coords[from_id]
        for to_id in point_ids:
            to_x, to_y = coords[to_id]
            if from_id == to_id:
                matrix.loc[from_id, to_id] = 0.0
                continue
            distance_m = hypot(from_x - to_x, from_y - to_y) * detour_factor
            matrix.loc[from_id, to_id] = distance_m / walking_speed
    return matrix


def _flight_leg_metrics(
    from_xyz: tuple[float, float, float],
    to_xyz: tuple[float, float, float],
    params: pd.DataFrame,
) -> tuple[float, float]:
    horizontal_speed = float(parameter_value(params, "horizontal_speed_mps", 12.0) or 12.0)
    vertical_speed = float(parameter_value(params, "vertical_speed_mps", 4.0) or 4.0)
    horizontal_energy = float(parameter_value(params, "horizontal_energy_J_per_m", 16.5) or 16.5)
    up_energy = float(parameter_value(params, "up_energy_J_per_m", 23.0) or 23.0)
    down_energy = float(parameter_value(params, "down_energy_J_per_m", 12.0) or 12.0)

    from_x, from_y, from_z = from_xyz
    to_x, to_y, to_z = to_xyz
    horizontal_distance = hypot(from_x - to_x, from_y - to_y)
    vertical_delta = to_z - from_z
    ascent = max(0.0, vertical_delta)
    descent = max(0.0, -vertical_delta)

    travel_time = horizontal_distance / horizontal_speed + (ascent + descent) / vertical_speed
    travel_energy = horizontal_distance * horizontal_energy + ascent * up_energy + descent * down_energy
    return travel_time, travel_energy


def _fill_flight_fallback_edges(
    nodes: pd.DataFrame,
    params: pd.DataFrame,
    flight_time: pd.DataFrame,
    flight_energy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filled_time = flight_time.copy()
    filled_energy = flight_energy.copy()

    coord_table = nodes[["node_id", "x_m", "y_m", "z_m"]].copy()
    coord_table = coord_table.dropna(subset=["node_id", "x_m", "y_m", "z_m"])
    coord_map = {
        str(int(row.node_id)): (float(row.x_m), float(row.y_m), float(row.z_m))
        for row in coord_table.itertuples(index=False)
    }
    for from_key, from_coords in coord_map.items():
        for to_key, to_coords in coord_map.items():
            if from_key == to_key:
                filled_time.loc[from_key, to_key] = 0.0
                filled_energy.loc[from_key, to_key] = 0.0
                continue

            leg_time, leg_energy = _flight_leg_metrics(from_coords, to_coords, params)
            if pd.isna(filled_time.loc[from_key, to_key]):
                filled_time.loc[from_key, to_key] = leg_time
            if pd.isna(filled_energy.loc[from_key, to_key]):
                filled_energy.loc[from_key, to_key] = leg_energy

    return filled_time, filled_energy


def load_c_problem_data(workbook: Path | None = None) -> CProblemData:
    workbook = workbook or WORKBOOK_PATH
    workbook = Path(workbook)

    params = _fill_effective_energy_limit(_load_named_table(workbook, "UAV_Params"))
    nodes = _load_named_table(workbook, "NodeData")
    manual_points = _load_named_table(workbook, "ManualPoints")
    flight_time = _load_matrix(workbook, "FlightTime")
    flight_energy = _load_matrix(workbook, "FlightEnergy")
    ground_time = _load_matrix(workbook, "GroundTime")

    numeric_columns = [
        "node_id",
        "x_m",
        "y_m",
        "z_m",
        "priority_weight",
        "base_hover_time_s",
        "direct_confirm_time_s",
        "extra_confirm_time_s",
        "manual_x_m",
        "manual_y_m",
        "manual_service_time_s",
        "direct_out_back_flight_energy_J",
        "direct_base_roundtrip_total_energy_J",
        "direct_confirm_roundtrip_total_energy_J",
    ]
    for column in numeric_columns:
        if column in nodes.columns:
            nodes[column] = pd.to_numeric(nodes[column], errors="coerce")

    if "node_id" in nodes.columns:
        nodes["node_id"] = nodes["node_id"].astype("Int64")

    manual_numeric_columns = ["mapped_target_id", "x_m", "y_m", "manual_service_time_s"]
    for column in manual_numeric_columns:
        if column in manual_points.columns:
            manual_points[column] = pd.to_numeric(manual_points[column], errors="coerce")

    if ground_time.isna().all().all():
        ground_time = _build_ground_time_fallback(manual_points, params)

    flight_time, flight_energy = _fill_flight_fallback_edges(nodes, params, flight_time, flight_energy)

    return CProblemData(
        params=params,
        nodes=nodes,
        manual_points=manual_points,
        flight_time=flight_time,
        flight_energy=flight_energy,
        ground_time=ground_time,
    )


def parameter_value(params: pd.DataFrame, parameter_name: str, default: float | None = None) -> float | None:
    value = _first_numeric_param_value(params, parameter_name)
    if value is None:
        return default
    return value
