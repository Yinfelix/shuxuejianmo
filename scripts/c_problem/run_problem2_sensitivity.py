from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from load_c_data import CProblemData, WORKBOOK_PATH, load_c_problem_data, parameter_value
from run_problem1_heuristic import solve_for_drone_count
from run_problem2_ga import run_ga_with_base_state
from run_problem2_joint import _build_node_state, _greedy_extra_hover, _summarize_node_state


THRESHOLD_SCALES = [1.0, 1.2, 1.4]
BATTERY_SWAP_SCALES = [0.5, 1.0, 1.5]
FOCUS_DRONE_COUNTS = [3, 4]
PROBLEM1_DETAIL_PATH = Path("data/processed/c_problem_problem1_route_detail.csv")


def _build_scaled_data(data: CProblemData, threshold_scale: float) -> CProblemData:
    nodes = data.nodes.copy()
    target_mask = nodes["node_id"] != 0
    nodes["direct_confirm_time_s"] = pd.to_numeric(nodes["direct_confirm_time_s"], errors="coerce").astype(float)
    nodes.loc[target_mask, "direct_confirm_time_s"] = nodes.loc[target_mask, "direct_confirm_time_s"] * threshold_scale
    return CProblemData(
        params=data.params.copy(),
        nodes=nodes,
        manual_points=data.manual_points.copy(),
        flight_time=data.flight_time.copy(),
        flight_energy=data.flight_energy.copy(),
        ground_time=data.ground_time.copy(),
    )


def _load_problem1_detail(drone_count: int) -> pd.DataFrame:
    if PROBLEM1_DETAIL_PATH.exists():
        detail = pd.read_csv(PROBLEM1_DETAIL_PATH, encoding="utf-8-sig")
        filtered = detail.loc[detail["drone_count"] == drone_count].copy()
        if not filtered.empty:
            return filtered
    _, problem1_detail = solve_for_drone_count(drone_count)
    return problem1_detail


def _annotate_single_sortie_capacity(
    node_state: pd.DataFrame,
    data: CProblemData,
    energy_limit: float,
    hover_power: float,
    horizon: float,
) -> pd.DataFrame:
    state = node_state.copy()
    state["direct_roundtrip_time_s"] = state["node_id"].apply(
        lambda node_id: float(data.flight_time.loc["0", str(int(node_id))] + data.flight_time.loc[str(int(node_id)), "0"])
    )
    state["max_hover_by_energy_s"] = (energy_limit - state["direct_out_back_flight_energy_J"]) / hover_power
    state["max_hover_by_time_s"] = horizon - state["direct_roundtrip_time_s"]
    state["max_single_sortie_hover_s"] = state[["max_hover_by_energy_s", "max_hover_by_time_s"]].min(axis=1).clip(lower=0.0)
    state["remaining_confirm_gap_s"] = (state["direct_confirm_time_s"] - state["allocated_hover_time_s"]).clip(lower=0.0)
    state["hard_point"] = state["remaining_confirm_gap_s"] > state["max_single_sortie_hover_s"]
    return state


def _pick_drone_for_sortie(drone_total: dict[int, float], sortie_duration: float, horizon: float) -> int | None:
    feasible = [drone_id for drone_id, total_time in drone_total.items() if total_time + sortie_duration <= horizon]
    if not feasible:
        return None
    return min(feasible, key=lambda drone_id: drone_total[drone_id])


def _hardpoint_first_hover(
    node_state: pd.DataFrame,
    data: CProblemData,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    battery_swap_time: float,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]], list[int], list[int]]:
    state = _annotate_single_sortie_capacity(node_state, data, energy_limit, hover_power, horizon)
    drone_total = {
        int(row.drone_id): float(row.drone_total_time_s)
        for row in state[["drone_id", "drone_total_time_s"]].drop_duplicates().itertuples(index=False)
    }
    records: list[dict[str, float | int | str]] = []
    prepared_nodes: list[int] = []
    infeasible_nodes: list[int] = []

    while True:
        state = _annotate_single_sortie_capacity(state, data, energy_limit, hover_power, horizon)
        hard_points = state.loc[(~state["direct_confirmed"]) & (state["hard_point"])].copy()
        if hard_points.empty:
            break

        hard_points = hard_points.sort_values(by=["priority_weight", "remaining_confirm_gap_s"], ascending=[False, False])
        progress_made = False
        for row in hard_points.itertuples(index=False):
            sortie_hover = float(row.max_single_sortie_hover_s)
            if sortie_hover <= 0:
                infeasible_nodes.append(int(row.node_id))
                continue

            sortie_duration = battery_swap_time + float(row.direct_roundtrip_time_s) + sortie_hover
            selected_drone = _pick_drone_for_sortie(drone_total, sortie_duration, horizon)
            if selected_drone is None:
                if int(row.node_id) not in infeasible_nodes:
                    infeasible_nodes.append(int(row.node_id))
                continue

            node_mask = state["node_id"] == int(row.node_id)
            state.loc[node_mask, "allocated_hover_time_s"] += sortie_hover
            state.loc[node_mask, "extra_hover_added_s"] += sortie_hover
            state.loc[node_mask, "confirmed_by_extra_hover"] = True
            if float(state.loc[node_mask, "allocated_hover_time_s"].iloc[0]) >= float(state.loc[node_mask, "direct_confirm_time_s"].iloc[0]):
                state.loc[node_mask, "direct_confirmed"] = True

            drone_total[selected_drone] += sortie_duration
            state.loc[state["drone_id"] == selected_drone, "drone_total_time_s"] = drone_total[selected_drone]

            prepared_nodes.append(int(row.node_id))
            records.append(
                {
                    "stage": "hardpoint_sortie",
                    "node_id": int(row.node_id),
                    "node_name": str(row.node_name),
                    "assigned_drone_id": selected_drone,
                    "sortie_hover_s": sortie_hover,
                    "sortie_roundtrip_time_s": float(row.direct_roundtrip_time_s),
                    "sortie_duration_s": sortie_duration,
                    "remaining_gap_after_sortie_s": max(
                        0.0,
                        float(state.loc[node_mask, "direct_confirm_time_s"].iloc[0])
                        - float(state.loc[node_mask, "allocated_hover_time_s"].iloc[0]),
                    ),
                }
            )
            progress_made = True
            break

        if not progress_made:
            break

    greedy_state, greedy_records = _greedy_extra_hover(
        node_state=state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
    )
    for record in greedy_records:
        records.append({"stage": "greedy_finish", **record})

    return greedy_state, records, sorted(set(prepared_nodes)), sorted(set(infeasible_nodes))


def _evaluate_solutions_for_scale(
    data: CProblemData,
    drone_count: int,
    threshold_scale: float,
    battery_swap_scale: float,
    horizon: float,
    energy_limit: float,
    hover_power: float,
    battery_swap_time: float,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    problem1_detail = _load_problem1_detail(drone_count)
    base_node_state = _build_node_state(problem1_detail, data.nodes)
    base_node_state = _annotate_single_sortie_capacity(base_node_state, data, energy_limit, hover_power, horizon)

    hard_points = base_node_state.loc[base_node_state["hard_point"], "node_id"].astype(int).tolist()
    hard_summary = {
        "threshold_scale": threshold_scale,
        "battery_swap_scale": battery_swap_scale,
        "battery_swap_time_s": battery_swap_time,
        "drone_count": drone_count,
        "hard_point_count": len(hard_points),
        "hard_point_ids": ",".join(str(node_id) for node_id in hard_points),
    }

    baseline_summary, _ = _summarize_node_state(
        node_state=base_node_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        drone_count=drone_count,
        solution_type="baseline",
        selected_nodes=[],
    )
    baseline_summary = baseline_summary.assign(runtime_s=0.0, **hard_summary)

    greedy_start = perf_counter()
    greedy_state, greedy_selected = _greedy_extra_hover(
        node_state=base_node_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
    )
    greedy_runtime = perf_counter() - greedy_start
    greedy_summary, _ = _summarize_node_state(
        node_state=greedy_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        drone_count=drone_count,
        solution_type="optimize_hover",
        selected_nodes=[str(record["node_id"]) for record in greedy_selected],
    )
    greedy_summary = greedy_summary.assign(runtime_s=greedy_runtime, **hard_summary)

    hard_start = perf_counter()
    hard_state, hard_records, prepared_nodes, infeasible_nodes = _hardpoint_first_hover(
        node_state=base_node_state,
        data=data,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
        battery_swap_time=battery_swap_time,
    )
    hard_runtime = perf_counter() - hard_start
    hard_summary_table, _ = _summarize_node_state(
        node_state=hard_state,
        manual_points=data.manual_points,
        ground_time=data.ground_time,
        drone_count=drone_count,
        solution_type="hardpoint_hover",
        selected_nodes=[str(node_id) for node_id in prepared_nodes],
    )
    hard_summary_table = hard_summary_table.assign(
        runtime_s=hard_runtime,
        prepared_hard_point_ids=",".join(str(node_id) for node_id in prepared_nodes),
        unresolved_hard_point_ids=",".join(str(node_id) for node_id in infeasible_nodes),
        **hard_summary,
    )

    ga_summary, _, _ = run_ga_with_base_state(
        drone_count=drone_count,
        seed=11,
        data=data,
        base_node_state=base_node_state,
        horizon=horizon,
        energy_limit=energy_limit,
        hover_power=hover_power,
    )
    ga_summary = ga_summary.assign(**hard_summary)

    hard_record_table = pd.DataFrame(hard_records)
    if hard_record_table.empty:
        hard_record_table = pd.DataFrame(columns=["stage", "node_id", "node_name", "assigned_drone_id", "sortie_hover_s", "sortie_roundtrip_time_s", "sortie_duration_s", "remaining_gap_after_sortie_s"])
    hard_record_table["threshold_scale"] = threshold_scale
    hard_record_table["drone_count"] = drone_count
    return [baseline_summary, greedy_summary, hard_summary_table, ga_summary], hard_record_table


def main() -> None:
    base_data = load_c_problem_data(WORKBOOK_PATH)
    horizon = float(parameter_value(base_data.params, "operating_horizon_s"))
    energy_limit = float(parameter_value(base_data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(base_data.params, "hover_power_J_per_s"))
    battery_swap_time_base = float(parameter_value(base_data.params, "battery_swap_time_s"))

    compare_frames: list[pd.DataFrame] = []
    hard_detail_frames: list[pd.DataFrame] = []
    for threshold_scale in THRESHOLD_SCALES:
        scaled_data = _build_scaled_data(base_data, threshold_scale)
        for battery_swap_scale in BATTERY_SWAP_SCALES:
            battery_swap_time = battery_swap_time_base * battery_swap_scale
            for drone_count in FOCUS_DRONE_COUNTS:
                summary_tables, hard_record_table = _evaluate_solutions_for_scale(
                    data=scaled_data,
                    drone_count=drone_count,
                    threshold_scale=threshold_scale,
                    battery_swap_scale=battery_swap_scale,
                    horizon=horizon,
                    energy_limit=energy_limit,
                    hover_power=hover_power,
                    battery_swap_time=battery_swap_time,
                )
                compare_frames.extend(summary_tables)
                hard_detail_frames.append(hard_record_table)

    compare_table = pd.concat(compare_frames, ignore_index=True)
    hard_detail_table = pd.concat(hard_detail_frames, ignore_index=True) if hard_detail_frames else pd.DataFrame()

    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    compare_path = output_dir / "c_problem_problem2_sensitivity_compare.csv"
    hard_detail_path = processed_dir / "c_problem_problem2_hardpoint_detail.csv"
    compare_table.to_csv(compare_path, index=False, encoding="utf-8-sig")
    hard_detail_table.to_csv(hard_detail_path, index=False, encoding="utf-8-sig")

    print("Problem 2 sensitivity comparison saved to:", compare_path)
    print(compare_table.to_string(index=False))
    print("\nProblem 2 hard-point detail saved to:", hard_detail_path)
    print(hard_detail_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()