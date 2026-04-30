from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import pstdev

import pandas as pd

from clustering.hierarchical_clustering import (
    ClusterResult,
    WeightedNode,
    assign_drones_to_clusters,
    build_cluster_summary,
    cluster_nodes,
    cluster_nodes_fixed_k,
    nodes_from_dataframe,
)
from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value


DEPOT_ID = 0
OUTPUT_DIR = Path("outputs/tables")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DroneAssignment:
    drone_id: int
    node_ids: list[int] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    route_order: list[int] = field(default_factory=list)
    total_time_s: float = 0.0
    total_energy_j: float = 0.0
    total_hover_s: float = 0.0
    energy_remaining_j: float = 0.0
    energy_capacity_j: float = 0.0
    route_count: int = 0
    all_assigned_completed: bool = True


@dataclass
class RollingResult:
    makespan_s: float
    total_energy_j: float
    total_hover_s: float
    load_std_s: float
    cluster_count: int
    n_heuristics_tried: int
    horizon_feasible: bool
    detail_df: pd.DataFrame
    summary_df: pd.DataFrame
    cluster_df: pd.DataFrame


def _metric(matrix: pd.DataFrame, from_id: int, to_id: int) -> float:
    return float(matrix.loc[str(from_id), str(to_id)])


def _score_action(
    delta_time: float,
    drone_time: float,
    delta_energy: float,
    drone_energy: float,
    capacity_j: float,
    horizon: float,
    node_priority: float,
    downstream_gain: float,
    construct_score: float,
) -> float:
    residual_energy_ratio = max(0.0, capacity_j - drone_energy) / capacity_j if capacity_j > 0 else 0.0
    residual_time_ratio = max(0.0, horizon - drone_time) / horizon if horizon > 0 else 0.0
    return (
        delta_time
        - 0.12 * downstream_gain
        - 16.0 * construct_score
        - 10.0 * residual_energy_ratio
        - 8.0 * residual_time_ratio
    )


def _greedy_assign(
    drone_id: int,
    node_ids: list[int],
    data,
    capacity_j: float,
    hover_power: float,
    horizon: float,
    battery_swap_time: float,
    all_node_priority: dict[int, float],
    all_node_downstream: dict[int, float],
    all_node_construct: dict[int, float],
    start_id: int = 0,
) -> DroneAssignment:
    assign = DroneAssignment(
        drone_id=drone_id,
        node_ids=node_ids,
        energy_capacity_j=capacity_j,
        energy_remaining_j=capacity_j,
    )
    if not node_ids:
        return assign

    remaining = set(node_ids)
    while remaining:
        if assign.route_count > 0:
            assign.total_time_s += battery_swap_time
        assign.route_count += 1
        current = start_id
        if not assign.route_order:
            assign.route_order = [start_id]
        else:
            assign.route_order.append(start_id)

        progress_made = False
        best_node = None
        sortie_energy = capacity_j
        while remaining:
            best_node = None
            best_score = float("inf")
            for cand in remaining:
                seg_time = _metric(data.flight_time, current, cand)
                seg_energy = _metric(data.flight_energy, current, cand)
                node_row = data.nodes.loc[data.nodes["node_id"] == cand].iloc[0]
                hover_time = float(node_row.base_hover_time_s)
                back_time = _metric(data.flight_time, cand, start_id)
                back_energy = _metric(data.flight_energy, cand, start_id)
                delta_time = seg_time + hover_time + back_time - _metric(data.flight_time, current, start_id)
                delta_energy = seg_energy + hover_time * hover_power + back_energy - _metric(data.flight_energy, current, start_id)
                if sortie_energy - delta_energy < 0:
                    continue
                score = _score_action(
                    delta_time=delta_time,
                    drone_time=assign.total_time_s,
                    delta_energy=delta_energy,
                    drone_energy=assign.total_energy_j,
                    capacity_j=capacity_j,
                    horizon=horizon,
                    node_priority=all_node_priority.get(cand, 0.0),
                    downstream_gain=all_node_downstream.get(cand, 0.0),
                    construct_score=all_node_construct.get(cand, 0.0),
                )
                if score < best_score:
                    best_score = score
                    best_node = cand

            if best_node is None:
                break

            seg_time = _metric(data.flight_time, current, best_node)
            seg_energy = _metric(data.flight_energy, current, best_node)
            node_row = data.nodes.loc[data.nodes["node_id"] == best_node].iloc[0]
            hover_time = float(node_row.base_hover_time_s)

            assign.total_time_s += seg_time + hover_time
            assign.total_hover_s += hover_time
            assign.total_energy_j += seg_energy + hover_time * hover_power
            assign.energy_remaining_j = capacity_j - (seg_energy + hover_time * hover_power)
            sortie_energy -= seg_energy + hover_time * hover_power

            assign.route_order.append(best_node)
            assign.completed.append(best_node)
            remaining.remove(best_node)
            current = best_node
            progress_made = True

        if not progress_made:
            assign.all_assigned_completed = False
            break

        back_time = _metric(data.flight_time, current, start_id)
        back_energy = _metric(data.flight_energy, current, start_id)
        assign.total_time_s += back_time
        assign.total_energy_j += back_energy
        assign.energy_remaining_j = capacity_j - back_energy
        assign.route_order.append(start_id)

    assign.all_assigned_completed = assign.all_assigned_completed and not remaining
    return assign


def _build_detail_rows(assignments: list[DroneAssignment], data, node_cluster_map: dict[int, int]) -> list[dict]:
    rows = []
    for assign in assignments:
        for order_idx, node_id in enumerate(assign.route_order, start=1):
            if node_id == DEPOT_ID:
                continue
            node_row = data.nodes.loc[data.nodes["node_id"] == node_id].iloc[0]
            rows.append({
                "drone_id": assign.drone_id,
                "node_id": node_id,
                "cluster_id": node_cluster_map.get(node_id, -1),
                "stop_order": order_idx,
                "hover_time_s": float(node_row.base_hover_time_s),
                "route_time_s": assign.total_time_s,
                "route_energy_j": assign.total_energy_j,
                "route": "->".join(str(n) for n in assign.route_order),
                "drone_total_time_s": assign.total_time_s,
            })
    return rows


def _compute_score(makespan: float, total_energy: float, load_std: float, horizon: float) -> float:
    infeasibility_penalty = 1e9 if makespan > horizon else 0.0
    return makespan + 0.001 * total_energy + 5.0 * load_std + infeasibility_penalty


def _try_config(
    data,
    wnodes: list[WeightedNode],
    drone_count: int,
    forced_k: int,
    capacity_j: float,
    hover_power: float,
    horizon: float,
    battery_swap_time: float,
    node_cluster_map: dict[int, int],
    all_node_priority: dict[int, float],
    all_node_downstream: dict[int, float],
    all_node_construct: dict[int, float],
    verbose: bool = False,
) -> tuple[list[DroneAssignment], ClusterResult] | None:
    cluster_result = cluster_nodes_fixed_k(wnodes, forced_k)
    for node, label in zip(wnodes, cluster_result.labels, strict=True):
        node_cluster_map[int(node.node_id)] = int(label)

    drone_assign = assign_drones_to_clusters(cluster_result.labels, drone_count, wnodes)

    assignments: dict[int, DroneAssignment] = {}
    for drone_id, node_ids in drone_assign.items():
        assignments[drone_id] = _greedy_assign(
            drone_id=drone_id,
            node_ids=node_ids,
            data=data,
            capacity_j=capacity_j,
            hover_power=hover_power,
            horizon=horizon,
            battery_swap_time=battery_swap_time,
            all_node_priority=all_node_priority,
            all_node_downstream=all_node_downstream,
            all_node_construct=all_node_construct,
        )

    if any(not assignment.all_assigned_completed for assignment in assignments.values()):
        return None

    makespan = max(a.total_time_s for a in assignments.values()) if assignments else float("inf")
    total_energy = sum(a.total_energy_j for a in assignments.values())
    times = [a.total_time_s for a in assignments.values()]
    load_std = pstdev(times) if len(times) > 1 else 0.0

    if verbose:
        print(f"    k={cluster_result.n_clusters}, makespan={makespan/60:.2f}min, "
              f"energy={total_energy:.0f}J, std={load_std:.2f}s, feasible={makespan<=horizon}")

    return list(assignments.values()), cluster_result


def rolling_plan(
    data,
    drone_count: int,
    k_max: int = 10,
    verbose: bool = True,
) -> RollingResult:
    energy_limit = float(parameter_value(data.params, "effective_energy_limit_J"))
    hover_power = float(parameter_value(data.params, "hover_power_J_per_s"))
    horizon = float(parameter_value(data.params, "operating_horizon_s"))
    battery_swap_time = float(parameter_value(data.params, "battery_swap_time_s") or 0.0)
    capacity_j = energy_limit

    wnodes = nodes_from_dataframe(data.nodes)
    node_cluster_map: dict[int, int] = {}

    # Precompute F-score components for all nodes
    all_node_ids = [int(n.node_id) for n in wnodes]
    priorities = {n.node_id: n.priority_weight for n in wnodes}
    hovers = {n.node_id: n.base_hover_time_s for n in wnodes}
    manuals = {n.node_id: n.manual_service_time_s for n in wnodes}
    confirms = {n.node_id: n.direct_confirm_time_s for n in wnodes}
    max_p = max(priorities.values()) if priorities else 1.0
    max_h = max(hovers.values()) if hovers else 1.0
    max_m = max(manuals.values()) if manuals else 1.0

    def _norm(v: float, mx: float) -> float:
        return v / mx if mx > 0 else 0.0

    all_node_priority = {nid: float(priorities[nid]) for nid in all_node_ids}
    all_node_downstream = {nid: float(manuals[nid] + max(0.0, confirms[nid] - hovers[nid])) for nid in all_node_ids}
    all_node_construct = {
        nid: 1.00 * _norm(priorities[nid], max_p)
           + 0.90 * _norm(manuals[nid], max_m)
           + 0.80 * _norm(max(0.0, confirms[nid] - hovers[nid]), max_h)
           + 0.25 * _norm(hovers[nid], max_h)
        for nid in all_node_ids
    }

    best_assignments: list[DroneAssignment] = []
    best_score = float("inf")
    best_cluster_result: ClusterResult | None = None
    best_makespan = float("inf")
    heuristics_tried = 0

    # Strategy A: variable cluster count — search from drone_count up to k_max
    # This ensures multi-drone collaboration (each drone gets at least one cluster)
    forced_k_start = max(1, drone_count)
    for k_attempt in range(forced_k_start, k_max + 1):
        result = _try_config(
            data, wnodes, drone_count, k_attempt, capacity_j, hover_power, horizon, battery_swap_time,
            node_cluster_map, all_node_priority, all_node_downstream, all_node_construct, verbose
        )
        if result is None:
            continue
        assignments, cluster_result = result
        heuristics_tried += 1
        makespan = max(a.total_time_s for a in assignments)
        total_energy = sum(a.total_energy_j for a in assignments)
        times = [a.total_time_s for a in assignments]
        load_std = pstdev(times) if len(times) > 1 else 0.0
        score = _compute_score(makespan, total_energy, load_std, horizon)
        if score < best_score:
            best_score = score
            best_assignments = assignments
            best_cluster_result = cluster_result
            best_makespan = makespan

    # Strategy B: k = drone_count (natural 1 drone per cluster)
    if drone_count >= 2:
        result2 = _try_config(
            data, wnodes, drone_count, drone_count, capacity_j, hover_power, horizon, battery_swap_time,
            node_cluster_map, all_node_priority, all_node_downstream, all_node_construct, verbose
        )
        if result2 is not None:
            assignments2, cluster_result2 = result2
            heuristics_tried += 1
            makespan2 = max(a.total_time_s for a in assignments2)
            total_energy2 = sum(a.total_energy_j for a in assignments2)
            times2 = [a.total_time_s for a in assignments2]
            load_std2 = pstdev(times2) if len(times2) > 1 else 0.0
            score2 = _compute_score(makespan2, total_energy2, load_std2, horizon)
            if score2 < best_score:
                best_score = score2
                best_assignments = assignments2
                best_cluster_result = cluster_result2
                best_makespan = makespan2

    # Strategy C: k = 2 (minimal clustering baseline)
    if drone_count > 2:
        result3 = _try_config(
            data, wnodes, drone_count, 2, capacity_j, hover_power, horizon, battery_swap_time,
            node_cluster_map, all_node_priority, all_node_downstream, all_node_construct, verbose
        )
        if result3 is not None:
            assignments3, cluster_result3 = result3
            heuristics_tried += 1
            makespan3 = max(a.total_time_s for a in assignments3)
            total_energy3 = sum(a.total_energy_j for a in assignments3)
            times3 = [a.total_time_s for a in assignments3]
            load_std3 = pstdev(times3) if len(times3) > 1 else 0.0
            score3 = _compute_score(makespan3, total_energy3, load_std3, horizon)
            if score3 < best_score:
                best_assignments = assignments3
                best_cluster_result = cluster_result3
                best_makespan = makespan3

    if not best_assignments or best_cluster_result is None:
        raise RuntimeError(f"Rolling plan failed for K={drone_count}")

    total_energy = sum(a.total_energy_j for a in best_assignments)
    total_hover = sum(a.total_hover_s for a in best_assignments)
    times = [a.total_time_s for a in best_assignments]
    load_std = pstdev(times) if len(times) > 1 else 0.0

    detail_rows = _build_detail_rows(best_assignments, data, node_cluster_map)
    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()
    summary_df = pd.DataFrame([{
        "method": "hierarchical_rolling",
        "makespan_s": best_makespan,
        "makespan_min": best_makespan / 60.0,
        "total_energy_j": total_energy,
        "total_hover_s": total_hover,
        "load_std_s": load_std,
        "feasible": best_makespan <= horizon,
        "horizon_s": horizon,
        "n_clusters": best_cluster_result.n_clusters,
        "silhouette": round(best_cluster_result.silhouette_score, 4),
        "drone_count": drone_count,
        "n_drones": len(best_assignments),
        "n_nodes_total": sum(len(a.completed) for a in best_assignments),
        "route_count": sum(a.route_count for a in best_assignments),
        "heuristics_tried": heuristics_tried,
    }])
    cluster_df = build_cluster_summary(
        wnodes, best_cluster_result.labels,
        {did: [n for n in wnodes if int(n.node_id) in best_assignments[did].node_ids]
         for did in range(len(best_assignments))},
        best_cluster_result.distance_matrix,
    )

    return RollingResult(
        makespan_s=best_makespan,
        total_energy_j=total_energy,
        total_hover_s=total_hover,
        load_std_s=load_std,
        cluster_count=best_cluster_result.n_clusters,
        n_heuristics_tried=heuristics_tried,
        horizon_feasible=best_makespan <= horizon,
        detail_df=detail_df,
        summary_df=summary_df,
        cluster_df=cluster_df,
    )


def compare_methods(data, drone_count: int, verbose: bool = True) -> pd.DataFrame:
    from run_problem1_heuristic import solve_for_drone_count

    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem 1 Method Comparison  (K={drone_count})")
        print(f"{'='*60}")

    original_summary, _ = solve_for_drone_count(drone_count, use_f_guidance=True)
    original_row = original_summary.iloc[0]
    makespan_orig = float(original_row["makespan_s"])
    energy_orig = float(original_row["total_energy_j"])
    hover_orig = float(original_row["total_hover_time_s"])
    std_orig = float(original_row["load_std_s"])
    horizon_orig = float(original_row["horizon_s"])

    if verbose:
        print(f"\n[Original Greedy + F-guidance]")
        print(f"  Makespan: {makespan_orig/60:.2f} min | Energy: {energy_orig:.1f} J | Load std: {std_orig:.2f} s")

    rolling = rolling_plan(data, drone_count, verbose=verbose)

    if verbose:
        print(f"\n[Hierarchical Rolling]")
        print(f"  Makespan: {rolling.makespan_s/60:.2f} min | Energy: {rolling.total_energy_j:.1f} J | Load std: {rolling.load_std_s:.2f} s")
        print(f"  Clusters: {rolling.cluster_count} | Heuristics tried: {rolling.n_heuristics_tried}")

    comparison = pd.DataFrame([{
        "method": "original_greedy",
        "makespan_s": makespan_orig,
        "makespan_min": makespan_orig / 60.0,
        "total_energy_j": energy_orig,
        "total_hover_s": hover_orig,
        "load_std_s": std_orig,
        "drone_count": drone_count,
        "feasible": bool(original_row["feasible_within_horizon"]),
        "horizon_s": horizon_orig,
    }])
    comparison = pd.concat([comparison, rolling.summary_df], ignore_index=True)
    return comparison


def main() -> None:
    data = load_c_problem_data()
    k_max = int(parameter_value(data.params, "K_max"))

    all_comparisons: list[pd.DataFrame] = []
    for k in range(1, k_max + 1):
        comp = compare_methods(data, k, verbose=True)
        all_comparisons.append(comp)

    comparison_table = pd.concat(all_comparisons, ignore_index=True)
    comparison_path = OUTPUT_DIR / "c_problem_problem1_method_comparison.csv"
    comparison_table.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"\nComparison saved to: {comparison_path}")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
