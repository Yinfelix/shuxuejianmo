from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value


ROUTE_SUMMARY_PATH = Path("outputs/tables/c_problem_problem1_route_summary.csv")
OUTPUT_PATH = Path("outputs/tables/c_problem_problem1_cross_k_merge_heuristic.csv")


def _pairings(route_ids: list[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(route_ids) != 4:
        raise ValueError("Expected exactly four routes for K=4 to K=2 pairing experiment.")
    pairings: list[tuple[tuple[int, int], tuple[int, int]]] = []
    anchor = route_ids[0]
    remaining = route_ids[1:]
    for partner in remaining:
        first_pair = tuple(sorted((anchor, partner)))
        leftover = [route_id for route_id in route_ids if route_id not in first_pair]
        second_pair = tuple(sorted((leftover[0], leftover[1])))
        normalized = tuple(sorted((first_pair, second_pair)))
        if normalized not in pairings:
            pairings.append(normalized)
    return pairings


def _route_duration_map(route_summary: pd.DataFrame, drone_count: int) -> dict[int, float]:
    rows = route_summary.loc[route_summary["drone_count"] == drone_count, ["route_id", "route_duration_s"]].copy()
    return {int(row.route_id): float(row.route_duration_s) for row in rows.itertuples(index=False)}


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    battery_swap_time = float(parameter_value(data.params, "battery_swap_time_s"))
    route_summary = pd.read_csv(ROUTE_SUMMARY_PATH)

    k4_routes = _route_duration_map(route_summary, drone_count=4)
    k2_routes = _route_duration_map(route_summary, drone_count=2)
    k2_baseline = float(route_summary.loc[route_summary["drone_count"] == 2, "drone_total_time_s"].max())
    k1_baseline = float(route_summary.loc[route_summary["drone_count"] == 1, "drone_total_time_s"].max())

    rows: list[dict[str, float | int | str]] = []
    for pairing_index, pairing in enumerate(_pairings(sorted(k4_routes)), start=1):
        pair_a, pair_b = pairing
        pair_a_time = sum(k4_routes[route_id] for route_id in pair_a) + battery_swap_time
        pair_b_time = sum(k4_routes[route_id] for route_id in pair_b) + battery_swap_time
        makespan = max(pair_a_time, pair_b_time)
        rows.append(
            {
                "experiment": "k4_to_k2_pairing",
                "candidate_id": pairing_index,
                "assignment": f"({pair_a[0]},{pair_a[1]}) | ({pair_b[0]},{pair_b[1]})",
                "drone_a_total_time_s": pair_a_time,
                "drone_b_total_time_s": pair_b_time,
                "candidate_makespan_s": makespan,
                "baseline_makespan_s": k2_baseline,
                "gain_vs_baseline_s": k2_baseline - makespan,
            }
        )

    merged_k1_time = sum(k2_routes.values()) + battery_swap_time
    rows.append(
        {
            "experiment": "k2_to_k1_merge",
            "candidate_id": 1,
            "assignment": "(1,2)",
            "drone_a_total_time_s": merged_k1_time,
            "drone_b_total_time_s": 0.0,
            "candidate_makespan_s": merged_k1_time,
            "baseline_makespan_s": k1_baseline,
            "gain_vs_baseline_s": k1_baseline - merged_k1_time,
        }
    )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Problem 1 cross-K merge heuristic saved to:", OUTPUT_PATH)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()