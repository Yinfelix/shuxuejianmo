from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data


LEGACY_BASELINE_SUMMARY_PATH = Path("outputs/tables/c_problem_problem2_algorithm_compare.csv")
F_DEFAULT_LIGHT_PATH = Path("outputs/tables/c_problem_problem2_f_default_runs_light.csv")
F_GA_LIGHT_PATH = Path("outputs/tables/c_problem_problem2_f_guidance_ga_runs_light.csv")
F_PSO_LIGHT_PATH = Path("outputs/tables/c_problem_problem2_f_guidance_pso_runs_light.csv")
F_ACO_LIGHT_PATH = Path("outputs/tables/c_problem_problem2_f_guidance_aco_runs_light.csv")
JOINT_REORDER_COMPARE_PATH = Path("outputs/tables/c_problem_problem2_joint_reorder_f_compare.csv")
JOINT_REORDER_DETAIL_PATH = Path("data/processed/c_problem_problem2_joint_reorder_f_detail.csv")
TASKCOUNT_COMPARE_PATH = Path("outputs/tables/c_problem_problem2_taskcount_joint_compare.csv")
TASKCOUNT_DETAIL_PATH = Path("data/processed/c_problem_problem2_taskcount_joint_detail.csv")

PRIORITY_LOSS_WEIGHT = 0.25
MANUAL_SERVICE_WEIGHT = 0.20
TOTAL_TIME_WEIGHT = 0.55


def _parse_selected_nodes(raw_value: object) -> set[int]:
    if raw_value is None or pd.isna(raw_value):
        return set()
    text = str(raw_value).strip()
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def _detail_confirmed_nodes(detail: pd.DataFrame, drone_count: int, solution_type: str) -> set[int]:
    filtered = detail.loc[(detail["drone_count"] == drone_count) & (detail["solution_type"] == solution_type)].copy()
    if filtered.empty:
        return set()
    return set(filtered.loc[filtered["direct_confirmed"].astype(bool), "node_id"].astype(int).tolist())


def _build_scheme_records(nodes: pd.DataFrame) -> list[dict[str, object]]:
    legacy_summary = pd.read_csv(LEGACY_BASELINE_SUMMARY_PATH, encoding="utf-8-sig")
    f_default = pd.read_csv(F_DEFAULT_LIGHT_PATH, encoding="utf-8-sig")
    f_ga = pd.read_csv(F_GA_LIGHT_PATH, encoding="utf-8-sig")
    f_pso = pd.read_csv(F_PSO_LIGHT_PATH, encoding="utf-8-sig")
    f_aco = pd.read_csv(F_ACO_LIGHT_PATH, encoding="utf-8-sig")
    joint_reorder_compare = pd.read_csv(JOINT_REORDER_COMPARE_PATH, encoding="utf-8-sig")
    joint_reorder_detail = pd.read_csv(JOINT_REORDER_DETAIL_PATH, encoding="utf-8-sig")
    taskcount_compare = pd.read_csv(TASKCOUNT_COMPARE_PATH, encoding="utf-8-sig")
    taskcount_detail = pd.read_csv(TASKCOUNT_DETAIL_PATH, encoding="utf-8-sig")

    scheme_records: list[dict[str, object]] = []
    for drone_count in [1, 2, 3, 4]:
        baseline_row = legacy_summary.loc[
            (legacy_summary["drone_count"] == drone_count) & (legacy_summary["solution_type"] == "baseline")
        ].iloc[0]
        scheme_records.append(
            {
                "drone_count": drone_count,
                "scheme": "baseline",
                "total_closed_loop_time_s": float(baseline_row["total_closed_loop_time_s"]),
                "direct_confirmed_nodes": set(),
            }
        )

        default_row = f_default.loc[f_default["drone_count"] == drone_count].iloc[0]
        scheme_records.append(
            {
                "drone_count": drone_count,
                "scheme": "f_default",
                "total_closed_loop_time_s": float(default_row["total_closed_loop_time_s"]),
                "direct_confirmed_nodes": _parse_selected_nodes(default_row.get("selected_direct_confirm_nodes", "")),
            }
        )

        ga_row = f_ga.loc[f_ga["drone_count"] == drone_count].iloc[0]
        scheme_records.append(
            {
                "drone_count": drone_count,
                "scheme": "f_ga_light",
                "total_closed_loop_time_s": float(ga_row["total_closed_loop_time_s"]),
                "direct_confirmed_nodes": _parse_selected_nodes(ga_row.get("selected_direct_confirm_nodes", "")),
            }
        )

        pso_row = f_pso.loc[f_pso["drone_count"] == drone_count].iloc[0]
        scheme_records.append(
            {
                "drone_count": drone_count,
                "scheme": "f_pso_light",
                "total_closed_loop_time_s": float(pso_row["total_closed_loop_time_s"]),
                "direct_confirmed_nodes": _parse_selected_nodes(pso_row.get("selected_direct_confirm_nodes", "")),
            }
        )

        aco_row = f_aco.loc[f_aco["drone_count"] == drone_count].iloc[0]
        scheme_records.append(
            {
                "drone_count": drone_count,
                "scheme": "f_aco_light",
                "total_closed_loop_time_s": float(aco_row["total_closed_loop_time_s"]),
                "direct_confirmed_nodes": _parse_selected_nodes(aco_row.get("selected_direct_confirm_nodes", "")),
            }
        )

        joint_row = joint_reorder_compare.loc[joint_reorder_compare["drone_count"] == drone_count].iloc[0]
        scheme_records.append(
            {
                "drone_count": drone_count,
                "scheme": "joint_reorder_f",
                "total_closed_loop_time_s": float(joint_row["joint_reorder_f_s"]),
                "direct_confirmed_nodes": _detail_confirmed_nodes(joint_reorder_detail, drone_count, "joint_reorder_f"),
            }
        )

        taskcount_row = taskcount_compare.loc[taskcount_compare["drone_count"] == drone_count].iloc[0]
        scheme_records.append(
            {
                "drone_count": drone_count,
                "scheme": "taskcount_joint",
                "total_closed_loop_time_s": float(taskcount_row["optimized_total_closed_loop_s"]),
                "direct_confirmed_nodes": _detail_confirmed_nodes(taskcount_detail, drone_count, "taskcount_joint"),
            }
        )

    node_meta = nodes[["node_id", "priority_weight", "manual_service_time_s"]].copy()
    node_meta["node_id"] = node_meta["node_id"].astype(int)
    node_meta["priority_weight"] = pd.to_numeric(node_meta["priority_weight"], errors="coerce").fillna(0.0)
    node_meta["manual_service_time_s"] = pd.to_numeric(node_meta["manual_service_time_s"], errors="coerce").fillna(0.0)
    total_priority = float(node_meta["priority_weight"].sum())
    total_manual_service = float(node_meta["manual_service_time_s"].sum())
    total_weighted_manual = float((node_meta["priority_weight"] * node_meta["manual_service_time_s"]).sum())
    high_priority_mask = node_meta["priority_weight"] >= 3
    high_priority_count = int(high_priority_mask.sum())

    enriched_records: list[dict[str, object]] = []
    baseline_by_k = {
        int(record["drone_count"]): float(record["total_closed_loop_time_s"])
        for record in scheme_records
        if str(record["scheme"]) == "baseline"
    }

    for record in scheme_records:
        confirmed_nodes = set(record["direct_confirmed_nodes"])
        confirmed_mask = node_meta["node_id"].isin(confirmed_nodes)
        direct_confirm_count = int(confirmed_mask.sum())
        direct_confirm_ratio = direct_confirm_count / len(node_meta) if len(node_meta) else 0.0
        priority_direct = float(node_meta.loc[confirmed_mask, "priority_weight"].sum())
        priority_direct_rate = priority_direct / total_priority if total_priority > 0 else 0.0
        priority_loss_ratio = 1.0 - priority_direct_rate
        manual_service_remaining = float(node_meta.loc[~confirmed_mask, "manual_service_time_s"].sum())
        manual_service_ratio = manual_service_remaining / total_manual_service if total_manual_service > 0 else 0.0
        weighted_manual_remaining = float(
            (node_meta.loc[~confirmed_mask, "priority_weight"] * node_meta.loc[~confirmed_mask, "manual_service_time_s"]).sum()
        )
        weighted_manual_ratio = weighted_manual_remaining / total_weighted_manual if total_weighted_manual > 0 else 0.0
        high_priority_direct_count = int((confirmed_mask & high_priority_mask).sum())
        high_priority_direct_rate = high_priority_direct_count / high_priority_count if high_priority_count > 0 else 0.0
        normalized_time_ratio = float(record["total_closed_loop_time_s"]) / baseline_by_k[int(record["drone_count"])]
        expanded_score = (
            TOTAL_TIME_WEIGHT * normalized_time_ratio
            + PRIORITY_LOSS_WEIGHT * priority_loss_ratio
            + MANUAL_SERVICE_WEIGHT * manual_service_ratio
        )
        enriched_records.append(
            {
                "drone_count": int(record["drone_count"]),
                "scheme": str(record["scheme"]),
                "legacy_total_closed_loop_time_s": float(record["total_closed_loop_time_s"]),
                "legacy_direct_confirm_count": direct_confirm_count,
                "legacy_direct_confirm_ratio": direct_confirm_ratio,
                "priority_direct_rate": priority_direct_rate,
                "priority_loss_ratio": priority_loss_ratio,
                "manual_service_remaining_s": manual_service_remaining,
                "manual_service_ratio": manual_service_ratio,
                "weighted_manual_burden_s": weighted_manual_remaining,
                "weighted_manual_burden_ratio": weighted_manual_ratio,
                "high_priority_direct_rate": high_priority_direct_rate,
                "expanded_eval_score": expanded_score,
                "confirmed_nodes": ",".join(str(node_id) for node_id in sorted(confirmed_nodes)),
            }
        )

    return enriched_records


def main() -> None:
    data = load_c_problem_data(WORKBOOK_PATH)
    records = _build_scheme_records(data.nodes)
    compare_df = pd.DataFrame(records)
    compare_df["legacy_rank"] = compare_df.groupby("drone_count")["legacy_total_closed_loop_time_s"].rank(method="dense")
    compare_df["expanded_rank"] = compare_df.groupby("drone_count")["expanded_eval_score"].rank(method="dense")

    representative_df = compare_df.loc[
        compare_df["scheme"].isin(["baseline", "f_default", "f_aco_light", "joint_reorder_f", "taskcount_joint"])
    ].copy()

    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_path = output_dir / "c_problem_problem2_evaluation_compare.csv"
    representative_path = output_dir / "c_problem_problem2_evaluation_representative.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    representative_df.to_csv(representative_path, index=False, encoding="utf-8-sig")

    print("Problem 2 evaluation comparison saved to:", compare_path)
    print(compare_df.to_string(index=False))
    print("\nRepresentative evaluation comparison saved to:", representative_path)
    print(representative_df.to_string(index=False))


if __name__ == "__main__":
    main()