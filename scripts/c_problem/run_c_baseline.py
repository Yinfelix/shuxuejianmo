from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_c_data import WORKBOOK_PATH, load_c_problem_data, parameter_value


def build_baseline_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_c_problem_data(WORKBOOK_PATH)
    nodes = data.nodes.copy()
    targets = nodes.loc[nodes["node_id"] != 0].copy()

    energy_limit = parameter_value(data.params, "effective_energy_limit_J")
    hover_power = parameter_value(data.params, "hover_power_J_per_s")
    horizon = parameter_value(data.params, "operating_horizon_s")
    k_max = parameter_value(data.params, "K_max")
    battery_swap_time = parameter_value(data.params, "battery_swap_time_s")

    targets["base_hover_energy_J"] = targets["base_hover_time_s"] * hover_power
    targets["base_total_energy_J"] = (
        targets["direct_out_back_flight_energy_J"] + targets["base_hover_energy_J"]
    )
    targets["confirm_total_energy_J"] = targets["direct_confirm_roundtrip_total_energy_J"]
    targets["direct_confirmable_at_base"] = (
        targets["base_hover_time_s"] >= targets["direct_confirm_time_s"]
    )
    targets["base_energy_feasible"] = targets["base_total_energy_J"] <= energy_limit
    targets["confirm_energy_feasible"] = targets["confirm_total_energy_J"] <= energy_limit
    targets["hover_gap_s"] = targets["direct_confirm_time_s"] - targets["base_hover_time_s"]

    summary = pd.DataFrame(
        {
            "metric": [
                "target_count",
                "priority_sum",
                "avg_base_hover_time_s",
                "avg_direct_confirm_time_s",
                "base_direct_confirmable_count",
                "base_direct_confirmable_ratio",
                "base_energy_feasible_count",
                "confirm_energy_feasible_count",
                "operating_horizon_s",
                "effective_energy_limit_J",
                "hover_power_J_per_s",
                "battery_swap_time_s",
                "K_max",
            ],
            "value": [
                len(targets),
                float(targets["priority_weight"].sum()),
                float(targets["base_hover_time_s"].mean()),
                float(targets["direct_confirm_time_s"].mean()),
                int(targets["direct_confirmable_at_base"].sum()),
                float(targets["direct_confirmable_at_base"].mean()),
                int(targets["base_energy_feasible"].sum()),
                int(targets["confirm_energy_feasible"].sum()),
                horizon,
                energy_limit,
                hover_power,
                battery_swap_time,
                k_max,
            ],
        }
    )
    detail_columns = [
        "node_id",
        "node_name",
        "priority_level",
        "priority_weight",
        "base_hover_time_s",
        "direct_confirm_time_s",
        "hover_gap_s",
        "base_total_energy_J",
        "confirm_total_energy_J",
        "base_energy_feasible",
        "confirm_energy_feasible",
        "direct_confirmable_at_base",
    ]
    detail = targets[detail_columns].sort_values(
        by=["priority_weight", "hover_gap_s"], ascending=[False, True]
    )
    return summary, detail


def main() -> None:
    summary, detail = build_baseline_summary()
    output_dir = Path("outputs/tables")
    processed_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "c_problem_baseline_summary.csv"
    detail_path = processed_dir / "c_problem_node_detail.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Baseline summary saved to:", summary_path)
    print(summary.to_string(index=False))
    print("\nNode detail saved to:", detail_path)
    print(detail.head(10).to_string(index=False))


if __name__ == "__main__":
    main()