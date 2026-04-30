from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


@dataclass
class WeightedNode:
    node_id: int
    x_m: float
    y_m: float
    z_m: float
    priority_weight: float
    base_hover_time_s: float
    direct_confirm_time_s: float
    manual_service_time_s: float


@dataclass
class ClusterResult:
    labels: list[int]
    n_clusters: int
    silhouette_score: float
    distance_matrix: pd.DataFrame


def _feature_frame(nodes: list[WeightedNode]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "node_id": [int(node.node_id) for node in nodes],
            "x_m": [float(node.x_m) for node in nodes],
            "y_m": [float(node.y_m) for node in nodes],
            "z_m": [float(node.z_m) for node in nodes],
            "priority_weight": [float(node.priority_weight) for node in nodes],
            "base_hover_time_s": [float(node.base_hover_time_s) for node in nodes],
            "manual_service_time_s": [float(node.manual_service_time_s) for node in nodes],
            "direct_confirm_time_s": [float(node.direct_confirm_time_s) for node in nodes],
        }
    )
    feature_columns = [
        "x_m",
        "y_m",
        "z_m",
        "priority_weight",
        "base_hover_time_s",
        "manual_service_time_s",
        "direct_confirm_time_s",
    ]
    for column in feature_columns:
        series = frame[column]
        span = float(series.max() - series.min())
        if span > 0:
            frame[column] = (series - float(series.min())) / span
        else:
            frame[column] = 0.0

    frame["priority_weight"] *= 1.6
    frame["manual_service_time_s"] *= 1.2
    frame["direct_confirm_time_s"] *= 1.1
    frame["base_hover_time_s"] *= 0.8
    return frame


def _distance_matrix(nodes: list[WeightedNode]) -> pd.DataFrame:
    features = _feature_frame(nodes)
    values = features.drop(columns=["node_id"]).to_numpy(dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    distances = np.sqrt((diff * diff).sum(axis=2))
    node_ids = features["node_id"].astype(int).tolist()
    return pd.DataFrame(distances, index=node_ids, columns=node_ids)


def nodes_from_dataframe(nodes_df: pd.DataFrame) -> list[WeightedNode]:
    filtered = nodes_df.copy()
    filtered = filtered.dropna(subset=["node_id", "x_m", "y_m", "z_m"])
    filtered = filtered.loc[filtered["node_id"].astype(int) != 0]
    return [
        WeightedNode(
            node_id=int(row.node_id),
            x_m=float(row.x_m),
            y_m=float(row.y_m),
            z_m=float(row.z_m),
            priority_weight=float(row.priority_weight),
            base_hover_time_s=float(row.base_hover_time_s),
            direct_confirm_time_s=float(row.direct_confirm_time_s),
            manual_service_time_s=float(row.manual_service_time_s),
        )
        for row in filtered.itertuples(index=False)
    ]


def _cluster(nodes: list[WeightedNode], n_clusters: int) -> ClusterResult:
    if not nodes:
        return ClusterResult(labels=[], n_clusters=0, silhouette_score=0.0, distance_matrix=pd.DataFrame())

    bounded_k = max(1, min(n_clusters, len(nodes)))
    features = _feature_frame(nodes)
    model = AgglomerativeClustering(n_clusters=bounded_k, linkage="ward")
    label_array = model.fit_predict(features.drop(columns=["node_id"]))
    if bounded_k <= 1 or bounded_k >= len(nodes):
        score = 0.0
    else:
        score = float(silhouette_score(features.drop(columns=["node_id"]), label_array))
    return ClusterResult(
        labels=[int(label) for label in label_array.tolist()],
        n_clusters=bounded_k,
        silhouette_score=score,
        distance_matrix=_distance_matrix(nodes),
    )


def cluster_nodes(nodes: list[WeightedNode], max_k: int = 10) -> ClusterResult:
    if not nodes:
        return ClusterResult(labels=[], n_clusters=0, silhouette_score=0.0, distance_matrix=pd.DataFrame())
    upper = max(1, min(max_k, len(nodes)))
    best_result: ClusterResult | None = None
    for k in range(2, upper + 1):
        result = _cluster(nodes, k)
        if best_result is None or result.silhouette_score > best_result.silhouette_score:
            best_result = result
    return best_result or _cluster(nodes, 1)


def cluster_nodes_fixed_k(nodes: list[WeightedNode], k: int) -> ClusterResult:
    return _cluster(nodes, k)


def assign_drones_to_clusters(labels: list[int], drone_count: int, nodes: list[WeightedNode]) -> dict[int, list[int]]:
    cluster_map: dict[int, list[int]] = {}
    for node, label in zip(nodes, labels, strict=True):
        cluster_map.setdefault(int(label), []).append(int(node.node_id))

    cluster_items = sorted(
        cluster_map.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    assignments: dict[int, list[int]] = {drone_id: [] for drone_id in range(drone_count)}
    drone_loads = {drone_id: 0 for drone_id in range(drone_count)}
    for _, node_ids in cluster_items:
        target_drone = min(drone_loads, key=lambda drone_id: (drone_loads[drone_id], drone_id))
        assignments[target_drone].extend(node_ids)
        drone_loads[target_drone] += len(node_ids)
    return assignments


def build_cluster_summary(
    nodes: list[WeightedNode],
    labels: list[int],
    drone_nodes: dict[int, list[WeightedNode]],
    distance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    node_to_label = {int(node.node_id): int(label) for node, label in zip(nodes, labels, strict=True)}
    label_groups: dict[int, list[int]] = {}
    for node_id, label in node_to_label.items():
        label_groups.setdefault(label, []).append(node_id)

    for label, node_ids in sorted(label_groups.items()):
        submatrix = distance_matrix.loc[node_ids, node_ids] if not distance_matrix.empty else pd.DataFrame()
        mean_inner_distance = float(submatrix.to_numpy().mean()) if not submatrix.empty else 0.0
        assigned_drone = -1
        for drone_id, members in drone_nodes.items():
            member_ids = {int(node.node_id) for node in members}
            if member_ids.intersection(node_ids):
                assigned_drone = int(drone_id)
                break
        rows.append(
            {
                "cluster_id": int(label),
                "assigned_drone": assigned_drone,
                "node_count": len(node_ids),
                "priority_sum": float(sum(node.priority_weight for node in nodes if int(node.node_id) in node_ids)),
                "mean_inner_distance": mean_inner_distance,
            }
        )
    return pd.DataFrame(rows)