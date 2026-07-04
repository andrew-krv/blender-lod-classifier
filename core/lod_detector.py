"""LOD ordering strategies."""

from __future__ import annotations

from dataclasses import dataclass

from ..fingerprints.fingerprint import Cluster
from .mesh_info import MeshInfo


@dataclass
class VertexCountLODDetector:
    """Default LOD strategy ordering by descending vertex count."""

    def order_cluster(self, cluster: Cluster, mesh_by_name: dict[str, MeshInfo]) -> list[str]:
        """Return ordered member list where first element is likely LOD0."""

        return sorted(
            cluster.members,
            key=lambda name: mesh_by_name.get(name).vertex_count if mesh_by_name.get(name) else 0,
            reverse=True,
        )
