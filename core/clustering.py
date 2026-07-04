"""Similarity-graph clustering for probable LOD families."""

from __future__ import annotations

from dataclasses import dataclass

from ..fingerprints.fingerprint import Cluster
from .mesh_info import MeshInfo


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


@dataclass
class SimilarityClusterer:
    """Build clusters by connecting pairs above a similarity threshold."""

    def build_clusters(
        self,
        meshes: list[MeshInfo],
        pair_scores: dict[tuple[str, str], float],
        threshold: float,
        include_singletons: bool = False,
    ) -> list[Cluster]:
        """Cluster meshes into connected components of high-similarity pairs."""

        names = [m.mesh_name for m in meshes]
        uf = _UnionFind(names)

        for (name_a, name_b), score in pair_scores.items():
            if score >= threshold:
                uf.union(name_a, name_b)

        groups: dict[str, list[str]] = {}
        for name in names:
            root = uf.find(name)
            groups.setdefault(root, []).append(name)

        clusters: list[Cluster] = []
        cluster_id = 1
        for members in groups.values():
            if len(members) == 1 and not include_singletons:
                continue

            confidence = self._cluster_confidence(members, pair_scores)
            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    members=sorted(members),
                    confidence=confidence,
                    ordered_lods=[],
                )
            )
            cluster_id += 1

        return clusters

    def _cluster_confidence(self, members: list[str], pair_scores: dict[tuple[str, str], float]) -> float:
        if len(members) <= 1:
            return 1.0

        values: list[float] = []
        for idx, member_a in enumerate(members):
            for member_b in members[idx + 1 :]:
                key = tuple(sorted((member_a, member_b)))
                if key in pair_scores:
                    values.append(pair_scores[key])

        if not values:
            return 0.0
        return sum(values) / len(values)
