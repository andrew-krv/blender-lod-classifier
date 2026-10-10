"""Similarity-graph clustering for probable LOD families."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from ..fingerprints.fingerprint import Cluster
from .mesh_info import MeshInfo


class UnionFind:
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
    """Agglomerative clustering over the sparse pair-similarity graph.

    ``linkage="average"`` (default) merges two groups only when the mean similarity over
    *all* cross pairs reaches the threshold; pairs that were never compared (removed by the
    candidate filter) count as 0. This prevents the chaining effect of plain connected
    components where A~B and B~C pulls unrelated A and C together.
    ``linkage="single"`` reproduces the old connected-components behaviour.
    """

    linkage: str = "average"

    def build_clusters(
        self,
        meshes: list[MeshInfo],
        pair_scores: dict[tuple[str, str], float],
        threshold: float,
        include_singletons: bool = False,
    ) -> list[Cluster]:
        """Cluster meshes into groups of mutually similar meshes."""

        names = [m.mesh_name for m in meshes]
        if self.linkage == "single":
            groups = self._single_linkage(names, pair_scores, threshold)
        else:
            groups = self._average_linkage(names, pair_scores, threshold)

        clusters: list[Cluster] = []
        cluster_id = 1
        for members in sorted(groups, key=lambda g: sorted(g)[0]):
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

    def _single_linkage(
        self,
        names: list[str],
        pair_scores: dict[tuple[str, str], float],
        threshold: float,
    ) -> list[list[str]]:
        uf = UnionFind(names)
        for (name_a, name_b), score in pair_scores.items():
            if score >= threshold and name_a in uf.parent and name_b in uf.parent:
                uf.union(name_a, name_b)

        groups: dict[str, list[str]] = {}
        for name in names:
            groups.setdefault(uf.find(name), []).append(name)
        return list(groups.values())

    def _average_linkage(
        self,
        names: list[str],
        pair_scores: dict[tuple[str, str], float],
        threshold: float,
    ) -> list[list[str]]:
        index = {name: i for i, name in enumerate(names)}
        members: dict[int, list[str]] = {i: [name] for i, name in enumerate(names)}
        # links[c][d] = sum of pair scores between clusters c and d (missing pairs add 0).
        links: dict[int, dict[int, float]] = {i: {} for i in members}
        for (name_a, name_b), score in pair_scores.items():
            if name_a not in index or name_b not in index or score <= 0.0:
                continue
            a, b = index[name_a], index[name_b]
            links[a][b] = links[a].get(b, 0.0) + score
            links[b][a] = links[b].get(a, 0.0) + score

        def average(c: int, d: int) -> float:
            return links[c][d] / (len(members[c]) * len(members[d]))

        heap = [(-average(a, b), a, b) for a in links for b in links[a] if a < b]
        heapq.heapify(heap)
        next_id = len(names)

        while heap:
            neg_score, a, b = heapq.heappop(heap)
            if -neg_score < threshold:
                break
            if a not in members or b not in members:
                continue  # stale entry: one side was merged already

            merged = next_id
            next_id += 1
            members[merged] = members.pop(a) + members.pop(b)
            links_a = links.pop(a)
            links_b = links.pop(b)
            combined: dict[int, float] = {}
            for other, value in list(links_a.items()) + list(links_b.items()):
                if other in (a, b):
                    continue
                combined[other] = combined.get(other, 0.0) + value
            links[merged] = combined

            for other, value in combined.items():
                neighbour = links[other]
                neighbour.pop(a, None)
                neighbour.pop(b, None)
                neighbour[merged] = value
                heapq.heappush(heap, (-average(merged, other), min(merged, other), max(merged, other)))

        return list(members.values())

    def _cluster_confidence(self, members: list[str], pair_scores: dict[tuple[str, str], float]) -> float:
        if len(members) <= 1:
            return 1.0

        values: list[float] = []
        for idx, member_a in enumerate(members):
            for member_b in members[idx + 1 :]:
                key = tuple(sorted((member_a, member_b)))
                values.append(pair_scores.get(key, 0.0))

        if not values:
            return 0.0
        return sum(values) / len(values)
