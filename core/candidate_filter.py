"""Candidate pair filtering to reduce expensive comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from .mesh_info import MeshInfo


@dataclass
class CandidateFilter:
    """Rule-based pre-filter removing unlikely LOD mesh pairs."""

    min_dimension_ratio: float = 0.45
    min_volume_ratio: float = 0.25
    min_vertex_ratio: float = 0.15

    def _ratio(self, a: float, b: float) -> float:
        if a <= 0.0 and b <= 0.0:
            return 1.0
        if a <= 0.0 or b <= 0.0:
            return 0.0
        return min(a, b) / max(a, b)

    def _passes(self, a: MeshInfo, b: MeshInfo) -> bool:
        dims_ok = all(
            self._ratio(x, y) >= self.min_dimension_ratio
            for x, y in zip(a.bounding_box_dimensions, b.bounding_box_dimensions)
        )
        if not dims_ok:
            return False

        volume_ok = self._ratio(a.estimated_volume, b.estimated_volume) >= self.min_volume_ratio
        if not volume_ok:
            return False

        vertices_ok = self._ratio(float(a.vertex_count), float(b.vertex_count)) >= self.min_vertex_ratio
        return vertices_ok

    def build_candidates(self, meshes: list[MeshInfo]) -> list[tuple[MeshInfo, MeshInfo]]:
        """Return candidate pairs likely to belong to the same LOD family."""

        candidates: list[tuple[MeshInfo, MeshInfo]] = []
        for idx, mesh_a in enumerate(meshes):
            for mesh_b in meshes[idx + 1 :]:
                if self._passes(mesh_a, mesh_b):
                    candidates.append((mesh_a, mesh_b))
        return candidates
