"""Duplicate mesh detection so identical geometry is not treated as separate LODs."""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.geometry import ratio_similarity
from .clustering import UnionFind
from .mesh_info import MeshInfo


@dataclass
class DuplicateDetector:
    """Groups meshes with identical geometry and picks one representative per group.

    Exact duplicates share a geometry hash (vertex order, triangle order, seam splits and
    translation do not matter). Near duplicates (float noise, re-triangulated copies) are
    detected afterwards from the already-computed pair details.
    """

    min_voxel_score: float = 0.97
    min_bbox_score: float = 0.995
    min_area_ratio: float = 0.999

    def group_exact(self, meshes: list[MeshInfo]) -> dict[str, list[str]]:
        """Return {representative: [duplicates]} for meshes sharing a geometry hash."""

        by_hash: dict[str, list[str]] = {}
        for mesh in meshes:
            if mesh.geometry_hash and mesh.geometry_hash != "empty":
                by_hash.setdefault(mesh.geometry_hash, []).append(mesh.mesh_name)

        groups: dict[str, list[str]] = {}
        for names in by_hash.values():
            if len(names) > 1:
                representative, *duplicates = sorted(names, key=lambda n: (len(n), n))
                groups[representative] = duplicates
        return groups

    def is_near_duplicate(self, mesh_a: MeshInfo, mesh_b: MeshInfo, details: dict[str, float]) -> bool:
        """True when two meshes are the same geometry up to numeric noise."""

        return (
            mesh_a.vertex_count == mesh_b.vertex_count
            and len(mesh_a.triangles) == len(mesh_b.triangles)
            and ratio_similarity(mesh_a.surface_area, mesh_b.surface_area) >= self.min_area_ratio
            and details.get("bbox", 0.0) >= self.min_bbox_score
            and details.get("voxel", 0.0) >= self.min_voxel_score
        )

    def merge_near(
        self,
        groups: dict[str, list[str]],
        mesh_by_name: dict[str, MeshInfo],
        details_map: dict[tuple[str, str], dict[str, float]],
    ) -> dict[str, list[str]]:
        """Fold near-duplicate pairs among representatives into the duplicate groups."""

        representatives = [name for name in mesh_by_name if name not in {d for ds in groups.values() for d in ds}]
        uf = UnionFind(representatives)
        for (name_a, name_b), details in details_map.items():
            if name_a not in uf.parent or name_b not in uf.parent:
                continue
            if self.is_near_duplicate(mesh_by_name[name_a], mesh_by_name[name_b], details):
                uf.union(name_a, name_b)

        components: dict[str, list[str]] = {}
        for name in representatives:
            components.setdefault(uf.find(name), []).append(name)

        merged: dict[str, list[str]] = {}
        for names in components.values():
            representative, *others = sorted(names, key=lambda n: (len(n), n))
            duplicates = list(groups.get(representative, []))
            for other in others:
                duplicates.append(other)
                duplicates.extend(groups.get(other, []))
            if duplicates:
                merged[representative] = sorted(duplicates)
        return merged
