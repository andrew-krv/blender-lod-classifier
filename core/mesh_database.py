"""Mesh database builder with simple in-memory cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.cache import SimpleCache
from ..utils.geometry import extract_geometry_payload
from .mesh_info import MeshInfo


@dataclass
class MeshDatabase:
    """Cached collection of MeshInfo objects for a scene scan."""

    cache: SimpleCache

    def build(self, context: Any) -> list[MeshInfo]:
        """Scan all mesh objects in the active scene and return MeshInfo records."""

        key = f"scene:{context.scene.name}:mesh_infos"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        mesh_infos: list[MeshInfo] = []
        for obj in context.scene.objects:
            if obj.type != "MESH":
                continue

            payload = extract_geometry_payload(obj)
            mesh_infos.append(
                MeshInfo(
                    object_ref=obj,
                    mesh_name=obj.name,
                    vertices=payload.vertices,
                    triangles=payload.triangles,
                    vertex_count=payload.vertex_count,
                    face_count=payload.face_count,
                    edge_count=payload.edge_count,
                    bounding_box=payload.bounding_box,
                    bounding_box_dimensions=payload.bbox_dimensions,
                    bounding_box_center=payload.bbox_center,
                    surface_area=payload.surface_area,
                    estimated_volume=payload.estimated_volume,
                    materials=payload.materials,
                    transform_matrix=payload.transform_matrix,
                )
            )

        self.cache.set(key, mesh_infos)
        return mesh_infos

    def invalidate(self, context: Any) -> None:
        """Invalidate cached database for the provided scene."""

        self.cache.delete(f"scene:{context.scene.name}:mesh_infos")
