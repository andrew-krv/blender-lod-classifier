"""Mesh database builder with simple in-memory cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.cache import SimpleCache
from ..utils.geometry import extract_geometry_payload
from .mesh_info import MeshInfo
from .renamer import is_sorted_copy


@dataclass
class MeshDatabase:
    """Cached collection of MeshInfo objects for a scene scan."""

    cache: SimpleCache

    def build(self, context: Any, collection_name: str = "", exclude_collection_name: str = "") -> list[MeshInfo]:
        """Scan mesh objects and return MeshInfo records.

        With ``collection_name`` only objects in that collection (and its child collections)
        are scanned; otherwise the whole active scene, minus ``exclude_collection_name``
        (the addon's own output collection). Tagged copies made by earlier runs are always
        skipped, wherever the user moved them, so results are never re-classified.
        """

        key = f"scene:{context.scene.name}:{collection_name}:mesh_infos"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        mesh_infos: list[MeshInfo] = []
        for obj in self._source_objects(context, collection_name, exclude_collection_name):
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
                    geometry_hash=payload.geometry_hash,
                )
            )

        self.cache.set(key, mesh_infos)
        return mesh_infos

    def _source_objects(self, context: Any, collection_name: str, exclude_collection_name: str) -> list[Any]:
        import bpy

        if collection_name:
            collection = bpy.data.collections.get(collection_name)
            if collection is None:
                raise ValueError(f"Collection '{collection_name}' not found")
            return [obj for obj in collection.all_objects if not is_sorted_copy(obj)]

        excluded = bpy.data.collections.get(exclude_collection_name) if exclude_collection_name else None
        excluded_names = {obj.name for obj in excluded.all_objects} if excluded is not None else set()
        return [obj for obj in context.scene.objects if obj.name not in excluded_names and not is_sorted_copy(obj)]

    def invalidate(self, context: Any, collection_name: str = "") -> None:
        """Invalidate cached database for the provided scene / collection."""

        self.cache.delete(f"scene:{context.scene.name}:{collection_name}:mesh_infos")
