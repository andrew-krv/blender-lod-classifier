"""Mesh renaming service for ordered LOD clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..fingerprints.fingerprint import Cluster
from .mesh_info import MeshInfo


@dataclass
class MeshRenamer:
    """Creates renamed mesh copies and organizes them into cluster collections."""

    template: str = "mesh_{cluster:03d}_vtx_{vertices}_lod_{lod}"
    root_collection_name: str = "LOD_Clusters"

    def duplicate_and_rename(
        self,
        context: Any,
        clusters: list[Cluster],
        mesh_by_name: dict[str, MeshInfo],
    ) -> dict[str, list[str]]:
        """Create renamed copies and return map from source object to created copy names."""

        import bpy

        renamed: dict[str, list[str]] = {}
        used_names = set(bpy.data.objects.keys())
        root_collection = self._ensure_child_collection(context.scene.collection, self.root_collection_name)

        for cluster in clusters:
            cluster_collection_name = f"cluster_{cluster.cluster_id:03d}"
            cluster_collection = self._ensure_child_collection(root_collection, cluster_collection_name)

            for lod_index, member_name in enumerate(cluster.ordered_lods):
                mesh = mesh_by_name.get(member_name)
                if mesh is None or mesh.object_ref is None:
                    continue

                base_name = self.template.format(
                    cluster=cluster.cluster_id,
                    vertices=mesh.vertex_count,
                    lod=lod_index,
                )
                new_name = self._make_unique(base_name, used_names)

                copy_object = self._duplicate_object(mesh.object_ref)
                copy_object.name = new_name
                if getattr(copy_object, "data", None) is not None:
                    copy_object.data.name = new_name

                self._place_in_collection(copy_object, cluster_collection)
                used_names.add(new_name)
                renamed.setdefault(mesh.object_ref.name, []).append(new_name)

        return renamed

    def _duplicate_object(self, source_object: Any) -> Any:
        """Duplicate object and mesh data so source geometry remains unchanged."""

        duplicate = source_object.copy()
        if getattr(source_object, "data", None) is not None:
            duplicate.data = source_object.data.copy()
        return duplicate

    def _ensure_child_collection(self, parent: Any, name: str) -> Any:
        """Return existing child collection by name or create a new one."""

        import bpy

        child = parent.children.get(name)
        if child is not None:
            return child

        child = bpy.data.collections.get(name)
        if child is None:
            child = bpy.data.collections.new(name)

        if child.name not in parent.children:
            parent.children.link(child)

        return child

    def _place_in_collection(self, obj: Any, target_collection: Any) -> None:
        """Ensure object is linked only to target collection."""

        if obj.name not in target_collection.objects:
            target_collection.objects.link(obj)

        for collection in tuple(obj.users_collection):
            if collection != target_collection and obj.name in collection.objects:
                collection.objects.unlink(obj)

    def _make_unique(self, base_name: str, used_names: set[str]) -> str:
        if base_name not in used_names:
            return base_name

        index = 1
        while True:
            candidate = f"{base_name}_{index:02d}"
            if candidate not in used_names:
                return candidate
            index += 1
