"""Creates tagged LOD copies and the scene operations that work on them."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ..fingerprints.fingerprint import Cluster
from .mesh_info import MeshInfo
from .naming import (
    TAG_AUTO_BASE,
    TAG_FAMILY,
    TAG_HASH,
    TAG_LOD,
    PreviousCopy,
    format_family_name,
    inherited_family_name,
    lod_name,
    split_lod_name,
)

TAG_LEVEL_COLLECTION = "lod_classifier_level"


def is_sorted_copy(obj: Any) -> bool:
    """True for objects created by the sorter (identified by tag, not by name)."""

    return TAG_FAMILY in obj.keys()


def sorted_copies() -> list[Any]:
    import bpy

    return [obj for obj in bpy.data.objects if is_sorted_copy(obj)]


def copy_lod_index(obj: Any) -> int:
    """LOD index from the object's name suffix, falling back to its tag."""

    _, lod = split_lod_name(obj.name)
    return lod if lod is not None else int(obj.get(TAG_LOD, 0))


@dataclass
class MeshRenamer:
    """Creates renamed, tagged mesh copies grouped into one collection per LOD family.

    Copies from a previous run are replaced. Families inherit any base name the user gave
    to one of their members' previous copies (matched by geometry hash, so renames are fine).
    """

    template: str = "mesh_{cluster:03d}"
    root_collection_name: str = "LOD_Clusters"
    levels_collection_name: str = "LOD_Levels"

    def duplicate_and_rename(
        self,
        context: Any,
        clusters: list[Cluster],
        mesh_by_name: dict[str, MeshInfo],
    ) -> dict[str, list[str]]:
        """Replace previous copies with new ones and return {source object: [copy names]}."""

        import bpy

        previous_by_hash = self._collect_previous()
        self.delete_copies()

        renamed: dict[str, list[str]] = {}
        used_bases: set[str] = set()
        root_collection = self._ensure_child_collection(context.scene.collection, self.root_collection_name)

        for cluster in clusters:
            members = [mesh_by_name[name] for name in cluster.ordered_lods if name in mesh_by_name]
            members = [mesh for mesh in members if mesh.object_ref is not None]
            if not members:
                continue

            auto_base = self._unique(
                format_family_name(self.template, cluster.cluster_id, members[0].vertex_count),
                used_bases,
            )
            previous = [copy for mesh in members for copy in previous_by_hash.get(mesh.geometry_hash, [])]
            inherited = inherited_family_name(previous)
            base = self._unique(inherited, used_bases) if inherited else auto_base
            used_bases.update((base, auto_base))

            family_id = uuid.uuid4().hex
            cluster_collection = bpy.data.collections.new(base)
            root_collection.children.link(cluster_collection)

            for lod_index, mesh in enumerate(members):
                copy_object = self._duplicate_object(mesh.object_ref)
                copy_object.name = lod_name(base, lod_index)
                if getattr(copy_object, "data", None) is not None:
                    copy_object.data.name = copy_object.name

                copy_object[TAG_FAMILY] = family_id
                copy_object[TAG_LOD] = lod_index
                copy_object[TAG_HASH] = mesh.geometry_hash
                copy_object[TAG_AUTO_BASE] = auto_base

                self._place_in_collection(copy_object, cluster_collection)
                renamed.setdefault(mesh.object_ref.name, []).append(copy_object.name)

        return renamed

    def delete_copies(self) -> int:
        """Delete every tagged copy (and its now unused mesh data) plus emptied output collections."""

        import bpy

        copies = sorted_copies()
        meshes = {obj.data for obj in copies if getattr(obj, "data", None) is not None}
        for obj in copies:
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        for name in (self.root_collection_name, self.levels_collection_name):
            root = bpy.data.collections.get(name)
            if root is not None:
                self._remove_empty_children(root)
        return len(copies)

    def propagate_family_name(self, source: Any) -> list[str]:
        """Rename every copy in ``source``'s family to source's base name, keeping LOD suffixes."""

        import bpy

        if not is_sorted_copy(source):
            raise ValueError(f"'{source.name}' is not a sorted LOD copy")

        base, _ = split_lod_name(source.name)
        family = [obj for obj in sorted_copies() if obj.get(TAG_FAMILY) == source.get(TAG_FAMILY)]
        new_names = [lod_name(base, copy_lod_index(obj)) for obj in family]

        # Two passes so members swapping names do not collide into ".001" suffixes.
        for obj in family:
            obj.name = f"__lod_classifier_tmp_{uuid.uuid4().hex}"
        for obj, new_name in zip(family, new_names):
            obj.name = new_name
            if getattr(obj, "data", None) is not None and obj.data.users == 1:
                obj.data.name = obj.name

        # Keep the family collection (if still under LOD_Clusters) named after the family.
        root = bpy.data.collections.get(self.root_collection_name)
        if root is not None:
            for collection in {c for obj in family for c in obj.users_collection}:
                if collection.name in root.children:
                    collection.name = base
        return [obj.name for obj in family]

    def build_level_collections(self, context: Any) -> dict[int, int]:
        """Move every tagged copy into LOD0..LODN collections by its name suffix."""

        import bpy

        levels_root = self._ensure_child_collection(context.scene.collection, self.levels_collection_name)
        level_collections = {
            int(child[TAG_LEVEL_COLLECTION]): child for child in levels_root.children if TAG_LEVEL_COLLECTION in child.keys()
        }

        counts: dict[int, int] = {}
        for obj in sorted_copies():
            lod = copy_lod_index(obj)
            target = level_collections.get(lod)
            if target is None:
                target = bpy.data.collections.new(f"LOD{lod}")
                target[TAG_LEVEL_COLLECTION] = lod
                levels_root.children.link(target)
                level_collections[lod] = target
            self._place_in_collection(obj, target)
            counts[lod] = counts.get(lod, 0) + 1

        root = bpy.data.collections.get(self.root_collection_name)
        if root is not None:
            self._remove_empty_children(root)
        return dict(sorted(counts.items()))

    def _collect_previous(self) -> dict[str, list[PreviousCopy]]:
        previous: dict[str, list[PreviousCopy]] = {}
        for obj in sorted_copies():
            base, _ = split_lod_name(obj.name)
            copy = PreviousCopy(base=base, auto_base=str(obj.get(TAG_AUTO_BASE, base)), lod=copy_lod_index(obj))
            previous.setdefault(str(obj.get(TAG_HASH, "")), []).append(copy)
        return previous

    def _remove_empty_children(self, collection: Any) -> None:
        import bpy

        for child in list(collection.children):
            self._remove_empty_children(child)
            if not child.objects and not child.children:
                bpy.data.collections.remove(child)

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

    def _unique(self, base_name: str, used_names: set[str]) -> str:
        if base_name not in used_names:
            return base_name

        index = 2
        while True:
            candidate = f"{base_name}_{index:02d}"
            if candidate not in used_names:
                return candidate
            index += 1
