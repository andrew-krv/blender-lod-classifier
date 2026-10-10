"""LOD Classifier UI panel."""

from __future__ import annotations

import json

import bpy
from bpy.types import Panel


def _summary(context) -> dict[str, int]:
    raw = context.scene.get("lod_classifier_summary", "")
    if not raw:
        return {}

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class VIEW3D_PT_LODClassifier(Panel):
    """Sidebar panel for running LOD classification tasks."""

    bl_label = "LOD Classifier"
    bl_idname = "VIEW3D_PT_lod_classifier"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LOD Classifier"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.lod_classifier_settings

        col = layout.column(align=True)
        col.label(text="Pipeline")
        col.prop(settings, "source_collection")
        col.prop(settings, "cluster_threshold")
        col.prop(settings, "include_singletons")
        col.prop(settings, "collapse_duplicates")
        col.prop(settings, "linkage")

        col.separator()
        col.label(text="Candidate Filter")
        col.prop(settings, "min_dimension_ratio")
        col.prop(settings, "min_volume_ratio")
        col.prop(settings, "min_vertex_ratio")

        col.separator()
        col.label(text="Performance")
        col.prop(settings, "voxel_resolution")
        col.prop(settings, "fast_prefilter")
        col.prop(settings, "incremental_mode")
        col.prop(settings, "tick_budget")

        col.separator()
        col.label(text="Output Copies")
        col.prop(settings, "rename_meshes", text="Create Renamed Copies (Keep Originals)")
        col.prop(settings, "rename_template")
        col.label(text="Originals are never renamed; only unique meshes are copied")
        col.label(text="Re-running replaces copies and keeps names you gave them")

        row = layout.row(align=True)
        row.operator("lod_classifier.classify_scene", icon="VIEWZOOM")
        row.operator("lod_classifier.cancel", icon="CANCEL")

        progress_box = layout.box()
        progress_box.label(text="Progress")
        progress_box.prop(settings, "progress_percent", slider=True)
        progress_box.label(text=settings.progress_message)

        box = layout.box()
        box.label(text="Sorted Copies")
        box.operator("lod_classifier.propagate_name", icon="SORTALPHA")
        box.operator("lod_classifier.build_lod_collections", icon="OUTLINER_COLLECTION")
        box.operator("lod_classifier.delete_copies", icon="TRASH")

        row = layout.row(align=True)
        row.operator("lod_classifier.export_json", icon="EXPORT")
        row.operator("lod_classifier.export_csv", icon="EXPORT")

        layout.operator("lod_classifier.show_runtime_info", icon="INFO")

        layout.operator("lod_classifier.clear_report", icon="TRASH")

        summary = _summary(context)
        if summary:
            box = layout.box()
            box.label(text="Last Run")
            box.label(text=f"Meshes: {summary.get('mesh_count', 0)}")
            box.label(text=f"Candidates: {summary.get('candidate_count', 0)}")
            box.label(text=f"Comparisons: {summary.get('comparison_count', 0)}")
            box.label(text=f"Compared: {summary.get('compared_pairs', 0)}")
            box.label(text=f"Reused: {summary.get('reused_pairs', 0)}")
            box.label(text=f"Clusters: {summary.get('cluster_count', 0)}")
            box.label(text=f"Duplicates: {summary.get('duplicate_count', 0)}")
            box.label(text=f"Copies: {summary.get('copy_count', 0)}")


classes = (VIEW3D_PT_LODClassifier,)
