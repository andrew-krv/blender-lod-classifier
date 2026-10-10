"""Blender operators for running classification and exporting reports."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, PropertyGroup, Scene

from ..core.pipeline import LODClassifierPipeline, PipelineSettings
from ..core.renamer import MeshRenamer, is_sorted_copy


def _scene_report_key() -> str:
    return "lod_classifier_report"


def _scene_summary_key() -> str:
    return "lod_classifier_summary"


class LODClassifierSettings(PropertyGroup):
    """User-configurable addon settings."""

    cluster_threshold: FloatProperty(name="Threshold", default=0.7, min=0.0, max=1.0)
    include_singletons: BoolProperty(
        name="Include Singletons",
        description="Also copy unique meshes that have no other LODs (as <name>_lod0)",
        default=True,
    )
    source_collection: PointerProperty(
        name="Collection",
        description="Only classify meshes in this collection (and its children). Empty = whole scene",
        type=bpy.types.Collection,
    )
    collapse_duplicates: BoolProperty(
        name="Collapse Duplicates",
        description="Treat meshes with identical geometry as one; only the representative is clustered and copied",
        default=True,
    )
    linkage: EnumProperty(
        name="Linkage",
        description="How groups are merged",
        items=(
            ("average", "Average", "Merge only when the mean similarity over all cross pairs passes the threshold"),
            ("single", "Single", "Merge on any single pair above the threshold (prone to chaining)"),
        ),
        default="average",
    )
    rename_meshes: BoolProperty(
        name="Create Renamed Copies",
        description="Create duplicated objects with LOD names. Original meshes are not renamed",
        default=True,
    )
    rename_template: StringProperty(
        name="Family Name Template",
        description="Base name for new families ({cluster}, {vertices} of LOD0). '_lod<N>' is appended. "
        "Families inherit names you gave their copies in earlier runs",
        default="mesh_{cluster:03d}",
    )
    min_dimension_ratio: FloatProperty(name="Min Dim Ratio", default=0.45, min=0.0, max=1.0)
    min_volume_ratio: FloatProperty(name="Min Volume Ratio", default=0.25, min=0.0, max=1.0)
    min_vertex_ratio: FloatProperty(name="Min Vertex Ratio", default=0.02, min=0.0, max=1.0)
    voxel_resolution: IntProperty(
        name="Voxel Resolution",
        description="Shape comparison grid size. Lower values are faster but less accurate",
        default=32,
        min=8,
        max=64,
    )
    fast_prefilter: BoolProperty(
        name="Fast Prefilter",
        description="Use bbox+topology to skip expensive comparisons",
        default=True,
    )
    tick_budget: IntProperty(
        name="Tick Budget",
        description="Operations per UI tick. Higher is faster, lower is more responsive",
        default=120,
        min=20,
        max=1200,
    )
    incremental_mode: BoolProperty(
        name="Incremental Mode",
        description="Reuse prior pair scores when meshes are unchanged",
        default=True,
    )
    cancel_requested: BoolProperty(name="Cancel Requested", default=False, options={"HIDDEN"})
    is_running: BoolProperty(name="Is Running", default=False, options={"HIDDEN"})
    progress_percent: IntProperty(name="Progress", default=0, min=0, max=100, subtype="PERCENTAGE")
    progress_message: StringProperty(name="Status", default="Idle")


class LODCLASSIFIER_OT_ClassifyScene(Operator):
    """Scan scene meshes, detect LOD groups, and optionally rename objects."""

    bl_idname = "lod_classifier.classify_scene"
    bl_label = "Scan / Classify"
    bl_options = {"REGISTER"}

    _timer = None
    _pipeline = None
    _session = None
    _last_message = ""

    def _start(self, context):
        settings = context.scene.lod_classifier_settings
        if settings.is_running:
            self.report({"WARNING"}, "Classification is already running")
            return {"CANCELLED"}

        settings.cancel_requested = False
        settings.is_running = True
        settings.progress_percent = 0
        settings.progress_message = "Starting"

        pipeline_settings = PipelineSettings(
            cluster_threshold=settings.cluster_threshold,
            include_singletons=settings.include_singletons,
            source_collection=settings.source_collection.name if settings.source_collection else "",
            rename_meshes=settings.rename_meshes,
            rename_template=settings.rename_template,
            collapse_duplicates=settings.collapse_duplicates,
            linkage=settings.linkage,
            min_dimension_ratio=settings.min_dimension_ratio,
            min_volume_ratio=settings.min_volume_ratio,
            min_vertex_ratio=settings.min_vertex_ratio,
            voxel_resolution=settings.voxel_resolution,
            fast_prefilter=settings.fast_prefilter,
            incremental_mode=settings.incremental_mode,
        )

        self._pipeline = LODClassifierPipeline(settings=pipeline_settings)
        try:
            self._session = self._pipeline.start(context)
        except ValueError as exc:
            settings.is_running = False
            settings.progress_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self._last_message = ""

        wm = context.window_manager
        wm.progress_begin(0, 9999)
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        self._set_status_text(context, "LOD Classifier: 0% - Starting")
        self._tag_redraw(context)
        self.report({"INFO"}, f"Scanned meshes: {len(self._session.mesh_infos)}")
        return {"RUNNING_MODAL"}

    def execute(self, context):
        return self._start(context)

    def invoke(self, context, event):
        return self._start(context)

    def modal(self, context, event):
        settings = context.scene.lod_classifier_settings
        wm = context.window_manager

        if event.type == "ESC":
            settings.cancel_requested = True

        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        step = self._pipeline.step(
            self._session,
            max_operations=settings.tick_budget,
            is_cancelled=lambda: bool(settings.cancel_requested),
        )

        wm.progress_update(step.value)
        settings.progress_percent = max(0, min(100, int((step.value / 9999) * 100)))
        settings.progress_message = step.message
        self._set_status_text(
            context,
            f"LOD Classifier: {settings.progress_percent}% - {step.message}",
        )
        self._tag_redraw(context)

        if step.message != self._last_message:
            self.report({"INFO"}, step.message)
            self._last_message = step.message

        if not step.done:
            return {"RUNNING_MODAL"}

        self._cleanup_modal(context)

        if step.cancelled:
            settings.progress_message = "Cancelled"
            settings.progress_percent = 0
            self._set_status_text(context, None)
            self.report({"WARNING"}, "LOD classification cancelled")
            return {"CANCELLED"}

        result = self._pipeline.finish(self._session)

        report_json = result.report.to_json()
        context.scene[_scene_report_key()] = report_json
        context.scene[_scene_summary_key()] = json.dumps(
            {
                "mesh_count": result.report.mesh_count,
                "candidate_count": result.report.candidate_count,
                "comparison_count": result.report.comparison_count,
                "compared_pairs": result.compared_pair_count,
                "reused_pairs": result.reused_pair_count,
                "cluster_count": len(result.clusters),
                "duplicate_count": sum(len(items) for items in result.report.duplicates.values()),
                "copy_count": sum(len(items) for items in result.renamed.values()),
            }
        )

        created_count = sum(len(items) for items in result.renamed.values())
        settings.progress_percent = 100
        settings.progress_message = "Done"
        self._set_status_text(context, "LOD Classifier: 100% - Done")
        self._tag_redraw(context)

        self.report(
            {"INFO"},
            f"LOD classification complete: {len(result.clusters)} clusters, {created_count} copies created",
        )
        return {"FINISHED"}

    def _cleanup_modal(self, context) -> None:
        """Release timer and reset execution flags."""

        settings = context.scene.lod_classifier_settings
        settings.is_running = False
        settings.cancel_requested = False

        wm = context.window_manager
        wm.progress_end()
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None

    def _set_status_text(self, context, text: str | None) -> None:
        """Update Blender status bar text safely."""

        workspace = getattr(context, "workspace", None)
        if workspace is not None:
            workspace.status_text_set(text)

    def _tag_redraw(self, context) -> None:
        """Force UI areas to redraw so progress is visibly updated."""

        wm = context.window_manager
        for window in wm.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type in {"VIEW_3D", "PROPERTIES", "TOPBAR", "OUTLINER"}:
                    area.tag_redraw()


class LODCLASSIFIER_OT_Cancel(Operator):
    """Request cancellation for the currently running classification."""

    bl_idname = "lod_classifier.cancel"
    bl_label = "Cancel"

    def execute(self, context):
        settings = context.scene.lod_classifier_settings
        settings.cancel_requested = True
        settings.progress_message = "Cancelling..."
        if settings.is_running:
            self.report({"INFO"}, "Cancel requested")
        else:
            self.report({"INFO"}, "No classification is currently running")
        return {"FINISHED"}


class LODCLASSIFIER_OT_ExportJson(Operator):
    """Export last report to a JSON file."""

    bl_idname = "lod_classifier.export_json"
    bl_label = "Export JSON"

    filepath: StringProperty(subtype="FILE_PATH", default="lod_classifier_report.json")

    def execute(self, context):
        report = context.scene.get(_scene_report_key(), "")
        if not report:
            self.report({"WARNING"}, "No report available. Run classification first.")
            return {"CANCELLED"}

        path = Path(self.filepath)
        path.write_text(report, encoding="utf-8")
        self.report({"INFO"}, f"Exported JSON report to {path}")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class LODCLASSIFIER_OT_ExportCsv(Operator):
    """Export last report to CSV."""

    bl_idname = "lod_classifier.export_csv"
    bl_label = "Export CSV"

    filepath: StringProperty(subtype="FILE_PATH", default="lod_classifier_report.csv")

    def execute(self, context):
        report_json = context.scene.get(_scene_report_key(), "")
        if not report_json:
            self.report({"WARNING"}, "No report available. Run classification first.")
            return {"CANCELLED"}

        data = json.loads(report_json)
        rows = ["cluster_id,confidence,member,lod_index,vertex_count,duplicates"]
        duplicates = data.get("duplicates", {})

        for cluster in data.get("clusters", []):
            ordered = cluster.get("ordered_lods", [])
            vertex_counts = cluster.get("vertex_counts", {})
            for lod_index, member in enumerate(ordered):
                rows.append(
                    f"{cluster.get('cluster_id')},{cluster.get('confidence')},{member},{lod_index},{vertex_counts.get(member, '')},{';'.join(duplicates.get(member, []))}"
                )

        path = Path(self.filepath)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.report({"INFO"}, f"Exported CSV report to {path}")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class LODCLASSIFIER_OT_ClearReport(Operator):
    """Clear last stored report data from scene custom properties."""

    bl_idname = "lod_classifier.clear_report"
    bl_label = "Clear Report"

    def execute(self, context):
        context.scene.pop(_scene_report_key(), None)
        context.scene.pop(_scene_summary_key(), None)
        self.report({"INFO"}, "LOD report cleared")
        return {"FINISHED"}


class LODCLASSIFIER_OT_PropagateName(Operator):
    """Rename every LOD copy in the active object's family to its base name, keeping _lod<N> suffixes"""

    bl_idname = "lod_classifier.propagate_name"
    bl_label = "Propagate Name to Family"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and is_sorted_copy(obj)

    def execute(self, context):
        renamed = MeshRenamer().propagate_family_name(context.active_object)
        self.report({"INFO"}, f"Renamed family: {', '.join(renamed)}")
        return {"FINISHED"}


class LODCLASSIFIER_OT_BuildLodCollections(Operator):
    """Move all sorted copies into LOD0..LODN collections based on their _lod<N> name suffix"""

    bl_idname = "lod_classifier.build_lod_collections"
    bl_label = "Build LOD Collections"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        counts = MeshRenamer().build_level_collections(context)
        if not counts:
            self.report({"WARNING"}, "No sorted copies found. Run Scan / Classify first.")
            return {"CANCELLED"}
        summary = ", ".join(f"LOD{lod}: {count}" for lod, count in counts.items())
        self.report({"INFO"}, f"Built LOD collections ({summary})")
        return {"FINISHED"}


class LODCLASSIFIER_OT_DeleteCopies(Operator):
    """Delete all copies created by the sorter (originals are untouched)"""

    bl_idname = "lod_classifier.delete_copies"
    bl_label = "Delete Copies"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        count = MeshRenamer().delete_copies()
        self.report({"INFO"}, f"Deleted {count} sorted copies")
        return {"FINISHED"}


class LODCLASSIFIER_OT_ShowRuntimeInfo(Operator):
    """Show addon runtime path and version to verify loaded code."""

    bl_idname = "lod_classifier.show_runtime_info"
    bl_label = "Show Runtime Info"

    def execute(self, context):
        import lod_classifier

        version = getattr(lod_classifier, "bl_info", {}).get("version", "unknown")
        addon_path = getattr(lod_classifier, "__file__", "unknown")
        has_copy_mode = hasattr(MeshRenamer, "duplicate_and_rename")

        self.report({"INFO"}, f"LOD Classifier version={version}, path={addon_path}")
        self.report({"INFO"}, f"Copy mode available={has_copy_mode}")
        return {"FINISHED"}


classes = (
    LODClassifierSettings,
    LODCLASSIFIER_OT_ClassifyScene,
    LODCLASSIFIER_OT_Cancel,
    LODCLASSIFIER_OT_ExportJson,
    LODCLASSIFIER_OT_ExportCsv,
    LODCLASSIFIER_OT_ClearReport,
    LODCLASSIFIER_OT_ShowRuntimeInfo,
    LODCLASSIFIER_OT_PropagateName,
    LODCLASSIFIER_OT_BuildLodCollections,
    LODCLASSIFIER_OT_DeleteCopies,
)


def register_properties() -> None:
    """Register scene-level addon settings."""

    bpy.types.Scene.lod_classifier_settings = PointerProperty(type=LODClassifierSettings)


def unregister_properties() -> None:
    """Unregister scene-level addon settings."""

    if hasattr(Scene, "lod_classifier_settings"):
        del bpy.types.Scene.lod_classifier_settings
