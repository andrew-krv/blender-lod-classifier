"""End-to-end LOD classification pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..analyzers.bbox import BoundingBoxAnalyzer
from ..analyzers.composite import CompositeAnalyzer
from ..analyzers.normals import NormalDistributionAnalyzer
from ..analyzers.sample import SurfaceSampleAnalyzer
from ..analyzers.topology import TopologyAnalyzer
from ..analyzers.volume import VolumeAnalyzer
from ..fingerprints.fingerprint import Cluster
from ..fingerprints.fingerprint import Fingerprint
from ..utils.cache import SimpleCache
from ..utils.logging import get_logger
from .candidate_filter import CandidateFilter
from .clustering import SimilarityClusterer
from .lod_detector import VertexCountLODDetector
from .mesh_database import MeshDatabase
from .mesh_info import MeshInfo
from .renamer import MeshRenamer
from .reporting import ClassificationReport

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class PipelineSettings:
    """Configurable runtime settings for classification."""

    cluster_threshold: float = 0.75
    include_singletons: bool = False
    rename_meshes: bool = False
    rename_template: str = "mesh_{cluster:03d}_vtx_{vertices}_lod_{lod}"
    min_dimension_ratio: float = 0.45
    min_volume_ratio: float = 0.25
    min_vertex_ratio: float = 0.15
    sample_count: int = 32
    fast_prefilter: bool = True
    fast_prefilter_floor: float = 0.55
    incremental_mode: bool = True


@dataclass
class PipelineResult:
    """Result payload for one pipeline execution."""

    report: ClassificationReport
    clusters: list[Cluster]
    renamed: dict[str, list[str]]
    compared_pair_count: int = 0
    reused_pair_count: int = 0


@dataclass
class PipelineProgress:
    """Progress payload emitted during incremental pipeline execution."""

    done: bool
    cancelled: bool
    value: int
    message: str


@dataclass
class _PipelineSession:
    """Mutable state for incremental pipeline processing."""

    context: Any
    mesh_infos: list[MeshInfo]
    ordered_meshes: list[MeshInfo]
    mesh_by_name: dict[str, MeshInfo]
    mesh_signatures: dict[str, str]
    dirty_mesh_names: set[str]
    incremental_reuse: bool
    settings_signature: str
    reused_pair_count: int
    compared_pair_count: int
    cached_pair_scores: dict[tuple[str, str], float]
    cached_details_map: dict[tuple[str, str], dict[str, float]]
    fingerprints: dict[str, dict[str, Fingerprint]]
    candidates: list[tuple[MeshInfo, MeshInfo]]
    pair_scores: dict[tuple[str, str], float]
    details_map: dict[tuple[str, str], dict[str, float]]
    clusters: list[Cluster]
    renamed: dict[str, list[str]]
    stage: str
    fingerprint_index: int
    candidate_i: int
    candidate_j: int
    candidate_checked: int
    compare_index: int


class LODClassifierPipeline:
    """Coordinates database scan, analysis, clustering, and renaming."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self.logger = get_logger("pipeline")
        self.cache = SimpleCache()
        self.database = MeshDatabase(self.cache)
        self.candidate_filter = CandidateFilter(
            min_dimension_ratio=settings.min_dimension_ratio,
            min_volume_ratio=settings.min_volume_ratio,
            min_vertex_ratio=settings.min_vertex_ratio,
        )
        self.composite = CompositeAnalyzer(
            analyzers=[
                BoundingBoxAnalyzer(),
                TopologyAnalyzer(),
                SurfaceSampleAnalyzer(sample_count=settings.sample_count),
                NormalDistributionAnalyzer(),
                VolumeAnalyzer(),
            ]
        )
        self.clusterer = SimilarityClusterer()
        self.lod_detector = VertexCountLODDetector()

    def start(self, context: Any) -> _PipelineSession:
        """Create incremental execution session for the active scene."""

        mesh_infos = self.database.build(context)
        mesh_by_name = {m.mesh_name: m for m in mesh_infos}
        ordered_meshes = sorted(mesh_infos, key=lambda m: max(float(m.estimated_volume), 1e-12))

        mesh_signatures = {mesh.mesh_name: self._mesh_signature(mesh) for mesh in mesh_infos}
        settings_signature = self._settings_signature()
        cache_blob = self._load_incremental_cache(context.scene)

        can_reuse = (
            self.settings.incremental_mode
            and cache_blob.get("settings_signature", "") == settings_signature
        )

        previous_signatures = cache_blob.get("mesh_signatures", {}) if can_reuse else {}
        dirty_mesh_names = {
            name
            for name, signature in mesh_signatures.items()
            if previous_signatures.get(name) != signature
        }

        cached_pair_scores = self._decode_pair_scores(cache_blob.get("pair_scores", {})) if can_reuse else {}
        cached_details_map = self._decode_pair_details(cache_blob.get("pair_details", {})) if can_reuse else {}

        return _PipelineSession(
            context=context,
            mesh_infos=mesh_infos,
            ordered_meshes=ordered_meshes,
            mesh_by_name=mesh_by_name,
            mesh_signatures=mesh_signatures,
            dirty_mesh_names=dirty_mesh_names,
            incremental_reuse=can_reuse,
            settings_signature=settings_signature,
            reused_pair_count=0,
            compared_pair_count=0,
            cached_pair_scores=cached_pair_scores,
            cached_details_map=cached_details_map,
            fingerprints={},
            candidates=[],
            pair_scores={},
            details_map={},
            clusters=[],
            renamed={},
            stage="fingerprints",
            fingerprint_index=0,
            candidate_i=0,
            candidate_j=1,
            candidate_checked=0,
            compare_index=0,
        )

    def step(
        self,
        session: _PipelineSession,
        max_operations: int = 200,
        is_cancelled: CancelCallback | None = None,
    ) -> PipelineProgress:
        """Process one incremental chunk and return updated progress info."""

        is_cancelled = is_cancelled or (lambda: False)
        if is_cancelled():
            return PipelineProgress(done=True, cancelled=True, value=self._progress_value(session), message="Cancelled")

        if len(session.mesh_infos) < 2:
            session.stage = "done"
            return PipelineProgress(done=True, cancelled=False, value=9999, message="Not enough meshes")

        if session.stage == "fingerprints":
            total = len(session.mesh_infos)
            target = min(total, session.fingerprint_index + max_operations)
            while session.fingerprint_index < target:
                mesh = session.mesh_infos[session.fingerprint_index]
                session.fingerprints[mesh.mesh_name] = self.composite.build(mesh)
                session.fingerprint_index += 1

            if session.fingerprint_index >= total:
                session.stage = "candidates"
            message = f"Fingerprints: {session.fingerprint_index}/{total} meshes"
            return PipelineProgress(done=False, cancelled=False, value=self._progress_value(session), message=message)

        if session.stage == "candidates":
            mesh_count = len(session.ordered_meshes)
            total_pairs = (mesh_count * (mesh_count - 1)) // 2
            ops = 0
            while ops < max_operations and session.candidate_i < mesh_count - 1:
                mesh_a = session.ordered_meshes[session.candidate_i]
                mesh_b = session.ordered_meshes[session.candidate_j]

                # Because meshes are volume-sorted, once this ratio drops below threshold
                # for a fixed mesh_a, all further mesh_b values will also fail.
                if self.candidate_filter._ratio(mesh_a.estimated_volume, mesh_b.estimated_volume) < self.settings.min_volume_ratio:
                    session.candidate_i += 1
                    session.candidate_j = session.candidate_i + 1
                    continue

                if self.candidate_filter._passes(mesh_a, mesh_b):
                    key = tuple(sorted((mesh_a.mesh_name, mesh_b.mesh_name)))
                    can_reuse_pair = (
                        session.incremental_reuse
                        and mesh_a.mesh_name not in session.dirty_mesh_names
                        and mesh_b.mesh_name not in session.dirty_mesh_names
                        and key in session.cached_pair_scores
                    )
                    if can_reuse_pair:
                        session.pair_scores[key] = session.cached_pair_scores[key]
                        if key in session.cached_details_map:
                            session.details_map[key] = session.cached_details_map[key]
                        session.reused_pair_count += 1
                    else:
                        session.candidates.append((mesh_a, mesh_b))

                session.candidate_checked += 1
                ops += 1

                session.candidate_j += 1
                if session.candidate_j >= mesh_count:
                    session.candidate_i += 1
                    session.candidate_j = session.candidate_i + 1

            if session.candidate_i >= mesh_count - 1:
                session.stage = "comparisons"

            message = (
                f"Candidate filter: checked {session.candidate_checked}/{total_pairs} pairs, "
                f"compare {len(session.candidates)}, reused {session.reused_pair_count}"
            )
            return PipelineProgress(done=False, cancelled=False, value=self._progress_value(session), message=message)

        if session.stage == "comparisons":
            total = len(session.candidates)
            if total == 0:
                session.stage = "clustering"
                return PipelineProgress(
                    done=False,
                    cancelled=False,
                    value=self._progress_value(session),
                    message="Comparisons: 0/0 pairs",
                )

            target = min(total, session.compare_index + max_operations)
            while session.compare_index < target:
                mesh_a, mesh_b = session.candidates[session.compare_index]
                fp_a = session.fingerprints[mesh_a.mesh_name]
                fp_b = session.fingerprints[mesh_b.mesh_name]

                if self.settings.fast_prefilter:
                    quick_score, quick_details = self._quick_similarity(fp_a, fp_b)
                    quick_gate = min(self.settings.cluster_threshold * 0.9, self.settings.fast_prefilter_floor)
                    if quick_score < quick_gate:
                        score = quick_score
                        details = quick_details
                    else:
                        score, details = self.composite.similarity(fp_a, fp_b)
                else:
                    score, details = self.composite.similarity(fp_a, fp_b)

                key = tuple(sorted((mesh_a.mesh_name, mesh_b.mesh_name)))
                session.pair_scores[key] = score
                session.details_map[key] = details
                session.compare_index += 1
                session.compared_pair_count += 1

            if session.compare_index >= total:
                session.stage = "clustering"

            message = f"Comparisons: {session.compare_index}/{total} pairs"
            return PipelineProgress(done=False, cancelled=False, value=self._progress_value(session), message=message)

        if session.stage == "clustering":
            session.clusters = self.clusterer.build_clusters(
                meshes=session.mesh_infos,
                pair_scores=session.pair_scores,
                threshold=self.settings.cluster_threshold,
                include_singletons=self.settings.include_singletons,
            )

            for cluster in session.clusters:
                cluster.ordered_lods = self.lod_detector.order_cluster(cluster, session.mesh_by_name)

            session.stage = "renaming" if self.settings.rename_meshes else "done"
            message = f"Clusters found: {len(session.clusters)}"
            return PipelineProgress(done=False, cancelled=False, value=self._progress_value(session), message=message)

        if session.stage == "renaming":
            renamer = MeshRenamer(template=self.settings.rename_template)
            session.renamed = renamer.duplicate_and_rename(
                session.context,
                session.clusters,
                session.mesh_by_name,
            )
            session.stage = "done"
            created_count = sum(len(items) for items in session.renamed.values())
            return PipelineProgress(
                done=False,
                cancelled=False,
                value=self._progress_value(session),
                message=f"Created copies: {created_count}",
            )

        if session.stage == "done":
            return PipelineProgress(done=True, cancelled=False, value=9999, message="Done")

        return PipelineProgress(done=True, cancelled=False, value=9999, message="Done")

    def finish(self, session: _PipelineSession) -> PipelineResult:
        """Build final report from finished session state."""

        if len(session.mesh_infos) < 2:
            report = ClassificationReport(
                scene_name=session.context.scene.name,
                mesh_count=len(session.mesh_infos),
                candidate_count=0,
                comparison_count=0,
                threshold=self.settings.cluster_threshold,
                clusters=[],
                renamed={},
            )
            return PipelineResult(report=report, clusters=[], renamed={})

        report_clusters = [
            {
                "cluster_id": cluster.cluster_id,
                "confidence": cluster.confidence,
                "members": cluster.members,
                "ordered_lods": cluster.ordered_lods,
                "vertex_counts": {
                    name: session.mesh_by_name[name].vertex_count
                    for name in cluster.members
                    if name in session.mesh_by_name
                },
                "pair_scores": {
                    f"{a}|{b}": session.pair_scores[(a, b)]
                    for i, a in enumerate(cluster.members)
                    for b in cluster.members[i + 1 :]
                    if (a, b) in session.pair_scores
                },
                "analyzer_scores": {
                    f"{a}|{b}": session.details_map[(a, b)]
                    for i, a in enumerate(cluster.members)
                    for b in cluster.members[i + 1 :]
                    if (a, b) in session.details_map
                },
            }
            for cluster in session.clusters
        ]

        report = ClassificationReport(
            scene_name=session.context.scene.name,
            mesh_count=len(session.mesh_infos),
            candidate_count=len(session.candidates),
            comparison_count=len(session.pair_scores),
            threshold=self.settings.cluster_threshold,
            clusters=report_clusters,
            renamed=session.renamed,
        )

        self._save_incremental_cache(session)

        return PipelineResult(
            report=report,
            clusters=session.clusters,
            renamed=session.renamed,
            compared_pair_count=session.compared_pair_count,
            reused_pair_count=session.reused_pair_count,
        )

    def _progress_value(self, session: _PipelineSession) -> int:
        """Compute monotonic 0..9999 value from staged pipeline progress."""

        mesh_total = max(1, len(session.mesh_infos))
        pair_total = max(1, (mesh_total * (mesh_total - 1)) // 2)
        compare_total = max(1, len(session.candidates))

        if session.stage == "fingerprints":
            ratio = session.fingerprint_index / mesh_total
            return int(1000 + 2500 * ratio)
        if session.stage == "candidates":
            ratio = session.candidate_checked / pair_total
            return int(3500 + 3000 * ratio)
        if session.stage == "comparisons":
            ratio = session.compare_index / compare_total
            return int(6500 + 2500 * ratio)
        if session.stage == "clustering":
            return 9300
        if session.stage == "renaming":
            return 9700
        return 9999

    def _mesh_signature(self, mesh: MeshInfo) -> str:
        """Return a stable signature used to detect mesh changes between runs."""

        dims = mesh.bounding_box_dimensions
        return "|".join(
            (
                str(mesh.vertex_count),
                str(mesh.face_count),
                str(mesh.edge_count),
                f"{dims[0]:.6f}",
                f"{dims[1]:.6f}",
                f"{dims[2]:.6f}",
                f"{mesh.surface_area:.6f}",
                f"{mesh.estimated_volume:.6f}",
            )
        )

    def _settings_signature(self) -> str:
        """Return cache signature for settings affecting pair scores."""

        payload = {
            "min_dimension_ratio": self.settings.min_dimension_ratio,
            "min_volume_ratio": self.settings.min_volume_ratio,
            "min_vertex_ratio": self.settings.min_vertex_ratio,
            "sample_count": self.settings.sample_count,
            "fast_prefilter": self.settings.fast_prefilter,
            "fast_prefilter_floor": self.settings.fast_prefilter_floor,
        }
        return json.dumps(payload, sort_keys=True)

    def _load_incremental_cache(self, scene: Any) -> dict[str, Any]:
        """Load scene-scoped incremental cache blob."""

        raw = scene.get("lod_classifier_incremental_cache", "")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_incremental_cache(self, session: _PipelineSession) -> None:
        """Persist cache for next run in scene custom properties."""

        valid_names = set(session.mesh_by_name.keys())
        pair_scores = {
            self._pair_key_to_string(key): value
            for key, value in session.pair_scores.items()
            if key[0] in valid_names and key[1] in valid_names
        }
        pair_details = {
            self._pair_key_to_string(key): value
            for key, value in session.details_map.items()
            if key[0] in valid_names and key[1] in valid_names
        }

        payload = {
            "settings_signature": session.settings_signature,
            "mesh_signatures": session.mesh_signatures,
            "pair_scores": pair_scores,
            "pair_details": pair_details,
        }
        session.context.scene["lod_classifier_incremental_cache"] = json.dumps(payload)

    def _decode_pair_scores(self, serialized: dict[str, Any]) -> dict[tuple[str, str], float]:
        """Decode serialized pair score map."""

        decoded: dict[tuple[str, str], float] = {}
        for key, value in serialized.items():
            pair = self._pair_key_from_string(key)
            if pair is None:
                continue
            try:
                decoded[pair] = float(value)
            except (TypeError, ValueError):
                continue
        return decoded

    def _decode_pair_details(self, serialized: dict[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
        """Decode serialized analyzer-detail map."""

        decoded: dict[tuple[str, str], dict[str, float]] = {}
        for key, value in serialized.items():
            pair = self._pair_key_from_string(key)
            if pair is None or not isinstance(value, dict):
                continue

            item: dict[str, float] = {}
            for detail_key, detail_value in value.items():
                try:
                    item[str(detail_key)] = float(detail_value)
                except (TypeError, ValueError):
                    continue
            decoded[pair] = item
        return decoded

    def _pair_key_to_string(self, key: tuple[str, str]) -> str:
        """Serialize tuple pair key."""

        return f"{key[0]}|{key[1]}"

    def _pair_key_from_string(self, key: str) -> tuple[str, str] | None:
        """Parse tuple pair key from serialized form."""

        if "|" not in key:
            return None
        a, b = key.split("|", 1)
        if not a or not b:
            return None
        return tuple(sorted((a, b)))

    def _quick_similarity(
        self,
        fp_a: dict[str, Fingerprint],
        fp_b: dict[str, Fingerprint],
    ) -> tuple[float, dict[str, float]]:
        """Fast prefilter based only on cheap analyzers."""

        bbox = self._analyzer_score("bbox", fp_a, fp_b)
        topology = self._analyzer_score("topology", fp_a, fp_b)
        details = {"bbox": bbox, "topology": topology}
        return (0.6 * bbox + 0.4 * topology), details

    def _analyzer_score(
        self,
        analyzer_name: str,
        fp_a: dict[str, Fingerprint],
        fp_b: dict[str, Fingerprint],
    ) -> float:
        """Compute one analyzer score by name or return 0 when unavailable."""

        analyzer = next((a for a in self.composite.analyzers if a.name == analyzer_name), None)
        if analyzer is None:
            return 0.0

        a = fp_a.get(analyzer_name)
        b = fp_b.get(analyzer_name)
        if a is None or b is None:
            return 0.0

        return max(0.0, min(1.0, float(analyzer.similarity(a, b))))

    def run(
        self,
        context: Any,
        progress: ProgressCallback | None = None,
        is_cancelled: CancelCallback | None = None,
    ) -> PipelineResult:
        """Run the full classification pipeline for the active scene."""

        progress = progress or (lambda current, total, message: None)
        is_cancelled = is_cancelled or (lambda: False)

        session = self.start(context)
        while True:
            step = self.step(session, max_operations=2000, is_cancelled=is_cancelled)
            progress(step.value, 9999, step.message)
            if step.done:
                break

        if step.cancelled:
            report = ClassificationReport(
                scene_name=context.scene.name,
                mesh_count=len(session.mesh_infos),
                candidate_count=len(session.candidates),
                comparison_count=len(session.pair_scores),
                threshold=self.settings.cluster_threshold,
                clusters=[],
                renamed={},
            )
            return PipelineResult(report=report, clusters=[], renamed={})

        return self.finish(session)
