"""Surface sampling analyzer for MVP geometry comparison."""

from __future__ import annotations

from statistics import fmean

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import average_point_distance, clamp01, ratio_similarity
from .analyzer import Analyzer


class SurfaceSampleAnalyzer(Analyzer):
    """Compares normalized point samples from mesh surfaces."""

    name = "surface_sample"
    version = "1"
    weight = 0.6

    def __init__(self, sample_count: int = 64) -> None:
        self.sample_count = max(8, sample_count)

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Build normalized vertex sample fingerprint."""

        if not mesh.vertices:
            payload = {"points": tuple(), "radial_mean": 0.0}
            return Fingerprint(method_name=self.name, version=self.version, payload=payload)

        cx = fmean(v[0] for v in mesh.vertices)
        cy = fmean(v[1] for v in mesh.vertices)
        cz = fmean(v[2] for v in mesh.vertices)

        centered = tuple((v[0] - cx, v[1] - cy, v[2] - cz) for v in mesh.vertices)
        max_abs = max(max(abs(c) for c in p) for p in centered) or 1.0
        normalized = tuple((p[0] / max_abs, p[1] / max_abs, p[2] / max_abs) for p in centered)

        step = max(1, len(normalized) // self.sample_count)
        points = normalized[::step][: self.sample_count]

        radial_mean = fmean((p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5 for p in points) if points else 0.0
        payload = {"points": points, "radial_mean": radial_mean, "vertices": mesh.vertex_count}
        return Fingerprint(method_name=self.name, version=self.version, payload=payload)

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Compute bidirectional nearest-neighbor sample similarity."""

        points_a = fingerprint_a.payload["points"]
        points_b = fingerprint_b.payload["points"]

        if not points_a or not points_b:
            return 0.0

        d_ab = average_point_distance(points_a, points_b)
        d_ba = average_point_distance(points_b, points_a)
        shape_distance = (d_ab + d_ba) * 0.5

        radial_score = ratio_similarity(
            fingerprint_a.payload["radial_mean"],
            fingerprint_b.payload["radial_mean"],
        )
        count_score = ratio_similarity(fingerprint_a.payload["vertices"], fingerprint_b.payload["vertices"])

        shape_score = 1.0 / (1.0 + 4.0 * shape_distance)
        return clamp01(0.7 * shape_score + 0.2 * radial_score + 0.1 * count_score)
