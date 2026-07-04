"""Fast bounding-box based analyzer."""

from __future__ import annotations

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, ratio_similarity
from .analyzer import Analyzer


class BoundingBoxAnalyzer(Analyzer):
    """Compares dimensions and aspect ratios as a coarse geometry signal."""

    name = "bbox"
    version = "1"
    weight = 0.1

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Create bounding-box dimensions and aspect-ratio fingerprint."""

        w, h, d = mesh.bounding_box_dimensions
        payload = {
            "dims": (w, h, d),
            "ratios": (
                (w / h) if h else 0.0,
                (w / d) if d else 0.0,
                (h / d) if d else 0.0,
            ),
        }
        return Fingerprint(method_name=self.name, version=self.version, payload=payload)

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Compare dimension and aspect-ratio similarity."""

        dims_a = fingerprint_a.payload["dims"]
        dims_b = fingerprint_b.payload["dims"]
        ratio_a = fingerprint_a.payload["ratios"]
        ratio_b = fingerprint_b.payload["ratios"]

        dim_score = sum(ratio_similarity(a, b) for a, b in zip(dims_a, dims_b)) / 3.0
        ratio_score = sum(ratio_similarity(a, b) for a, b in zip(ratio_a, ratio_b)) / 3.0
        return clamp01(0.7 * dim_score + 0.3 * ratio_score)
