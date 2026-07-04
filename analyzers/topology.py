"""Topology analyzer based on primitive counts."""

from __future__ import annotations

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, ratio_similarity
from .analyzer import Analyzer


class TopologyAnalyzer(Analyzer):
    """Compares vertex, face, and edge counts to infer LOD relation."""

    name = "topology"
    version = "1"
    weight = 0.1

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Create count-based topology fingerprint."""

        payload = {
            "vertices": mesh.vertex_count,
            "faces": mesh.face_count,
            "edges": mesh.edge_count,
        }
        return Fingerprint(method_name=self.name, version=self.version, payload=payload)

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Compute weighted similarity based on topology counts."""

        va, fa, ea = (
            fingerprint_a.payload["vertices"],
            fingerprint_a.payload["faces"],
            fingerprint_a.payload["edges"],
        )
        vb, fb, eb = (
            fingerprint_b.payload["vertices"],
            fingerprint_b.payload["faces"],
            fingerprint_b.payload["edges"],
        )

        score = (
            0.5 * ratio_similarity(va, vb)
            + 0.3 * ratio_similarity(fa, fb)
            + 0.2 * ratio_similarity(ea, eb)
        )
        return clamp01(score)
