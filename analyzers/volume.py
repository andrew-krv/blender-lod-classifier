"""Volume and area analyzer."""

from __future__ import annotations

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, ratio_similarity
from .analyzer import Analyzer


class VolumeAnalyzer(Analyzer):
    """Compares bbox volume and surface area (both roughly preserved across LODs)."""

    name = "volume"
    version = "2"
    weight = 0.25

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Create geometric magnitude fingerprint."""

        payload = {
            "volume": mesh.estimated_volume,
            "area": mesh.surface_area,
        }
        return Fingerprint(method_name=self.name, version=self.version, payload=payload)

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Compare volume- and area-related scalar metrics."""

        volume_score = ratio_similarity(fingerprint_a.payload["volume"], fingerprint_b.payload["volume"])
        area_score = ratio_similarity(fingerprint_a.payload["area"], fingerprint_b.payload["area"])
        return clamp01(0.5 * volume_score + 0.5 * area_score)
