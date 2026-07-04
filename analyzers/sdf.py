"""Future signed-distance-field analyzer placeholder."""

from __future__ import annotations

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from .analyzer import Analyzer


class SignedDistanceFieldAnalyzer(Analyzer):
    """Reserved for future SDF implementation."""

    name = "sdf"
    version = "0"
    weight = 0.0

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Return placeholder payload for unsupported analyzer."""

        return Fingerprint(method_name=self.name, version=self.version, payload={})

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Return zero until analyzer is implemented."""

        return 0.0
