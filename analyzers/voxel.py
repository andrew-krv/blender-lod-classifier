"""Future voxel occupancy analyzer placeholder."""

from __future__ import annotations

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from .analyzer import Analyzer


class VoxelAnalyzer(Analyzer):
    """Reserved for future occupancy-grid implementation."""

    name = "voxel"
    version = "0"
    weight = 0.0

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Return placeholder payload for unsupported analyzer."""

        return Fingerprint(method_name=self.name, version=self.version, payload={})

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Return zero until analyzer is implemented."""

        return 0.0
