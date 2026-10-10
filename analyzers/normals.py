"""Normal-distribution analyzer based on area-weighted orientation histograms."""

from __future__ import annotations

import itertools

import numpy as np

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, triangle_areas_normals
from .analyzer import Analyzer

# 26 reference directions: cube faces, edges and corners.
_DIRECTIONS = np.array([d for d in itertools.product((-1, 0, 1), repeat=3) if any(d)], dtype=np.float64)
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1)[:, None]
_CHUNK = 65536


class NormalDistributionAnalyzer(Analyzer):
    """Compares how surface area is distributed over orientations (signed, soft-binned)."""

    name = "normals"
    version = "2"
    weight = 0.5

    def _normal_histogram(self, mesh: MeshInfo) -> np.ndarray:
        areas, normals = triangle_areas_normals(mesh.vertices, mesh.triangles)
        hist = np.zeros(len(_DIRECTIONS))
        for start in range(0, len(normals), _CHUNK):
            # Soft assignment avoids score jumps when decimation nudges normals across bin edges.
            response = np.clip(normals[start : start + _CHUNK] @ _DIRECTIONS.T, 0.0, None) ** 4
            norm = response.sum(axis=1, keepdims=True)
            response = np.divide(response, norm, out=np.zeros_like(response), where=norm > 0.0)
            hist += areas[start : start + _CHUNK] @ response

        total = hist.sum()
        return hist / total if total > 0.0 else hist

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Create a normal-direction histogram fingerprint."""

        return Fingerprint(method_name=self.name, version=self.version, payload={"hist": self._normal_histogram(mesh)})

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Histogram intersection of the orientation distributions."""

        hist_a = fingerprint_a.payload["hist"]
        hist_b = fingerprint_b.payload["hist"]
        if not hist_a.any() or not hist_b.any():
            return 1.0 if not hist_a.any() and not hist_b.any() else 0.0
        return clamp01(float(np.minimum(hist_a, hist_b).sum()))
