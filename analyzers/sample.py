"""Surface sampling analyzer comparing point clouds with a chamfer distance."""

from __future__ import annotations

import numpy as np

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, sample_surface
from .analyzer import Analyzer


def _nearest_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Distance from every point in A to its nearest neighbour in B."""

    d2 = (
        (points_a * points_a).sum(axis=1)[:, None]
        + (points_b * points_b).sum(axis=1)[None, :]
        - 2.0 * points_a @ points_b.T
    )
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


class SurfaceSampleAnalyzer(Analyzer):
    """Compares area-weighted surface samples in bbox-centered (not rescaled) space.

    Optional analyzer: the voxel analyzer is the default shape signal. Sparse random samples
    never coincide exactly, so this score saturates below 1.0 even for identical meshes.
    """

    name = "surface_sample"
    version = "2"
    weight = 0.1

    def __init__(self, sample_count: int = 256) -> None:
        self.sample_count = max(32, sample_count)

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Build bbox-centered surface sample fingerprint."""

        points, _ = sample_surface(mesh.vertices, mesh.triangles, self.sample_count)
        payload = {
            "points": points - np.asarray(mesh.bounding_box_center),
            "diag": float(np.linalg.norm(mesh.bounding_box_dimensions)),
        }
        return Fingerprint(method_name=self.name, version=self.version, payload=payload)

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Symmetric chamfer distance relative to the larger bbox diagonal."""

        points_a = fingerprint_a.payload["points"]
        points_b = fingerprint_b.payload["points"]
        diag = max(fingerprint_a.payload["diag"], fingerprint_b.payload["diag"])
        if len(points_a) == 0 or len(points_b) == 0 or diag <= 0.0:
            return 0.0

        chamfer = 0.5 * (_nearest_distances(points_a, points_b).mean() + _nearest_distances(points_b, points_a).mean())
        return clamp01(float(np.exp(-chamfer / (0.05 * diag))))
