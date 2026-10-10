"""Voxel occupancy analyzer comparing surface shape on a shared normalized grid."""

from __future__ import annotations

import numpy as np

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, sample_surface
from .analyzer import Analyzer


def _dilate(grid: np.ndarray) -> np.ndarray:
    """3x3x3 binary dilation via separable per-axis shifts."""

    out = grid.copy()
    for axis in range(3):
        src = out.copy()
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis] = slice(1, None)
        tail[axis] = slice(None, -1)
        out[tuple(head)] |= src[tuple(tail)]
        out[tuple(tail)] |= src[tuple(head)]
    return out


def _popcount(packed: np.ndarray) -> int:
    return int(np.unpackbits(packed).sum())


class VoxelAnalyzer(Analyzer):
    """Rasterizes dense surface samples into an occupancy grid and compares grids.

    The grid is centered on the bbox center and scaled isotropically by the largest bbox
    dimension, so meshes sharing a pivot/extent (typical for LODs) land in the same frame.
    Similarity is an F-score where a voxel counts as matched if the other mesh occupies it
    or one of its 26 neighbours, which tolerates the small surface drift of decimation.
    """

    name = "voxel"
    version = "1"
    weight = 1.0

    def __init__(self, resolution: int = 32, samples_per_cell: int = 40) -> None:
        self.resolution = max(8, int(resolution))
        self.sample_count = min(250_000, samples_per_cell * self.resolution * self.resolution)

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Build packed occupancy and dilated-occupancy grids."""

        points, _ = sample_surface(mesh.vertices, mesh.triangles, self.sample_count)
        scale = max(mesh.bounding_box_dimensions)
        if len(points) == 0 or scale <= 0.0:
            payload = {"occupied": None, "dilated": None, "count": 0}
            return Fingerprint(method_name=self.name, version=self.version, payload=payload)

        res = self.resolution
        local = (points - np.asarray(mesh.bounding_box_center)) / scale + 0.5
        cells = np.clip((local * res).astype(np.int64), 0, res - 1)
        grid = np.zeros((res, res, res), dtype=bool)
        grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True

        payload = {
            "occupied": np.packbits(grid.ravel()),
            "dilated": np.packbits(_dilate(grid).ravel()),
            "count": int(grid.sum()),
        }
        return Fingerprint(method_name=self.name, version=self.version, payload=payload)

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Tolerant F-score between occupancy grids."""

        a = fingerprint_a.payload
        b = fingerprint_b.payload
        if not a["count"] or not b["count"]:
            return 1.0 if a["count"] == b["count"] else 0.0
        if len(a["occupied"]) != len(b["occupied"]):
            return 0.0

        precision = _popcount(a["occupied"] & b["dilated"]) / a["count"]
        recall = _popcount(b["occupied"] & a["dilated"]) / b["count"]
        if precision + recall <= 0.0:
            return 0.0
        return clamp01(2.0 * precision * recall / (precision + recall))
