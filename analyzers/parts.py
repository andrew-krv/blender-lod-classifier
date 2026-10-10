"""Part-structure analyzer comparing the connected islands that make up a mesh."""

from __future__ import annotations

import numpy as np

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, compute_parts
from .analyzer import Analyzer


class PartStructureAnalyzer(Analyzer):
    """Compares meshes island-by-island instead of through one global bounding box.

    Two signals are combined:

    * size spectrum: share of surface area held by parts of each (log2) size class. This is
      robust when LODs drop or merge many small pieces (foliage cards, bolts, ...).
    * major parts: islands holding at least ``major_area_share`` of the area, matched greedily
      by normalized position and extent. This catches meshes that share an overall bbox but
      are assembled from differently placed / sized pieces.
    """

    name = "parts"
    version = "1"
    weight = 0.5

    def __init__(
        self,
        major_area_share: float = 0.02,
        max_major_parts: int = 32,
        spectrum_bins: int = 8,
        position_tolerance: float = 0.1,
    ) -> None:
        self.major_area_share = major_area_share
        self.max_major_parts = max_major_parts
        self.spectrum_bins = spectrum_bins
        self.position_tolerance = position_tolerance

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Build size spectrum and normalized major-part table."""

        parts = compute_parts(mesh.vertices, mesh.triangles)
        total_area = sum(p.area for p in parts)
        scale = max(mesh.bounding_box_dimensions)
        if not parts or total_area <= 0.0 or scale <= 0.0:
            payload = {"count": len(parts), "spectrum": np.zeros(self.spectrum_bins), "major": np.zeros((0, 7))}
            return Fingerprint(method_name=self.name, version=self.version, payload=payload)

        center = np.asarray(mesh.bounding_box_center)
        shares = np.array([p.area / total_area for p in parts])
        extents = np.array([max(p.dimensions) / scale for p in parts])

        size_class = np.clip(np.floor(-np.log2(np.maximum(extents, 1e-12))), 0, self.spectrum_bins - 1).astype(int)
        spectrum = np.bincount(size_class, weights=shares, minlength=self.spectrum_bins)

        major = [
            (share, *((np.asarray(p.centroid) - center) / scale), *(np.asarray(p.dimensions) / scale))
            for p, share in zip(parts, shares)
            if share >= self.major_area_share
        ][: self.max_major_parts]

        payload = {
            "count": len(parts),
            "spectrum": spectrum,
            "major": np.array(major, dtype=np.float64).reshape(-1, 7),
        }
        return Fingerprint(method_name=self.name, version=self.version, payload=payload)

    def _major_similarity(self, major_a: np.ndarray, major_b: np.ndarray) -> float:
        if len(major_a) == 0 and len(major_b) == 0:
            return 1.0
        if len(major_a) == 0 or len(major_b) == 0:
            return 0.0

        offsets = np.linalg.norm(major_a[:, None, 1:4] - major_b[None, :, 1:4], axis=2)
        position = np.exp(-((offsets / self.position_tolerance) ** 2))
        dims_a = np.maximum(major_a[:, None, 4:7], 1e-6)
        dims_b = np.maximum(major_b[None, :, 4:7], 1e-6)
        extent = (np.minimum(dims_a, dims_b) / np.maximum(dims_a, dims_b)).mean(axis=2)
        pair = position * extent

        used_a: set[int] = set()
        used_b: set[int] = set()
        matched = 0.0
        for flat in np.argsort(-pair, axis=None):
            i, j = divmod(int(flat), pair.shape[1])
            if pair[i, j] < 0.25:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matched += pair[i, j] * (major_a[i, 0] + major_b[j, 0]) * 0.5

        total = (major_a[:, 0].sum() + major_b[:, 0].sum()) * 0.5
        return matched / total if total > 0.0 else 0.0

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Blend size-spectrum overlap with greedy major-part matching."""

        a = fingerprint_a.payload
        b = fingerprint_b.payload
        if not a["count"] or not b["count"]:
            return 1.0 if a["count"] == b["count"] else 0.0

        spectrum = float(np.minimum(a["spectrum"], b["spectrum"]).sum())
        major = self._major_similarity(a["major"], b["major"])
        return clamp01(0.4 * spectrum + 0.6 * major)
