"""Normal-distribution analyzer based on triangle orientation histograms."""

from __future__ import annotations

from math import sqrt

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01, ratio_similarity
from .analyzer import Analyzer


class NormalDistributionAnalyzer(Analyzer):
    """Compares coarse triangle-normal orientation distributions."""

    name = "normals"
    version = "1"
    weight = 0.1

    def _normal_histogram(self, mesh: MeshInfo) -> tuple[float, float, float]:
        if not mesh.triangles or not mesh.vertices:
            return (0.0, 0.0, 0.0)

        sx = sy = sz = 0.0
        valid = 0

        for i0, i1, i2 in mesh.triangles:
            p0 = mesh.vertices[i0]
            p1 = mesh.vertices[i1]
            p2 = mesh.vertices[i2]

            ux, uy, uz = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
            vx, vy, vz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])

            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx

            mag = sqrt(nx * nx + ny * ny + nz * nz)
            if mag <= 1e-12:
                continue

            sx += abs(nx) / mag
            sy += abs(ny) / mag
            sz += abs(nz) / mag
            valid += 1

        if valid == 0:
            return (0.0, 0.0, 0.0)

        return (sx / valid, sy / valid, sz / valid)

    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Create a normal-direction histogram fingerprint."""

        return Fingerprint(method_name=self.name, version=self.version, payload={"hist": self._normal_histogram(mesh)})

    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Compare histogram ratios to score orientation similarity."""

        hist_a = fingerprint_a.payload["hist"]
        hist_b = fingerprint_b.payload["hist"]
        score = sum(ratio_similarity(a, b) for a, b in zip(hist_a, hist_b)) / 3.0
        return clamp01(score)
