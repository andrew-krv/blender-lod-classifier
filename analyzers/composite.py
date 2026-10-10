"""Composite analyzer orchestrating multiple independent analyzers."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint
from ..utils.geometry import clamp01
from .analyzer import Analyzer


@dataclass
class CompositeAnalyzer:
    """Builds per-analyzer fingerprints and computes weighted similarity.

    Scores are combined as a weighted product ``prod(score ** weight)``: every analyzer acts
    as a penalty factor and the weight is its strictness exponent. Unlike an average, analyzers
    that trivially agree (same bbox) cannot dilute one that disagrees (different shape).
    Analyzers with zero weight are still evaluated and reported in details.
    """

    analyzers: list[Analyzer]

    def build(self, mesh: MeshInfo) -> dict[str, Fingerprint]:
        """Build fingerprint map indexed by analyzer name."""

        return {analyzer.name: analyzer.build(mesh) for analyzer in self.analyzers}

    def similarity(self, fp_a: dict[str, Fingerprint], fp_b: dict[str, Fingerprint]) -> tuple[float, dict[str, float]]:
        """Compute weighted similarity and return analyzer-level details."""

        log_sum = 0.0
        weight_total = 0.0
        details: dict[str, float] = {}

        for analyzer in self.analyzers:
            a = fp_a.get(analyzer.name)
            b = fp_b.get(analyzer.name)
            if a is None or b is None:
                continue

            score = clamp01(analyzer.similarity(a, b))
            details[analyzer.name] = score
            if analyzer.weight <= 0.0:
                continue
            log_sum += analyzer.weight * log(max(score, 1e-6))
            weight_total += analyzer.weight

        if weight_total == 0.0:
            return 0.0, details

        return clamp01(exp(log_sum)), details
