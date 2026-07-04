"""Shared dataclasses for fingerprints and classification results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Fingerprint:
    """Analyzer-specific fingerprint payload for a mesh."""

    method_name: str
    version: str
    payload: Any


@dataclass(frozen=True)
class SimilarityResult:
    """Pairwise similarity score produced by an analyzer pipeline."""

    mesh_a: str
    mesh_b: str
    method: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    """Cluster of meshes that likely represent one LOD family."""

    cluster_id: int
    members: list[str]
    confidence: float
    ordered_lods: list[str] = field(default_factory=list)
