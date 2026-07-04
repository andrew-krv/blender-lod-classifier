"""Analyzer interface for fingerprint and similarity methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.mesh_info import MeshInfo
from ..fingerprints.fingerprint import Fingerprint


class Analyzer(ABC):
    """Base interface that every geometry analyzer must implement."""

    name: str = "base"
    version: str = "1"
    weight: float = 1.0

    @abstractmethod
    def build(self, mesh: MeshInfo) -> Fingerprint:
        """Build an analyzer-specific fingerprint from MeshInfo."""

    @abstractmethod
    def similarity(self, fingerprint_a: Fingerprint, fingerprint_b: Fingerprint) -> float:
        """Return normalized similarity score in range [0, 1]."""
