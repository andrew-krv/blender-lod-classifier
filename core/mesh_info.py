"""Immutable mesh metadata used by analyzers and clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class MeshInfo:
    """Represents immutable geometry information for one mesh object."""

    object_ref: Any
    mesh_name: str
    vertices: np.ndarray  # (N, 3) float64, local space
    triangles: np.ndarray  # (T, 3) int64
    vertex_count: int
    face_count: int
    edge_count: int
    bounding_box: tuple[tuple[float, float, float], ...]
    bounding_box_dimensions: tuple[float, float, float]
    bounding_box_center: tuple[float, float, float]
    surface_area: float
    estimated_volume: float
    materials: tuple[str, ...]
    transform_matrix: tuple[tuple[float, float, float, float], ...]
    geometry_hash: str = ""
