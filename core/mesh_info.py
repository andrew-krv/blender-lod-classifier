"""Immutable mesh metadata used by analyzers and clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MeshInfo:
    """Represents immutable geometry information for one mesh object."""

    object_ref: Any
    mesh_name: str
    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
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
