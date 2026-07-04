"""Geometry extraction and basic numeric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Any


@dataclass(frozen=True)
class GeometryPayload:
    """Raw geometry facts extracted from a Blender mesh object."""

    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    vertex_count: int
    face_count: int
    edge_count: int
    bounding_box: tuple[tuple[float, float, float], ...]
    bbox_dimensions: tuple[float, float, float]
    bbox_center: tuple[float, float, float]
    surface_area: float
    estimated_volume: float
    materials: tuple[str, ...]
    transform_matrix: tuple[tuple[float, float, float, float], ...]


def _bbox_dimensions_and_center(
    bbox: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return dimensions and center for a bound-box represented as 8 points."""

    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    zs = [p[2] for p in bbox]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    dims = (max_x - min_x, max_y - min_y, max_z - min_z)
    center = ((max_x + min_x) * 0.5, (max_y + min_y) * 0.5, (max_z + min_z) * 0.5)
    return dims, center


def clamp01(value: float) -> float:
    """Clamp a number to the inclusive [0, 1] range."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def ratio_similarity(a: float, b: float) -> float:
    """Similarity in [0, 1] based on relative ratio for non-negative values."""

    x = max(float(a), 0.0)
    y = max(float(b), 0.0)
    if x == 0.0 and y == 0.0:
        return 1.0
    if x == 0.0 or y == 0.0:
        return 0.0
    small = min(x, y)
    large = max(x, y)
    return clamp01(small / large)


def average_point_distance(
    points_a: tuple[tuple[float, float, float], ...],
    points_b: tuple[tuple[float, float, float], ...],
) -> float:
    """Average nearest-neighbor distance from points_a to points_b."""

    if not points_a or not points_b:
        return 1e9

    total = 0.0
    for pa in points_a:
        nearest = min(dist(pa, pb) for pb in points_b)
        total += nearest
    return total / len(points_a)


def extract_geometry_payload(obj: Any) -> GeometryPayload:
    """Extract mesh facts from a Blender object into plain Python values."""

    mesh = obj.to_mesh()
    try:
        mesh.calc_loop_triangles()

        vertices = tuple((float(v.co.x), float(v.co.y), float(v.co.z)) for v in mesh.vertices)
        triangles = tuple(
            (int(t.vertices[0]), int(t.vertices[1]), int(t.vertices[2])) for t in mesh.loop_triangles
        )

        bbox = tuple((float(p[0]), float(p[1]), float(p[2])) for p in obj.bound_box)
        dims, center = _bbox_dimensions_and_center(bbox)
        area = float(sum(t.area for t in mesh.loop_triangles))
        volume = float(max(dims[0] * dims[1] * dims[2], 0.0))

        materials = tuple(m.name for m in obj.data.materials if m is not None)
        matrix = tuple(
            (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
            )
            for row in obj.matrix_world
        )

        return GeometryPayload(
            vertices=vertices,
            triangles=triangles,
            vertex_count=len(vertices),
            face_count=len(mesh.polygons),
            edge_count=len(mesh.edges),
            bounding_box=bbox,
            bbox_dimensions=dims,
            bbox_center=center,
            surface_area=area,
            estimated_volume=volume,
            materials=materials,
            transform_matrix=matrix,
        )
    finally:
        obj.to_mesh_clear()
