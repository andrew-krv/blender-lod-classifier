"""Geometry extraction and basic numeric helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# Relative (to bbox diagonal) tolerances used for welding and hashing.
WELD_TOLERANCE = 1e-5
HASH_TOLERANCE = 1e-5
SAMPLE_SEED = 1337


@dataclass(frozen=True, eq=False)
class GeometryPayload:
    """Raw geometry facts extracted from a Blender mesh object."""

    vertices: np.ndarray
    triangles: np.ndarray
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
    geometry_hash: str


@dataclass(frozen=True)
class PartInfo:
    """One connected surface island of a mesh (after welding split vertices)."""

    area: float
    centroid: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    triangle_count: int


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


def as_vertex_array(vertices: Any) -> np.ndarray:
    """Return vertices as a (N, 3) float64 array."""

    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3)


def as_triangle_array(triangles: Any) -> np.ndarray:
    """Return triangles as a (T, 3) int64 array."""

    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def bbox_from_vertices(
    vertices: np.ndarray,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float], tuple[float, float, float]]:
    """Return 8 bbox corners, dimensions and center computed from vertex positions."""

    if len(vertices) == 0:
        zero = (0.0, 0.0, 0.0)
        return (zero,) * 8, zero, zero

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    corners = tuple(
        (float(x), float(y), float(z))
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    )
    dims = tuple(float(v) for v in hi - lo)
    center = tuple(float(v) for v in (hi + lo) * 0.5)
    return corners, dims, center


def padded_volume(dims: tuple[float, float, float]) -> float:
    """Bounding-box volume with flat axes padded so planes/cards are not zero volume."""

    diag = float(np.linalg.norm(dims))
    if diag <= 0.0:
        return 0.0
    floor = diag * 0.01
    return float(np.prod([max(d, floor) for d in dims]))


def triangle_areas_normals(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-triangle areas and unit normals (zero normal for degenerate triangles)."""

    if len(triangles) == 0:
        return np.zeros(0), np.zeros((0, 3))

    p0 = vertices[triangles[:, 0]]
    cross = np.cross(vertices[triangles[:, 1]] - p0, vertices[triangles[:, 2]] - p0)
    mag = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, mag[:, None], out=np.zeros_like(cross), where=mag[:, None] > 1e-20)
    return mag * 0.5, normals


def weld_vertices(vertices: np.ndarray, tolerance: float = WELD_TOLERANCE) -> np.ndarray:
    """Map each vertex to a welded index, merging vertices split at UV seams / hard edges."""

    if len(vertices) == 0:
        return np.zeros(0, dtype=np.int64)

    lo = vertices.min(axis=0)
    diag = float(np.linalg.norm(vertices.max(axis=0) - lo)) or 1.0
    quantized = np.round((vertices - lo) / (diag * tolerance)).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def triangle_part_labels(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Label each triangle with a connected-component id (welded vertex connectivity)."""

    if len(triangles) == 0:
        return np.zeros(0, dtype=np.int64)

    tris = weld_vertices(vertices)[triangles]
    labels = np.arange(int(tris.max()) + 1)
    edge_a = np.concatenate((tris[:, 0], tris[:, 1]))
    edge_b = np.concatenate((tris[:, 1], tris[:, 2]))

    # Vectorized union-find: hook larger roots onto smaller ones, then pointer-jump.
    while True:
        root_a = labels[edge_a]
        root_b = labels[edge_b]
        differ = root_a != root_b
        if not differ.any():
            break
        np.minimum.at(labels, np.maximum(root_a[differ], root_b[differ]), np.minimum(root_a[differ], root_b[differ]))
        while True:
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped

    return labels[tris[:, 0]]


def compute_parts(vertices: np.ndarray, triangles: np.ndarray) -> tuple[PartInfo, ...]:
    """Split a mesh into connected surface islands, sorted by area (largest first)."""

    if len(triangles) == 0:
        return tuple()

    areas, _ = triangle_areas_normals(vertices, triangles)
    _, part_index = np.unique(triangle_part_labels(vertices, triangles), return_inverse=True)
    part_index = part_index.reshape(-1)
    part_count = int(part_index.max()) + 1

    corners = vertices[triangles]
    weights = areas + 1e-30
    part_area = np.bincount(part_index, weights=areas, minlength=part_count)
    part_weight = np.bincount(part_index, weights=weights, minlength=part_count)
    tri_centroids = corners.mean(axis=1)
    centroids = np.stack(
        [np.bincount(part_index, weights=weights * tri_centroids[:, k], minlength=part_count) for k in range(3)],
        axis=1,
    ) / part_weight[:, None]

    part_min = np.full((part_count, 3), np.inf)
    part_max = np.full((part_count, 3), -np.inf)
    np.minimum.at(part_min, part_index, corners.min(axis=1))
    np.maximum.at(part_max, part_index, corners.max(axis=1))
    tri_counts = np.bincount(part_index, minlength=part_count)

    order = np.argsort(-part_area, kind="stable")
    return tuple(
        PartInfo(
            area=float(part_area[i]),
            centroid=tuple(float(v) for v in centroids[i]),
            dimensions=tuple(float(v) for v in part_max[i] - part_min[i]),
            triangle_count=int(tri_counts[i]),
        )
        for i in order
    )


def sample_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    count: int,
    seed: int = SAMPLE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted random surface samples (points and normals), deterministic per mesh."""

    areas, normals = triangle_areas_normals(vertices, triangles)
    total = float(areas.sum())
    if count <= 0 or total <= 0.0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(triangles), size=count, p=areas / total)
    r1 = np.sqrt(rng.random(count))[:, None]
    r2 = rng.random(count)[:, None]
    tris = triangles[chosen]
    p0 = vertices[tris[:, 0]]
    p1 = vertices[tris[:, 1]]
    p2 = vertices[tris[:, 2]]
    points = (1.0 - r1) * p0 + r1 * (1.0 - r2) * p1 + r1 * r2 * p2
    return points, normals[chosen]


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    return values[np.lexsort(values.T[::-1])]


def geometry_hash(vertices: np.ndarray, triangles: np.ndarray) -> str:
    """Translation- and ordering-invariant hash of mesh geometry, used for duplicate detection."""

    if len(vertices) == 0:
        return "empty"

    lo = vertices.min(axis=0)
    diag = float(np.linalg.norm(vertices.max(axis=0) - lo))
    step = (diag or 1.0) * HASH_TOLERANCE
    unique_vertices = np.unique(np.round((vertices - lo) / step).astype(np.int64), axis=0)

    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.int64([len(unique_vertices), len(triangles), round(diag / step)]).tobytes())
    digest.update(np.ascontiguousarray(unique_vertices).tobytes())
    if len(triangles):
        centroids = (vertices[triangles].mean(axis=1) - lo) / step
        digest.update(np.ascontiguousarray(_sorted_rows(np.round(centroids).astype(np.int64))).tobytes())
    return digest.hexdigest()


def build_payload(
    vertices: Any,
    triangles: Any,
    face_count: int,
    edge_count: int,
    materials: tuple[str, ...] = (),
    transform_matrix: tuple[tuple[float, float, float, float], ...] = (),
) -> GeometryPayload:
    """Compute derived geometry facts from raw vertex/triangle data."""

    vertices = as_vertex_array(vertices)
    triangles = as_triangle_array(triangles)
    bbox, dims, center = bbox_from_vertices(vertices)
    areas, _ = triangle_areas_normals(vertices, triangles)

    return GeometryPayload(
        vertices=vertices,
        triangles=triangles,
        vertex_count=len(vertices),
        face_count=face_count,
        edge_count=edge_count,
        bounding_box=bbox,
        bbox_dimensions=dims,
        bbox_center=center,
        surface_area=float(areas.sum()),
        estimated_volume=padded_volume(dims),
        materials=materials,
        transform_matrix=transform_matrix,
        geometry_hash=geometry_hash(vertices, triangles),
    )


def extract_geometry_payload(obj: Any) -> GeometryPayload:
    """Extract mesh facts from a Blender object into plain Python / numpy values."""

    mesh = obj.to_mesh()
    try:
        mesh.calc_loop_triangles()

        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri)

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

        return build_payload(
            vertices=co,
            triangles=tri,
            face_count=len(mesh.polygons),
            edge_count=len(mesh.edges),
            materials=materials,
            transform_matrix=matrix,
        )
    finally:
        obj.to_mesh_clear()
