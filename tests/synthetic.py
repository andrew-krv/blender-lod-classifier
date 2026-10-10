"""Procedural test meshes (no Blender required)."""

from __future__ import annotations

import numpy as np

from lod_classifier.core.mesh_info import MeshInfo
from lod_classifier.utils.geometry import build_payload


def mesh_info(name: str, vertices: np.ndarray, triangles: np.ndarray) -> MeshInfo:
    payload = build_payload(vertices, triangles, face_count=len(triangles), edge_count=0)
    return MeshInfo(
        object_ref=None,
        mesh_name=name,
        vertices=payload.vertices,
        triangles=payload.triangles,
        vertex_count=payload.vertex_count,
        face_count=payload.face_count,
        edge_count=payload.edge_count,
        bounding_box=payload.bounding_box,
        bounding_box_dimensions=payload.bbox_dimensions,
        bounding_box_center=payload.bbox_center,
        surface_area=payload.surface_area,
        estimated_volume=payload.estimated_volume,
        materials=payload.materials,
        transform_matrix=payload.transform_matrix,
        geometry_hash=payload.geometry_hash,
    )


def grid_surface(func, rows: int, cols: int, wrap_u: bool = True, split_seam: bool = True):
    """Parametric surface with (u, v) in [0, 1]^2. Seam vertices are duplicated like game exports."""

    u = np.linspace(0.0, 1.0, cols + 1)
    v = np.linspace(0.0, 1.0, rows + 1)
    uu, vv = np.meshgrid(u, v)
    vertices = func(uu.ravel(), vv.ravel())
    width = cols + 1
    tris = []
    for r in range(rows):
        for c in range(cols):
            a = r * width + c
            b = a + 1
            d = a + width
            e = d + 1
            tris.append((a, b, e))
            tris.append((a, e, d))
    return vertices, np.array(tris)


def ellipsoid(rows: int, cols: int, radii=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    def f(u, v):
        theta = u * 2 * np.pi
        phi = v * np.pi
        return np.stack(
            (
                radii[0] * np.sin(phi) * np.cos(theta) + center[0],
                radii[1] * np.sin(phi) * np.sin(theta) + center[1],
                radii[2] * np.cos(phi) + center[2],
            ),
            axis=1,
        )

    return grid_surface(f, rows, cols)


def cylinder(rows: int, cols: int, radius=1.0, height=2.0):
    def f(u, v):
        theta = u * 2 * np.pi
        return np.stack((radius * np.cos(theta), radius * np.sin(theta), (v - 0.5) * height), axis=1)

    side_v, side_t = grid_surface(f, rows, cols)
    # caps as triangle fans
    verts = [side_v]
    tris = [side_t]
    offset = len(side_v)
    for z in (-height / 2, height / 2):
        ring = np.stack(
            (radius * np.cos(np.linspace(0, 2 * np.pi, cols, endpoint=False)),
             radius * np.sin(np.linspace(0, 2 * np.pi, cols, endpoint=False)),
             np.full(cols, z)),
            axis=1,
        )
        cap = np.vstack(([0.0, 0.0, z], ring))
        fan = [(offset, offset + 1 + i, offset + 1 + (i + 1) % cols) for i in range(cols)]
        verts.append(cap)
        tris.append(np.array(fan))
        offset += len(cap)
    return np.vstack(verts), np.vstack(tris)


def box(subdiv: int, size=(2.0, 2.0, 2.0)):
    verts = []
    tris = []
    offset = 0
    lin = np.linspace(-1.0, 1.0, subdiv + 1)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            a, b = [k for k in range(3) if k != axis]
            uu, vv = np.meshgrid(lin, lin)
            face = np.zeros((uu.size, 3))
            face[:, axis] = sign
            face[:, a] = uu.ravel()
            face[:, b] = vv.ravel()
            w = subdiv + 1
            for r in range(subdiv):
                for c in range(subdiv):
                    i = offset + r * w + c
                    tris.append((i, i + 1, i + w + 1))
                    tris.append((i, i + w + 1, i + w))
            verts.append(face)
            offset += len(face)
    v = np.vstack(verts) * (np.asarray(size) / 2.0)
    return v, np.array(tris)


def torus(rows: int, cols: int, major=0.75, minor=0.25):
    def f(u, v):
        theta = u * 2 * np.pi
        phi = v * 2 * np.pi
        r = major + minor * np.cos(phi)
        return np.stack((r * np.cos(theta), r * np.sin(theta), minor * np.sin(phi)), axis=1)

    return grid_surface(f, rows, cols)


def merge(*meshes):
    verts = []
    tris = []
    offset = 0
    for v, t in meshes:
        verts.append(v)
        tris.append(t + offset)
        offset += len(v)
    return np.vstack(verts), np.vstack(tris)


def shuffled(vertices: np.ndarray, triangles: np.ndarray, seed: int = 7, translate=(0.0, 0.0, 0.0)):
    """Same geometry with permuted vertex/triangle order and a translation."""

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vertices))
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    new_vertices = vertices[perm] + np.asarray(translate)
    new_triangles = inverse[triangles][rng.permutation(len(triangles))]
    return new_vertices, new_triangles
