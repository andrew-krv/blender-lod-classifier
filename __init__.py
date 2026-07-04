"""Blender addon entry point for LOD Classifier."""

from __future__ import annotations

from typing import Any

bl_info = {
    "name": "LOD Classifier",
    "author": "LOD Classifier Contributors",
    "version": (0, 4, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > LOD Classifier",
    "description": "Recover probable LOD groups from imported meshes.",
    "category": "Object",
}

try:
    from .addon import register, unregister
except ModuleNotFoundError:
    # Allow importing core modules in non-Blender environments (tests, CI).
    def register() -> Any:
        raise RuntimeError("Blender Python API (bpy) is required to register this addon.")

    def unregister() -> Any:
        raise RuntimeError("Blender Python API (bpy) is required to unregister this addon.")

__all__ = ["register", "unregister", "bl_info"]
