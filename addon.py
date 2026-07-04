"""Addon registration helpers."""

from __future__ import annotations

from .ui.operators import classes as operator_classes
from .ui.operators import register_properties, unregister_properties
from .ui.panel import classes as panel_classes


def register() -> None:
    """Register addon classes and properties."""
    import bpy

    for cls in operator_classes + panel_classes:
        bpy.utils.register_class(cls)
    register_properties()


def unregister() -> None:
    """Unregister addon classes and properties."""
    import bpy

    unregister_properties()
    for cls in reversed(operator_classes + panel_classes):
        bpy.utils.unregister_class(cls)
