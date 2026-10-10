"""Name parsing and family-name inheritance for sorted LOD copies (no Blender required)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Custom-property tags written on every created copy. They survive renaming.
TAG_FAMILY = "lod_classifier_family"
TAG_LOD = "lod_classifier_lod"
TAG_HASH = "lod_classifier_hash"
TAG_AUTO_BASE = "lod_classifier_auto_base"

_LOD_SUFFIX = re.compile(r"^(?P<base>.*?)[_\-. ]?lod_?(?P<lod>\d+)$", re.IGNORECASE)
_BLENDER_DUPLICATE_SUFFIX = re.compile(r"\.\d{3,}$")


def lod_name(base: str, lod: int) -> str:
    return f"{base}_lod{lod}"


def split_lod_name(name: str) -> tuple[str, int | None]:
    """Split ``windows_front_lod2`` / ``windows_front_LOD_2.001`` into (base, lod).

    Returns (name, None) when there is no LOD suffix.
    """

    stripped = _BLENDER_DUPLICATE_SUFFIX.sub("", name)
    match = _LOD_SUFFIX.match(stripped)
    if match is None or not match.group("base"):
        return stripped, None
    return match.group("base"), int(match.group("lod"))


@dataclass(frozen=True)
class PreviousCopy:
    """What a previous sort run left behind for one geometry."""

    base: str  # current base name (after any user renaming)
    auto_base: str  # base name the addon originally assigned
    lod: int

    @property
    def user_renamed(self) -> bool:
        return self.base != self.auto_base


def inherited_family_name(previous: list[PreviousCopy]) -> str | None:
    """Pick the base name a new family should inherit from earlier copies of its members.

    Only names the user changed are inherited. With several candidates the most common one
    wins, ties going to the name found on the lowest LOD.
    """

    renamed = [copy for copy in previous if copy.user_renamed]
    if not renamed:
        return None

    counts = Counter(copy.base for copy in renamed)
    lowest_lod = {}
    for copy in renamed:
        lowest_lod[copy.base] = min(copy.lod, lowest_lod.get(copy.base, copy.lod))
    return min(counts, key=lambda base: (-counts[base], lowest_lod[base], base))


def format_family_name(template: str, cluster: int, vertices: int) -> str:
    """Format the auto family base name; falls back to the default for invalid templates."""

    try:
        return template.format(cluster=cluster, vertices=vertices)
    except (KeyError, IndexError, ValueError):
        return f"mesh_{cluster:03d}"
