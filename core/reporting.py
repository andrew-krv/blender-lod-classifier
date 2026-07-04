"""Report serialization helpers for classification results."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassificationReport:
    """Serializable report for clustering, scores, and rename decisions."""

    scene_name: str
    mesh_count: int
    candidate_count: int
    comparison_count: int
    threshold: float
    clusters: list[dict[str, Any]] = field(default_factory=list)
    renamed: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return report as plain dictionary."""

        return {
            "scene": self.scene_name,
            "mesh_count": self.mesh_count,
            "candidate_count": self.candidate_count,
            "comparison_count": self.comparison_count,
            "threshold": self.threshold,
            "clusters": self.clusters,
            "renamed": self.renamed,
        }

    def to_json(self) -> str:
        """Serialize report as JSON text."""

        return json.dumps(self.to_dict(), indent=2)

    def to_csv(self) -> str:
        """Serialize flattened cluster data as CSV text."""

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["cluster_id", "confidence", "member", "lod_index", "vertex_count"])

        for cluster in self.clusters:
            ordered = cluster.get("ordered_lods", [])
            vertices = cluster.get("vertex_counts", {})
            for lod_index, member in enumerate(ordered):
                writer.writerow(
                    [
                        cluster.get("cluster_id"),
                        cluster.get("confidence"),
                        member,
                        lod_index,
                        vertices.get(member),
                    ]
                )

        return output.getvalue()
