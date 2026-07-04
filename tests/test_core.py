"""Unit tests for non-Blender core logic."""

from __future__ import annotations

import unittest

from lod_classifier.analyzers.bbox import BoundingBoxAnalyzer
from lod_classifier.analyzers.composite import CompositeAnalyzer
from lod_classifier.analyzers.topology import TopologyAnalyzer
from lod_classifier.core.clustering import SimilarityClusterer
from lod_classifier.core.lod_detector import VertexCountLODDetector
from lod_classifier.core.mesh_info import MeshInfo


def _mesh(name: str, vertex_count: int, dims: tuple[float, float, float]) -> MeshInfo:
    return MeshInfo(
        object_ref=None,
        mesh_name=name,
        vertices=((0.0, 0.0, 0.0),) * max(1, vertex_count),
        triangles=((0, 0, 0),),
        vertex_count=vertex_count,
        face_count=max(1, vertex_count // 2),
        edge_count=max(1, vertex_count // 2),
        bounding_box=((0.0, 0.0, 0.0),) * 8,
        bounding_box_dimensions=dims,
        bounding_box_center=(0.0, 0.0, 0.0),
        surface_area=float(max(1, vertex_count)),
        estimated_volume=max(dims[0] * dims[1] * dims[2], 0.0),
        materials=(),
        transform_matrix=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


class TestCore(unittest.TestCase):
    def test_composite_similarity(self) -> None:
        mesh_a = _mesh("A", 1000, (1.0, 2.0, 1.0))
        mesh_b = _mesh("B", 800, (1.0, 2.1, 1.05))

        composite = CompositeAnalyzer(analyzers=[BoundingBoxAnalyzer(), TopologyAnalyzer()])
        fp_a = composite.build(mesh_a)
        fp_b = composite.build(mesh_b)
        score, details = composite.similarity(fp_a, fp_b)

        self.assertGreater(score, 0.6)
        self.assertIn("bbox", details)
        self.assertIn("topology", details)

    def test_cluster_and_lod_order(self) -> None:
        meshes = [_mesh("M0", 1000, (1.0, 1.0, 1.0)), _mesh("M1", 600, (1.0, 1.0, 1.0)), _mesh("M2", 300, (1.0, 1.0, 1.0))]
        pair_scores = {
            ("M0", "M1"): 0.9,
            ("M0", "M2"): 0.85,
            ("M1", "M2"): 0.88,
        }

        clusterer = SimilarityClusterer()
        clusters = clusterer.build_clusters(meshes, pair_scores, threshold=0.8, include_singletons=True)
        self.assertEqual(len(clusters), 1)

        detector = VertexCountLODDetector()
        order = detector.order_cluster(clusters[0], {m.mesh_name: m for m in meshes})
        self.assertEqual(order, ["M0", "M1", "M2"])


if __name__ == "__main__":
    unittest.main()
