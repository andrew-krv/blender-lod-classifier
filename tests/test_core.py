"""Unit tests for non-Blender core logic."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from lod_classifier.analyzers.bbox import BoundingBoxAnalyzer
from lod_classifier.analyzers.composite import CompositeAnalyzer
from lod_classifier.analyzers.topology import TopologyAnalyzer
from lod_classifier.core.clustering import SimilarityClusterer
from lod_classifier.core.duplicates import DuplicateDetector
from lod_classifier.core.lod_detector import VertexCountLODDetector
from lod_classifier.core.naming import PreviousCopy, inherited_family_name, split_lod_name
from lod_classifier.core.pipeline import LODClassifierPipeline, PipelineSettings
from lod_classifier.tests.synthetic import box, cylinder, ellipsoid, merge, mesh_info, shuffled, torus
from lod_classifier.utils.geometry import compute_parts


class _FakeScene(dict):
    name = "Scene"


def _run_pipeline(meshes, **settings):
    pipeline = LODClassifierPipeline(PipelineSettings(**settings))
    pipeline.database.build = lambda context, **kwargs: meshes
    context = SimpleNamespace(scene=_FakeScene())
    return pipeline.run(context)


class TestCore(unittest.TestCase):
    def test_composite_similarity(self) -> None:
        mesh_a = mesh_info("A", *box(4, (1.0, 2.0, 1.0)))
        mesh_b = mesh_info("B", *box(2, (1.0, 2.1, 1.05)))

        composite = CompositeAnalyzer(analyzers=[BoundingBoxAnalyzer(), TopologyAnalyzer()])
        fp_a = composite.build(mesh_a)
        fp_b = composite.build(mesh_b)
        score, details = composite.similarity(fp_a, fp_b)

        self.assertGreater(score, 0.6)
        self.assertIn("bbox", details)
        self.assertIn("topology", details)

    def test_cluster_and_lod_order(self) -> None:
        meshes = [mesh_info("M0", *ellipsoid(32, 64)), mesh_info("M1", *ellipsoid(16, 32)), mesh_info("M2", *ellipsoid(8, 16))]
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

    def test_average_linkage_does_not_chain(self) -> None:
        meshes = [mesh_info(n, *box(1)) for n in ("A", "B", "C")]
        # A~B and B~C, but A and C were never similar enough to be compared.
        pair_scores = {("A", "B"): 0.9, ("B", "C"): 0.9}

        average = SimilarityClusterer().build_clusters(meshes, pair_scores, threshold=0.75)
        self.assertEqual([c.members for c in average], [["A", "B"]])

        single = SimilarityClusterer(linkage="single").build_clusters(meshes, pair_scores, threshold=0.75)
        self.assertEqual([c.members for c in single], [["A", "B", "C"]])

    def test_parts_weld_seams_and_split_islands(self) -> None:
        sphere = ellipsoid(16, 32)  # seam column and pole vertices are duplicated
        self.assertEqual(len(compute_parts(*sphere)), 1)

        pair = merge(ellipsoid(8, 16, (0.4, 1, 1), (-0.6, 0, 0)), ellipsoid(8, 16, (0.4, 1, 1), (0.6, 0, 0)))
        parts = compute_parts(*pair)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(parts[0].centroid[0], -parts[1].centroid[0], places=3)

    def test_duplicate_hash_ignores_order_and_translation(self) -> None:
        vertices, triangles = torus(12, 24)
        original = mesh_info("rock", vertices, triangles)
        moved = mesh_info("rock.001", *shuffled(vertices, triangles, translate=(5.0, -2.0, 1.0)))
        other = mesh_info("rock_lod1", *torus(6, 12))

        self.assertEqual(original.geometry_hash, moved.geometry_hash)
        self.assertNotEqual(original.geometry_hash, other.geometry_hash)
        self.assertEqual(DuplicateDetector().group_exact([moved, original, other]), {"rock": ["rock.001"]})

    def test_lods_score_above_same_bbox_lookalikes(self) -> None:
        pipeline = LODClassifierPipeline(PipelineSettings())
        meshes = {
            "sphere_lod0": ellipsoid(32, 64),
            "sphere_lod2": ellipsoid(8, 16),
            "cylinder": cylinder(8, 32),
            "cube": box(8),
            "two_halves": merge(ellipsoid(16, 32, (0.45, 1, 1), (-0.55, 0, 0)), ellipsoid(16, 32, (0.45, 1, 1), (0.55, 0, 0))),
        }
        fps = {name: pipeline.composite.build(mesh_info(name, *m)) for name, m in meshes.items()}

        def score(a: str, b: str) -> float:
            return pipeline.composite.similarity(fps[a], fps[b])[0]

        lod_score = score("sphere_lod0", "sphere_lod2")
        self.assertGreater(lod_score, 0.9)
        for other in ("cylinder", "cube", "two_halves"):
            self.assertLess(score("sphere_lod0", other), 0.6, other)

    def test_pipeline_end_to_end(self) -> None:
        sphere0 = ellipsoid(32, 64)
        meshes = [
            mesh_info("sphere_a", *sphere0),
            mesh_info("sphere_b", *ellipsoid(16, 32)),
            mesh_info("sphere_c", *ellipsoid(8, 16)),
            mesh_info("sphere_a.001", *shuffled(*sphere0, translate=(3.0, 0.0, 0.0))),
            mesh_info("box_a", *box(8)),
            mesh_info("box_b", *box(2)),
            mesh_info("cylinder", *cylinder(8, 32)),
            mesh_info("torus", *torus(16, 32)),
        ]

        result = _run_pipeline(meshes)
        groups = sorted(c.ordered_lods for c in result.clusters)
        self.assertEqual(groups, [["box_a", "box_b"], ["cylinder"], ["sphere_a", "sphere_b", "sphere_c"], ["torus"]])
        self.assertEqual(result.report.duplicates, {"sphere_a": ["sphere_a.001"]})

        # Without collapsing, the copy is clustered as an extra "LOD" (the old behaviour).
        result = _run_pipeline(meshes, collapse_duplicates=False)
        sphere_group = next(c for c in result.clusters if "sphere_a" in c.members)
        self.assertIn("sphere_a.001", sphere_group.members)

    def test_split_lod_name(self) -> None:
        self.assertEqual(split_lod_name("mesh_200_lod0"), ("mesh_200", 0))
        self.assertEqual(split_lod_name("windows_front_LOD_12.001"), ("windows_front", 12))
        self.assertEqual(split_lod_name("wheel.lod3"), ("wheel", 3))
        self.assertEqual(split_lod_name("crate"), ("crate", None))
        self.assertEqual(split_lod_name("lod2"), ("lod2", None))

    def test_inherited_family_name_prefers_user_renames(self) -> None:
        self.assertIsNone(inherited_family_name([PreviousCopy("mesh_001", "mesh_001", 0)]))
        previous = [
            PreviousCopy("mesh_001", "mesh_001", 0),
            PreviousCopy("window_front", "mesh_001", 1),
            PreviousCopy("window_side", "mesh_001", 2),
        ]
        self.assertEqual(inherited_family_name(previous), "window_front")


if __name__ == "__main__":
    unittest.main()
