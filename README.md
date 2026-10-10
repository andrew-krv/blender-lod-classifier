# LOD Classifier Add-on Package

This folder contains the Blender add-on package code.

## Version

Current addon version: 0.6.0

## What It Does

- Scans mesh objects in the active scene.
- Detects probable LOD families by geometry similarity.
- Collapses duplicate meshes (identical geometry) so they are not reported as extra LODs.
- Orders members by likely LOD level (highest vertex count = LOD0).
- Optionally creates renamed copies (original meshes are never renamed).
- Places created copies into collections:
  - LOD_Clusters
  - cluster_001, cluster_002, etc.
- Exports JSON/CSV reports from panel operators.

## Main Modules

- __init__.py: Blender addon metadata and entry.
- addon.py: register/unregister wiring.
- core/: database, pipeline, filtering, clustering, renaming, reporting.
- analyzers/: pluggable similarity analyzers.
- fingerprints/: shared result dataclasses.
- ui/: operators and panel.
- utils/: geometry/cache/logging helpers.

## UI (View3D Sidebar -> LOD Classifier)

Pipeline:
- Collection (optional: only classify meshes in this collection and its children; empty = whole scene)
- Threshold
- Include Singletons
- Collapse Duplicates
- Linkage (Average / Single)

Candidate Filter:
- Min Dim Ratio
- Min Volume Ratio
- Min Vertex Ratio

Performance:
- Voxel Resolution
- Fast Prefilter
- Incremental Mode
- Tick Budget

Output Copies:
- Create Renamed Copies (Keep Originals)
- Family Name Template (base name, `_lod<N>` is appended)

Sorted Copies:
- Propagate Name to Family: rename the active copy (e.g. `mesh_200_lod0` -> `window_front_lod0`),
  press this, and its whole family becomes `window_front_lod<N>`.
- Build LOD Collections: moves every sorted copy into `LOD_Levels/LOD0..LODN` by its `_lod<N>`
  name suffix (edit the suffix to fix an ordering).
- Delete Copies: removes everything the sorter created.

Progress:
- Live percentage and status text during run.

## How Matching Works

1. Duplicates: meshes with the same geometry hash (independent of vertex/triangle order,
   UV-seam splits and translation) are collapsed to one representative. Near duplicates
   (float noise, re-ordered triangles) are folded in after comparison.
2. Candidate filter: cheap bbox / volume / vertex-ratio gates.
3. Analyzers, combined as a weighted product `prod(score ** weight)` so one failing check
   cannot be averaged away:
   - voxel (1.0): occupancy grid of dense area-weighted surface samples, tolerant F-score.
   - parts (0.5): connected islands after welding split vertices; compares the area size
     spectrum and matches major parts by position/extent.
   - normals (0.5): area-weighted orientation histogram.
   - bbox (0.5): dimensions and aspect ratios.
   - volume (0.25): bbox volume and surface area.
   - topology (0.0): vertex/face/edge counts, reported only (LODs differ by design).
4. Clustering: average-linkage agglomerative merging; pairs that were never compared count
   as 0, so A~B and B~C no longer drags unrelated A and C into one family.

Per-analyzer scores for every pair are in the JSON report (`analyzer_scores`), which is the
place to look when tuning the threshold or weights on real data.

## Sort Workflow

1. Scan / Classify copies one object per unique mesh into `LOD_Clusters/<family>` as
   `<family>_lod<N>`; duplicates are not copied.
2. Rename copies as you like. Copies carry custom-property tags (`lod_classifier_*`: family id,
   LOD index, geometry hash, original auto name), so they stay recognisable after renaming.
3. Re-running the sort replaces all copies; a family inherits any base name you gave to one of
   its members before (matched by geometry hash). LOD indices are recomputed.
   Only names carry over - other edits made on copies are lost on re-sort.
4. Build LOD Collections when done.

Tagged copies are never scanned as source meshes, wherever they are moved.

## Important Behavior

- Copy mode duplicates objects and mesh data; no linked instances are created.
- Originals remain untouched.
- If Create Renamed Copies is disabled, no scene objects are changed.
- Incremental Mode reuses pair scores for unchanged meshes to speed up repeated runs.
- Cache automatically refreshes when mesh signatures or analyzer-impacting settings change.

## Runtime Verification

Use panel action:
- Show Runtime Info

It prints loaded addon version and path so you can confirm Blender is running this package copy.

## Notes For Git Backup

Commit this folder as the canonical addon source:
- lod_classifier/

You can zip this folder for Blender installation if needed.
