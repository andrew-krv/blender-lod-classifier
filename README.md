# LOD Classifier Add-on Package

This folder contains the Blender add-on package code.

## Version

Current addon version: 0.3.0

## What It Does

- Scans mesh objects in the active scene.
- Detects probable LOD families by geometry similarity.
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
- Threshold
- Include Singletons

Candidate Filter:
- Min Dim Ratio
- Min Volume Ratio
- Min Vertex Ratio

Performance:
- Surface Samples
- Fast Prefilter
- Tick Budget

Output Copies:
- Create Renamed Copies (Keep Originals)
- Copy Name Template

Progress:
- Live percentage and status text during run.

## Important Behavior

- Copy mode duplicates objects and mesh data; no linked instances are created.
- Originals remain untouched.
- If Create Renamed Copies is disabled, no scene objects are changed.

## Runtime Verification

Use panel action:
- Show Runtime Info

It prints loaded addon version and path so you can confirm Blender is running this package copy.

## Notes For Git Backup

Commit this folder as the canonical addon source:
- lod_classifier/

You can zip this folder for Blender installation if needed.
