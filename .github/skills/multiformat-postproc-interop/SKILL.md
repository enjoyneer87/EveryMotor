---
name: multiformat-postproc-interop
description: "Use when: MotorCAD postprocess interoperability across h5/txt/vtu/pyvista/babylon.js is needed; building reusable adapters, format contracts, and export bridges for future visualization pipelines."
---

# Multiformat Postprocess Interop Skill

## Purpose
- Build and maintain a reusable interoperability layer for MotorCAD postprocess outputs.
- Start from h5/txt today, but keep a stable contract so vtu, PyVista, and Babylon.js can be added without breaking downstream code.

## Use When
- You need to parse MotorCAD outputs and normalize them into one internal data model.
- You want compatible exports to VTK/VTU, PyVista objects, or Babylon.js-friendly JSON.
- You want to avoid hard-coding one format in notebooks and training utilities.

## Core Design Rules
1. Keep one canonical in-memory contract (`MeshFrame`) and map all formats to/from it.
2. Isolate file-format specifics in adapters (`adapters/*.py`).
3. Isolate visualization/export specifics in exporters (`exporters/*.py`).
4. Use lazy optional imports for heavy dependencies (pyvista, meshio, vtk).
5. Preserve unit metadata and field semantics (`A`, `Bx`, `By`, `J`) explicitly.

## Recommended Workflow
1. Define/extend schema in `postproc_interop/model.py`.
2. Implement read adapter (`adapters/*`) for source format.
3. Validate required keys and attach warnings for partial data.
4. Implement export bridge (`exporters/*`) from canonical schema.
5. Add a thin CLI path in `postproc_interop/cli.py` for repeatable conversions.

## Expected Outputs
- Canonical mesh+field object from source file.
- Optional exports:
  - VTU file path
  - PyVista `PolyData`/`UnstructuredGrid`
  - Babylon.js JSON payload for web visualization

## Dependency Policy
- Base: `numpy`, `h5py`
- Optional: `meshio`, `pyvista`, `vtk`
- If optional package is missing, fail with actionable message and keep base path usable.

## Suggested Commands
- Parse and summarize:
  - `python -m postproc_interop.cli summary --input <path.h5>`
- Convert to Babylon JSON:
  - `python -m postproc_interop.cli export-babylon --input <path.h5> --output <out.json>`
- Convert to VTU (adapter/exporter implemented):
  - `python -m postproc_interop.cli export-vtu --input <path.h5> --output <out.vtu>`
