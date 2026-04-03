from __future__ import annotations

import numpy as np

from postproc_interop.model import MeshFrame


def to_pyvista_unstructured_grid(frame: MeshFrame):
    """Convert MeshFrame to PyVista UnstructuredGrid.

    Returns a pyvista.UnstructuredGrid. Requires optional dependency pyvista.
    """

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for to_pyvista_unstructured_grid. "
            "Install with `pip install pyvista vtk`."
        ) from exc

    x = frame.nodes.x_mm.astype(np.float32)
    y = frame.nodes.y_mm.astype(np.float32)
    z = np.zeros_like(x)
    points = np.column_stack([x, y, z])

    n1 = frame.topology.node_1.astype(np.int64)
    n2 = frame.topology.node_2.astype(np.int64)
    n3 = frame.topology.node_3.astype(np.int64)

    # VTK cell format: [num_points, i0, i1, i2, num_points, ...]
    tri = np.column_stack([np.full_like(n1, 3), n1, n2, n3]).reshape(-1)
    celltypes = np.full(n1.shape[0], pv.CellType.TRIANGLE, dtype=np.uint8)

    grid = pv.UnstructuredGrid(tri, celltypes, points)

    for key, val in frame.fields.items():
        arr = np.asarray(val)
        if arr.ndim == 1 and arr.shape[0] == points.shape[0]:
            grid.point_data[key] = arr

    return grid
