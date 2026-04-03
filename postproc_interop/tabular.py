from __future__ import annotations

from pathlib import Path
from typing import Any

from postproc_interop.adapters.MotorCADMeshSolutionH5Reader import MotorCADMeshSolutionH5Reader
from postproc_interop.model.MeshMat import MeshMat
from postproc_interop.model.MeshSolution import MeshSolution


def _decode_region_name(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def meshsolution_to_dataframes(meshsol: MeshSolution):
    """Convert MeshSolution to nodetable/regiontable/elementtable DataFrames.

    Returns
    -------
    tuple
        (nodetable, regiontable, elementtable, elementtable_with_xy, stats)
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for tabular conversion") from exc

    m: MeshMat = meshsol.mesh

    nodetable = pd.DataFrame({
        "NodeIndex": m.node_id,
        "X": m.x_mm,
        "Y": m.y_mm,
    })

    regiontable = None
    if m.region_code is not None and m.region_name is not None:
        regiontable = pd.DataFrame({
            "RegionCode": m.region_code,
            "RegionName": [_decode_region_name(v) for v in m.region_name],
        })

    elementtable = pd.DataFrame({
        "TriIndex": m.tri_index,
        "Node1": m.node_1,
        "Node2": m.node_2,
        "Node3": m.node_3,
        "RegCode": m.reg_code,
    })

    if not (len(elementtable["TriIndex"])
            == len(elementtable["Node1"])
            == len(elementtable["Node2"])
            == len(elementtable["Node3"])
            == len(elementtable["RegCode"])):
        raise ValueError("Element connectivity arrays must have the same length")

    node_index_set = set(nodetable["NodeIndex"].tolist())
    missing_n1 = int((~elementtable["Node1"].isin(node_index_set)).sum())
    missing_n2 = int((~elementtable["Node2"].isin(node_index_set)).sum())
    missing_n3 = int((~elementtable["Node3"].isin(node_index_set)).sum())

    node_xy = nodetable[["NodeIndex", "X", "Y"]].copy()
    elementtable_with_xy = (
        elementtable
        .merge(node_xy.rename(columns={"NodeIndex": "Node1", "X": "Node1_X", "Y": "Node1_Y"}),
               on="Node1", how="left")
        .merge(node_xy.rename(columns={"NodeIndex": "Node2", "X": "Node2_X", "Y": "Node2_Y"}),
               on="Node2", how="left")
        .merge(node_xy.rename(columns={"NodeIndex": "Node3", "X": "Node3_X", "Y": "Node3_Y"}),
               on="Node3", how="left")
    )

    stats = {"missing_n1": missing_n1, "missing_n2": missing_n2, "missing_n3": missing_n3}
    return nodetable, regiontable, elementtable, elementtable_with_xy, stats


def load_h5_as_tables(h5_path: str | Path):
    """Read MotorCAD H5 and return tabular views.

    Returns
    -------
    tuple
        (meshsol, nodetable, regiontable, elementtable, elementtable_with_xy, stats)
    """
    meshsol = MotorCADMeshSolutionH5Reader().read(Path(h5_path))
    nodetable, regiontable, elementtable, elementtable_with_xy, stats = (
        meshsolution_to_dataframes(meshsol)
    )
    return meshsol, nodetable, regiontable, elementtable, elementtable_with_xy, stats


# ---------------------------------------------------------------------------
# Legacy alias — keeps old call sites working during migration
# ---------------------------------------------------------------------------
def meshframe_to_dataframes(frame):
    """Backward-compat wrapper: accepts MeshSolution or old MeshFrame."""
    return meshsolution_to_dataframes(frame)
