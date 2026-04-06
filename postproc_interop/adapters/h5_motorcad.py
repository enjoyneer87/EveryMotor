from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from postproc_interop.adapters.base import PostprocAdapter
from postproc_interop.conversions import build_solver_metadata
from postproc_interop.model import (
    FieldObservation,
    MeshFrame,
    MeshTopology,
    NodeTable,
    RegionTable,
)


class MotorCADH5Adapter(PostprocAdapter):
    """Read MotorCAD-like h5 postprocess outputs into MeshFrame."""

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in {".h5", ".hdf5"}

    def read(self, path: Path) -> MeshFrame:
        with h5py.File(path, "r") as f:
            topology = MeshTopology(
                tri_index=np.asarray(f["mesh/tri_index"][:]),
                node_1=np.asarray(f["mesh/node_1"][:]),
                node_2=np.asarray(f["mesh/node_2"][:]),
                node_3=np.asarray(f["mesh/node_3"][:]),
                reg_code=np.asarray(f["mesh/reg_code"][:]),
            )
            nodes = NodeTable(
                node_id=np.asarray(f["mesh/node_id"][:]),
                x_mm=np.asarray(f["mesh/node_x_mm"][:]),
                y_mm=np.asarray(f["mesh/node_y_mm"][:]),
            )

            regions = None
            if "regions/reg_code" in f and "regions/name" in f:
                regions = RegionTable(
                    reg_code=np.asarray(f["regions/reg_code"][:]),
                    name=np.asarray(f["regions/name"][:]),
                )

            n_nodes = int(nodes.node_id.shape[0])
            field_observations: dict[str, FieldObservation] = {}
            for key in ("a", "bx", "by", "j"):
                h5_key = f"fields/{key}"
                if h5_key in f:
                    values = np.asarray(f[h5_key][:])
                    association = (
                        "node"
                        if (
                            values.ndim >= 1
                            and int(values.shape[-1]) == n_nodes
                        )
                        else "element"
                    )
                    field_observations[key] = FieldObservation(
                        name=key,
                        values=values,
                        association=association,
                        source_key=key,
                    )

            attrs = {"source_path": str(path)}
            step_index = -1
            if "steps" in f:
                attrs["steps"] = np.asarray(f["steps"][:])
                if int(attrs["steps"].size) > 0:
                    step_index = int(attrs["steps"].ravel()[0])

        return MeshFrame.from_components(
            topology=topology,
            nodes=nodes,
            regions=regions,
            field_observations=field_observations,
            solver_metadata=build_solver_metadata(
                source_path=path,
                step_index=step_index,
                reader_name=self.__class__.__name__,
            ),
            attrs=attrs,
        )
