from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from postproc_interop.model import MeshFrame


def _decode_region_name(v: object) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    return str(v)


def export_babylon_meshframe(frame: MeshFrame, output_path: Path) -> Path:
    """Export MeshFrame to Babylon.js-friendly JSON payload.

    This writes a neutral JSON schema consumable by Babylon.js loader code.
    """

    x = frame.nodes.x_mm.astype(np.float32)
    y = frame.nodes.y_mm.astype(np.float32)
    z = np.zeros_like(x)

    positions = np.column_stack([x, y, z]).reshape(-1).tolist()

    n1 = frame.topology.node_1.astype(np.int64)
    n2 = frame.topology.node_2.astype(np.int64)
    n3 = frame.topology.node_3.astype(np.int64)
    indices = np.column_stack([n1, n2, n3]).reshape(-1).tolist()

    payload: dict[str, object] = {
        "meta": {
            "format": "MeshFrame->BabylonJSON",
            "source": frame.attrs.get("source_path", ""),
        },
        "mesh": {
            "positions": positions,
            "indices": indices,
        },
        "fields": {k: np.asarray(v).tolist() for k, v in frame.fields.items()},
    }

    if frame.regions is not None:
        payload["regions"] = {
            "reg_code": frame.regions.reg_code.astype(np.int64).tolist(),
            "name": [
                _decode_region_name(v) for v in frame.regions.name.tolist()
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
