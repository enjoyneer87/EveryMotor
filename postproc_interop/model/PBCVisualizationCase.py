from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PBCVisualizationCase:
    """Immutable payload for tutorial-side PBC boundary visualization."""

    case_idx: int
    source_file_name: str
    source_file_type: str
    step_index: int
    step_semantics: str
    coupling_policy: str
    pos_xy: np.ndarray
    triangles: np.ndarray
    reg_code: np.ndarray
    master_line_idx: np.ndarray
    slave_line_idx: np.ndarray
    pbc_forward_index: np.ndarray
    match_ratio: float
    master_line_groups: tuple[np.ndarray, ...] = field(default_factory=tuple)
    slave_line_groups: tuple[np.ndarray, ...] = field(default_factory=tuple)
    group_labels: tuple[str, ...] = field(default_factory=tuple)
    error_code: str = ""
    match_mode: str = ""
    max_rotation_residual: float = 0.0

    def __post_init__(self) -> None:
        if self.pos_xy.ndim != 2 or self.pos_xy.shape[1] != 2:
            raise ValueError(
                f"pos_xy must have shape [N, 2], got {self.pos_xy.shape}"
            )
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape [E, 3], got {self.triangles.shape}"
            )
        if self.reg_code.ndim != 1:
            raise ValueError(
                f"reg_code must be 1-D, got {self.reg_code.shape}"
            )
        if self.master_line_idx.ndim != 1:
            raise ValueError(
                "master_line_idx must be 1-D, "
                f"got {self.master_line_idx.shape}"
            )
        if self.slave_line_idx.ndim != 1:
            raise ValueError(
                f"slave_line_idx must be 1-D, got {self.slave_line_idx.shape}"
            )
        master_groups = self.master_line_groups or (self.master_line_idx,)
        slave_groups = self.slave_line_groups or (self.slave_line_idx,)
        if len(master_groups) != len(slave_groups):
            raise ValueError("master/slave group counts must match")
        labels = self.group_labels or tuple(
            f"group_{group_idx}"
            for group_idx in range(len(master_groups))
        )
        if len(labels) != len(master_groups):
            raise ValueError("group_labels length must match group count")
        if (
            self.pbc_forward_index.ndim != 2
            or self.pbc_forward_index.shape[0] != 2
        ):
            raise ValueError(
                "pbc_forward_index must have shape [2, P], "
                f"got {self.pbc_forward_index.shape}"
            )

        self.pos_xy.flags.writeable = False
        self.triangles.flags.writeable = False
        self.reg_code.flags.writeable = False
        self.master_line_idx.flags.writeable = False
        self.slave_line_idx.flags.writeable = False
        frozen_master_groups = tuple(
            np.asarray(group, dtype=np.int64)
            for group in master_groups
        )
        frozen_slave_groups = tuple(
            np.asarray(group, dtype=np.int64)
            for group in slave_groups
        )
        for group in frozen_master_groups + frozen_slave_groups:
            if group.ndim != 1:
                raise ValueError("boundary groups must be 1-D")
            group.flags.writeable = False
        object.__setattr__(self, "master_line_groups", frozen_master_groups)
        object.__setattr__(self, "slave_line_groups", frozen_slave_groups)
        object.__setattr__(self, "group_labels", tuple(str(label) for label in labels))
        self.pbc_forward_index.flags.writeable = False

    def summary(self) -> dict[str, object]:
        return {
            "case_idx": int(self.case_idx),
            "source_file_name": self.source_file_name,
            "source_file_type": self.source_file_type,
            "step_index": int(self.step_index),
            "step_semantics": self.step_semantics,
            "coupling_policy": self.coupling_policy,
            "pair_count": int(self.pbc_forward_index.shape[1]),
            "group_count": int(len(self.group_labels)),
            "master_line_count": int(self.master_line_idx.size),
            "slave_line_count": int(self.slave_line_idx.size),
            "match_ratio": float(self.match_ratio),
            "error_code": self.error_code,
            "match_mode": self.match_mode,
            "max_rotation_residual": float(self.max_rotation_residual),
        }
