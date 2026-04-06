from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PBCBoundaryCandidate:
    """Immutable extracted master/slave boundary line candidate."""

    rotation_origin_xy: np.ndarray
    master_line_idx: np.ndarray
    slave_line_idx: np.ndarray
    rotation_deg: float
    master_line_groups: tuple[np.ndarray, ...] = field(default_factory=tuple)
    slave_line_groups: tuple[np.ndarray, ...] = field(default_factory=tuple)
    group_labels: tuple[str, ...] = field(default_factory=tuple)
    extraction_method: str = ""
    error_code: str = ""

    def __post_init__(self) -> None:
        if self.rotation_origin_xy.shape != (2,):
            raise ValueError(
                "rotation_origin_xy must have shape [2], "
                f"got {self.rotation_origin_xy.shape}"
            )
        if self.master_line_idx.ndim != 1:
            raise ValueError("master_line_idx must be 1-D")
        if self.slave_line_idx.ndim != 1:
            raise ValueError("slave_line_idx must be 1-D")

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

        self.rotation_origin_xy.flags.writeable = False
        self.master_line_idx.flags.writeable = False
        self.slave_line_idx.flags.writeable = False
