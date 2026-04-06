from __future__ import annotations

from contextlib import suppress
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from postproc_interop.adapters.MotorCADMeshSolutionH5Reader import (
    MotorCADMeshSolutionH5Reader,
)
from postproc_interop.adapters.MotorCADMeshSolutionH5PyMCADReader import (
    MotorCADMeshSolutionH5PyMCADReader,
)
from postproc_interop.conversions import (
    build_solver_metadata,
    classify_motorcad_h5_filename,
    meshsolution_to_meshframe,
)
from postproc_interop.model.MeshSolution import MeshSolution
from postproc_interop.model.SolverMetadata import SolverMetadata
from postproc_interop.model.PBCVisualizationCase import PBCVisualizationCase
from postproc_interop.pbc import (
    build_pbc_boundary_candidate,
    build_pbc_pair_set,
    build_pbc_visualization_case,
)


class MotorCADPBCVisualizationBridge:
    """Build tutorial-ready PBC visualization payloads from DOE H5 cases."""

    def __init__(
        self,
        root_dir: Path,
        *,
        reader: MotorCADMeshSolutionH5PyMCADReader | None = None,
        rotation_deg: float = -45.0,
        anti_periodic: bool = True,
        atol_mm: float = 5e-2,
        rtol: float = 1e-6,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.reader = reader or MotorCADMeshSolutionH5PyMCADReader(
            repo_root=self.root_dir,
        )
        self._fallback_reader = MotorCADMeshSolutionH5Reader()
        self.rotation_deg = float(rotation_deg)
        self.anti_periodic = bool(anti_periodic)
        self.atol_mm = float(atol_mm)
        self.rtol = float(rtol)

    def build_case_overview(
        self,
        *,
        case_indices: Sequence[int],
        source_file_types: Sequence[str],
        data_dir: Path | None = None,
        output_dir: Path | None = None,
        figure_path: Path | None = None,
    ) -> dict[str, Any]:
        cases, skipped = self.collect_cases(
            case_indices=case_indices,
            source_file_types=source_file_types,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        summary = self.plot_case_overview(cases=cases, figure_path=figure_path)
        summary["skipped_cases"] = skipped
        return summary

    def collect_cases(
        self,
        *,
        case_indices: Sequence[int],
        source_file_types: Sequence[str],
        data_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> tuple[tuple[PBCVisualizationCase, ...], list[dict[str, Any]]]:
        data_root = (
            Path(data_dir)
            if data_dir is not None
            else self.root_dir / "doe_data"
        )
        manifest_path = data_root / "doe_manifest.json"
        output_root = (
            Path(output_dir)
            if output_dir is not None
            else self.root_dir / "results" / "pbc_case_vis"
        )
        output_root.mkdir(parents=True, exist_ok=True)

        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

        source_type_set = set(source_file_types)
        case_map = {
            int(case.get("index", -1)): case
            for case in manifest.get("cases", [])
            if int(case.get("index", -1)) >= 0
        }

        built_cases: list[PBCVisualizationCase] = []
        skipped_cases: list[dict[str, Any]] = []

        for case_idx in case_indices:
            case = case_map.get(int(case_idx))
            if case is None:
                skipped_cases.append(
                    {"case_idx": int(case_idx), "status": "case_not_found"}
                )
                continue

            selected_ref = None
            selected_type = None
            for h5_ref in case.get("h5_paths", []) or []:
                source_type = classify_motorcad_h5_filename(str(h5_ref))
                if source_type in source_type_set:
                    selected_ref = str(h5_ref)
                    selected_type = source_type
                    break

            if selected_ref is None or selected_type is None:
                skipped_cases.append(
                    {
                        "case_idx": int(case_idx),
                        "status": "source_type_not_found",
                    }
                )
                continue

            h5_path = self._resolve_h5_candidate(
                data_root,
                int(case_idx),
                selected_ref,
            )
            source_file_name = Path(selected_ref.replace("\\", "/")).name
            if h5_path is None:
                skipped_cases.append(
                    {
                        "case_idx": int(case_idx),
                        "status": "h5_not_found",
                        "source_file_name": source_file_name,
                    }
                )
                continue

            case_payload, error = self._build_case_payload(
                case_idx=int(case_idx),
                source_file_name=source_file_name,
                source_file_type=selected_type,
                h5_path=h5_path,
            )
            if case_payload is None:
                skipped_cases.append(
                    {
                        "case_idx": int(case_idx),
                        "status": error or "case_build_failed",
                        "source_file_name": source_file_name,
                    }
                )
                continue

            self._save_case_npz(case_payload, output_root)
            built_cases.append(case_payload)

        return tuple(built_cases), skipped_cases

    def plot_case_overview(
        self,
        *,
        cases: Sequence[PBCVisualizationCase],
        figure_path: Path | None = None,
    ) -> dict[str, Any]:
        if not cases:
            raise RuntimeError("시각화 가능한 케이스가 없습니다.")

        png_path = (
            Path(figure_path)
            if figure_path is not None
            else self.root_dir
            / "logs"
            / "tutorial"
            / "13_two_case_pbc_visualization.png"
        )
        png_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            1,
            len(cases),
            figsize=(8 * len(cases), 7),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for axis, case in zip(axes_flat, cases):
            self._plot_case_axis(axis, case)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle

        fig.suptitle(
            "Two DOE Cases: Boundary Lines With Actual PBC Edges",
            fontsize=14,
            y=0.98,
        )
        fig.text(
            0.5,
            0.93,
            (
                "Blue = extracted master boundary line, "
                "Red = extracted slave boundary line, "
                "Purple = actual pbc_edge overlay."
            ),
            ha="center",
            va="center",
            fontsize=10,
        )
        if unique:
            fig.legend(
                list(unique.values()),
                list(unique.keys()),
                loc="lower center",
                ncol=min(4, len(unique)),
                frameon=False,
                bbox_to_anchor=(0.5, 0.02),
            )

        plt.tight_layout(rect=[0, 0.08, 1, 0.9])
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        return {
            "saved_png": str(png_path),
            "case_indices": [int(case.case_idx) for case in cases],
            "source_file_types": sorted(
                {case.source_file_type for case in cases}
            ),
            "pairs": {
                str(case.case_idx): int(case.pbc_forward_index.shape[1])
                for case in cases
            },
            "match_ratios": {
                str(case.case_idx): round(float(case.match_ratio), 4)
                for case in cases
            },
        }

    def build_case_selector_widget(
        self,
        *,
        cases: Sequence[PBCVisualizationCase],
        default_group: str = "all",
        default_show_nodes: bool = False,
        default_show_pbc_edges: bool = False,
    ) -> Any:
        try:
            import ipywidgets as widgets
            from IPython.display import clear_output
        except ImportError as exc:
            raise ImportError(
                "ipywidgets and IPython are required for the PBC GUI view"
            ) from exc

        if not cases:
            raise RuntimeError("시각화 가능한 케이스가 없습니다.")

        case_lut = {int(case.case_idx): case for case in cases}
        case_options = [
            (
                f"case {case.case_idx:04d} | {case.source_file_name}",
                int(case.case_idx),
            )
            for case in cases
        ]
        case_dropdown = widgets.Dropdown(
            options=case_options,
            value=case_options[0][1],
            description="Case",
            layout=widgets.Layout(width="420px"),
        )
        group_dropdown = widgets.Dropdown(
            description="Group",
            layout=widgets.Layout(width="220px"),
        )
        show_nodes = widgets.Checkbox(
            value=bool(default_show_nodes),
            description="Show nodes",
        )
        show_master = widgets.Checkbox(value=True, description="Master")
        show_slave = widgets.Checkbox(value=True, description="Slave")
        show_pbc_edges = widgets.Checkbox(
            value=bool(default_show_pbc_edges),
            description="Purple PBC edges",
        )
        output = widgets.Output()

        def _group_options(
            case: PBCVisualizationCase,
        ) -> list[tuple[str, str]]:
            options = [("all", "all")]
            options.extend(
                (str(label), str(label))
                for label in (case.group_labels or tuple())
            )
            return options

        def _selected_groups(case: PBCVisualizationCase) -> set[str] | None:
            value = str(group_dropdown.value)
            if value == "all":
                return None
            labels = set(
                str(label) for label in (case.group_labels or tuple())
            )
            return {value} if value in labels else None

        def _refresh(*_args: Any) -> None:
            case = case_lut[int(case_dropdown.value)]
            with output:
                clear_output(wait=True)
                fig, axis = plt.subplots(1, 1, figsize=(8, 8))
                self._plot_case_axis(
                    axis,
                    case,
                    selected_groups=_selected_groups(case),
                    show_nodes=bool(show_nodes.value),
                    show_master=bool(show_master.value),
                    show_slave=bool(show_slave.value),
                    show_pbc_edges=bool(show_pbc_edges.value),
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig)

        def _sync_group_options(*_args: Any) -> None:
            case = case_lut[int(case_dropdown.value)]
            options = _group_options(case)
            group_dropdown.options = options
            available = {value for _, value in options}
            group_dropdown.value = (
                default_group if default_group in available else "all"
            )
            _refresh()

        case_dropdown.observe(_sync_group_options, names="value")
        group_dropdown.observe(_refresh, names="value")
        show_nodes.observe(_refresh, names="value")
        show_master.observe(_refresh, names="value")
        show_slave.observe(_refresh, names="value")
        show_pbc_edges.observe(_refresh, names="value")

        _sync_group_options()
        controls = widgets.HBox(
            [
                case_dropdown,
                group_dropdown,
                show_nodes,
                show_master,
                show_slave,
                show_pbc_edges,
            ]
        )
        return widgets.VBox([controls, output])

    def _build_case_payload(
        self,
        *,
        case_idx: int,
        source_file_name: str,
        source_file_type: str,
        h5_path: Path,
    ) -> tuple[PBCVisualizationCase | None, str | None]:
        mesh_solution: MeshSolution | None = None
        with suppress(
            FileNotFoundError,
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            mesh_solution = self.reader.read(h5_path)
        if mesh_solution is not None:
            return self._build_case_from_mesh_solution(
                case_idx=case_idx,
                source_file_name=source_file_name,
                source_file_type=source_file_type,
                source_path=h5_path,
                mesh_solution=mesh_solution,
            )
        return self._build_case_from_h5_reader(
            case_idx=case_idx,
            source_file_name=source_file_name,
            source_file_type=source_file_type,
            h5_path=h5_path,
        )

    def _build_case_from_mesh_solution(
        self,
        *,
        case_idx: int,
        source_file_name: str,
        source_file_type: str,
        source_path: Path,
        mesh_solution: MeshSolution,
    ) -> tuple[PBCVisualizationCase | None, str | None]:
        metadata = self._build_solver_metadata(
            mesh_solution=mesh_solution,
            source_path=source_path,
            source_file_name=source_file_name,
            source_file_type=source_file_type,
        )
        mesh_frame = meshsolution_to_meshframe(
            mesh_solution,
            solver_metadata=metadata,
        )
        return self._assemble_case(
            case_idx=case_idx,
            mesh_frame=mesh_frame,
        )

    def _build_case_from_record(
        self,
        *,
        case_idx: int,
        source_file_name: str,
        source_file_type: str,
        h5_path: Path,
    ) -> tuple[PBCVisualizationCase | None, str | None]:
        raise NotImplementedError()

    def _build_case_from_h5_reader(
        self,
        *,
        case_idx: int,
        source_file_name: str,
        source_file_type: str,
        h5_path: Path,
    ) -> tuple[PBCVisualizationCase | None, str | None]:
        mesh_solution = self._fallback_reader.read(h5_path)
        return self._build_case_from_mesh_solution(
            case_idx=case_idx,
            source_file_name=source_file_name,
            source_file_type=source_file_type,
            source_path=h5_path,
            mesh_solution=mesh_solution,
        )

    def _assemble_case(
        self,
        *,
        case_idx: int,
        mesh_frame,
    ) -> tuple[PBCVisualizationCase | None, str | None]:
        boundary_candidate, boundary_error = build_pbc_boundary_candidate(
            mesh_frame,
            rotation_deg=self.rotation_deg,
        )
        if boundary_candidate is None:
            return None, boundary_error or "chain_extract_failed"

        pair_set, pair_error = build_pbc_pair_set(
            mesh_frame,
            boundary_candidate,
            anti_periodic=self.anti_periodic,
            atol_mm=self.atol_mm,
            rtol=self.rtol,
        )
        if pair_set is None:
            return None, pair_error or "pbc_match_empty"

        return (
            build_pbc_visualization_case(
                case_idx=case_idx,
                mesh_frame=mesh_frame,
                boundary_candidate=boundary_candidate,
                pair_set=pair_set,
            ),
            None,
        )

    def _resolve_h5_candidate(
        self,
        data_dir: Path,
        case_idx: int,
        h5_ref: str,
    ) -> Path | None:
        basename = h5_ref.replace("\\", "/").split("/")[-1]
        candidates = [
            Path(h5_ref),
            data_dir / f"case_{case_idx:04d}" / "postproc" / basename,
            data_dir / basename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_solver_metadata(
        self,
        *,
        mesh_solution: MeshSolution,
        source_path: Path,
        source_file_name: str,
        source_file_type: str,
    ) -> SolverMetadata:
        step_index = int(mesh_solution.attrs.get("step", -1))
        if step_index < 0:
            steps = mesh_solution.attrs.get("steps")
            if isinstance(steps, np.ndarray) and steps.size > 0:
                step_index = int(steps.ravel()[0])
        reader_name = str(mesh_solution.attrs.get("reader", ""))
        return build_solver_metadata(
            source_path=source_path,
            source_file_name=source_file_name,
            fidelity_type=source_file_type,
            step_index=step_index,
            reader_name=reader_name,
        )

    def _save_case_npz(
        self,
        case: PBCVisualizationCase,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"case_{case.case_idx:04d}_pbc_vis.npz"
        np.savez_compressed(
            output_path,
            pos=case.pos_xy,
            triangles=case.triangles,
            reg_code=case.reg_code,
            master_line_idx=case.master_line_idx,
            slave_line_idx=case.slave_line_idx,
            master_group_sizes=np.asarray(
                [group.size for group in case.master_line_groups],
                dtype=np.int64,
            ),
            slave_group_sizes=np.asarray(
                [group.size for group in case.slave_line_groups],
                dtype=np.int64,
            ),
            group_labels=np.asarray(case.group_labels, dtype=object),
            pbc_forward_index=case.pbc_forward_index,
            match_ratio=np.asarray([case.match_ratio], dtype=np.float32),
            error_code=np.asarray([case.error_code]),
            match_mode=np.asarray([case.match_mode]),
            max_rotation_residual=np.asarray(
                [case.max_rotation_residual],
                dtype=np.float32,
            ),
            step_index=np.asarray([case.step_index], dtype=np.int64),
            source_file_name=np.asarray([case.source_file_name]),
            source_file_type=np.asarray([case.source_file_type]),
            step_semantics=np.asarray([case.step_semantics]),
            coupling_policy=np.asarray([case.coupling_policy]),
        )
        return output_path

    def _plot_case_axis(
        self,
        axis: Any,
        case: PBCVisualizationCase,
        *,
        selected_groups: set[str] | None = None,
        show_nodes: bool = True,
        show_master: bool = True,
        show_slave: bool = True,
        show_pbc_edges: bool = True,
    ) -> None:
        pos = case.pos_xy
        master_groups = case.master_line_groups or (case.master_line_idx,)
        slave_groups = case.slave_line_groups or (case.slave_line_idx,)
        group_labels = case.group_labels or tuple(
            f"group_{group_idx}"
            for group_idx in range(len(master_groups))
        )

        if show_nodes:
            axis.scatter(
                pos[:, 0],
                pos[:, 1],
                s=8,
                c="#C7CDD4",
                label="All nodes",
                zorder=1,
            )

        selected_master_nodes: set[int] = set()
        selected_slave_nodes: set[int] = set()
        for group_idx, (label, group) in enumerate(
            zip(group_labels, master_groups)
        ):
            if (
                selected_groups is not None
                and str(label) not in selected_groups
            ):
                continue
            selected_master_nodes.update(
                int(node_id) for node_id in group.tolist()
            )
            if not show_master:
                continue
            if group.size > 1:
                axis.plot(
                    pos[group, 0],
                    pos[group, 1],
                    color="#1565C0",
                    linewidth=3.0,
                    marker="o",
                    markersize=2.8,
                    label=(
                        "Master boundary line"
                        if group_idx == 0 else None
                    ),
                    zorder=4,
                )
            elif group.size == 1:
                axis.scatter(
                    pos[group, 0],
                    pos[group, 1],
                    s=32,
                    c="#1565C0",
                    label=(
                        "Master boundary line"
                        if group_idx == 0 else None
                    ),
                    zorder=4,
                )

        for group_idx, (label, group) in enumerate(
            zip(group_labels, slave_groups)
        ):
            if (
                selected_groups is not None
                and str(label) not in selected_groups
            ):
                continue
            selected_slave_nodes.update(
                int(node_id) for node_id in group.tolist()
            )
            if not show_slave:
                continue
            if group.size > 1:
                axis.plot(
                    pos[group, 0],
                    pos[group, 1],
                    color="#D32F2F",
                    linewidth=3.0,
                    marker="o",
                    markersize=2.8,
                    label=(
                        "Slave boundary line"
                        if group_idx == 0 else None
                    ),
                    zorder=4,
                )
            elif group.size == 1:
                axis.scatter(
                    pos[group, 0],
                    pos[group, 1],
                    s=32,
                    c="#D32F2F",
                    label=(
                        "Slave boundary line"
                        if group_idx == 0 else None
                    ),
                    zorder=4,
                )

        if show_pbc_edges:
            first_edge = True
            for master_node, slave_node in case.pbc_forward_index.T.tolist():
                if selected_groups is not None:
                    if int(master_node) not in selected_master_nodes:
                        continue
                    if int(slave_node) not in selected_slave_nodes:
                        continue
                axis.plot(
                    [pos[master_node, 0], pos[slave_node, 0]],
                    [pos[master_node, 1], pos[slave_node, 1]],
                    color="#7B1FA2",
                    alpha=0.55,
                    linewidth=1.0,
                    label=("Actual pbc_edge" if first_edge else None),
                    zorder=3,
                )
                first_edge = False

        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"case {case.case_idx:04d}", fontsize=12, pad=12)
        axis.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"file: {case.source_file_name}",
                    (
                        f"step={case.step_index} | "
                        f"pair_count={case.pbc_forward_index.shape[1]}"
                    ),
                    (
                        f"groups={len(case.group_labels)} | "
                        f"master_nodes={case.master_line_idx.size} | "
                        f"slave_nodes={case.slave_line_idx.size}"
                    ),
                    f"match_ratio={case.match_ratio:.2f}",
                    f"error={case.error_code or 'OK'}",
                ]
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "#B0BEC5",
            },
            zorder=5,
        )
