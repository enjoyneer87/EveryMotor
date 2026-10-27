"""Read-only Elmer ASCII3/P1 reintegration. No solver, network, or training calls.

Uses native.result's explicit node IDs, never nearest-coordinate recovery.
Only the 2-D, 36-copper-region, BE/BDF2, initial-A=0 acquisition is supported.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import re
import struct
import xml.etree.ElementTree as ET
import numpy as np


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def square_integral(area, nodal):
    """Exact integral of the square of a linear field on a triangle."""
    return area / 12 * (nodal.sum(axis=1) ** 2 + (nodal ** 2).sum(axis=1))


def native_frames(path):
    """ASCII3 values are expanded nodal values in the explicit Perm row order.

    The first Perm column is original node ID; the second identifies constrained
    DOFs. Reading and checking both on every frame preserves duplicate-DOF nodes.
    This reader uses copper only: native values at antiperiodic outer/rotor nodes
    may need a sign transformation. Copper values are cross-checked with VTU.
    """
    with Path(path).open("rb") as f:
        assert f.readline().strip() == b"ASCII 3", "Only inspected ASCII3 supported"
        first_perm = None
        for line in f:
            if not line.startswith(b"Time:"):
                continue
            _, step, _, stamp = line.split()
            assert f.readline().strip() == b"a"
            pline = f.readline().split()
            assert pline[0] == b"Perm:" and len(pline) == 3
            nperm, nvalues = map(int, pline[1:])
            assert nperm == nvalues
            perm_bytes = b"".join(itertools.islice(f, nperm))
            perm = np.fromstring(perm_bytes.decode("ascii"), sep=" ", dtype=np.int64).reshape(nperm, 2)
            assert len(np.unique(perm[:, 0])) == nperm and np.all(perm[:, 0] > 0)
            if first_perm is None:
                first_perm = perm.copy()
            assert np.array_equal(perm, first_perm), "Changing native node/DOF mapping"
            a = np.fromstring(b"".join(itertools.islice(f, nvalues)).decode("ascii"), sep=" ")
            assert len(a) == nvalues and np.isfinite(a).all()
            yield int(step), float(stamp), perm, a


def vtu_arrays(path):
    data = Path(path).read_bytes()
    stop = data.index(b"<AppendedData")
    root = ET.fromstring(data[:stop] + b"</VTKFile>")
    assert root.attrib["byte_order"] == "LittleEndian" and "compressor" not in root.attrib
    assert root.attrib.get("header_type", "UInt32") == "UInt32"
    start = data.index(b"_", data.index(b">", stop)) + 1
    def arr(e):
        offset = start + int(e.attrib["offset"])
        n = struct.unpack_from("<I", data, offset)[0]
        dtype = {"Float64": "<f8", "Int32": "<i4"}[e.attrib["type"]]
        a = np.frombuffer(data, dtype=dtype, count=n // np.dtype(dtype).itemsize, offset=offset + 4)
        return a.reshape(-1, int(e.attrib.get("NumberOfComponents", 1)))
    out = {e.attrib.get("Name", "points"): arr(e) for e in root.findall(".//DataArray")}
    for key, label in [("version", "version"), ("revision", "revision")]:
        match = re.search(("Elmer " + label + r": ([^<]+) -->").encode(), data[:stop])
        out[key] = match.group(1).decode().strip() if match else None
    return out


def audit(raw, expected, out, full_history_out=None):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    params, model, contract = [read_json(raw / p) for p in ["PARAMETERS.json", "MODEL.json", "INPUT_CONTRACT.json"]]
    archive_files = {r["path"]: r for r in expected["archive_files"]}
    consumed = []
    def checked(rel):
        p = raw / rel
        observed = sha(p)
        want = archive_files.get(rel, {}).get("sha256")
        item = dict(logical_source_key=f"full/{expected['case_id']}/raw/{rel}", bytes=p.stat().st_size,
                    sha256=observed, expected_sha256=want, status="PASS" if observed == want else "UNKNOWN" if want is None else "FAIL")
        consumed.append(item)
        if item["status"] == "FAIL":
            raise ValueError(f"Source hash mismatch: {rel}")
        return p
    for rel in ["PARAMETERS.json", "MODEL.json", "INPUT_CONTRACT.json", "case.sif", "circuits.dat", "circuits.dat.names",
                "circuits.definitions", "solver.log", "mesh/mesh.nodes", "mesh/mesh.elements", "mesh/mesh.boundary", "mesh/mesh.header"]:
        checked(rel)
    native = checked("mesh/native.result")
    source_stats = {r["logical_source_key"].split("/raw/", 1)[1]: ((raw / r["logical_source_key"].split("/raw/", 1)[1]).stat().st_size,
                   (raw / r["logical_source_key"].split("/raw/", 1)[1]).stat().st_mtime_ns) for r in consumed}
    mesh_nodes = np.loadtxt(raw / "mesh/mesh.nodes")
    node_ids = mesh_nodes[:, 0].astype(np.int64)
    assert len(node_ids) == len(np.unique(node_ids))
    node_lookup = {int(n): i for i, n in enumerate(node_ids)}
    all_elements = [list(map(int, line.split())) for line in (raw / "mesh/mesh.elements").read_text().splitlines()]
    cu = np.array([r for r in all_elements if r[2] == 303 and 2301 <= r[1] <= 2336], dtype=np.int64)
    elem_ids, physical, tri_ids = cu[:, 0], cu[:, 1], cu[:, 3:]
    assert len(np.unique(elem_ids)) == len(elem_ids)
    used_ids = np.unique(tri_ids)
    used_lookup = {int(n): i for i, n in enumerate(used_ids)}
    tri = np.array([[used_lookup[int(n)] for n in row] for row in tri_ids])
    xy = mesh_nodes[[node_lookup[int(n)] for n in used_ids], 2:4]
    xyz = xy[tri]
    u, v = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    area = np.abs(det) / 2
    assert np.all(area > 0)
    bar = physical - 2301
    bar_area = np.bincount(bar, weights=area, minlength=36)
    assert np.all(bar_area > 0)
    names = {m[2].strip(): int(m[1]) - 1 for m in re.finditer(r"^\s*(\d+):\s*(.*)$", (raw / "circuits.dat.names").read_text(), re.M)}
    circuits = np.loadtxt(raw / "circuits.dat").reshape(-1, len(names))
    components = sorted([r for r in contract["components"] if r["kind"] == "copper"], key=lambda c: c["physical"])
    assert [c["physical"] for c in components] == list(range(2301, 2337))
    sigma, length = params["sigma_cu"], model["axial_length_m"]
    n = params["NbSteps"]
    assert n == 256 and params["NbPeriods"] == 2 and len(circuits) == 512
    assert contract["initial_A"] == 0 and contract["circuit_model_depth_m"] == length
    sif = (raw / "case.sif").read_text(encoding="utf-8")
    assert re.search(r"BDF Order\s*=\s*2\b", sif)
    frequency = params["rpm"] * 4 / 60
    dt = 1 / frequency / n
    scale = 8 * length / sigma
    initial = np.zeros(len(used_ids)); last2 = initial.copy(); last = initial.copy()
    sample_steps = np.arange(288, 513, 32)  # eight complete P2 phase endpoints; not a substitute for P2 mean
    snapshots = {k: [] for k in ["A_node_Wb_per_m", "B_element_T", "J_total_element_node_Am2", "J_residual_element_node_Am2"]}
    scalar, currents, imposed_all, circuit_all, bmean, bsecond, bar_loss, times, a_history, voltages = [], [], [], [], [], [], [], [], [], []
    max_identity = 0.; map_checks = []; max_dt_relative = 0.; native_time_delta = 0.
    previous_time = 0.
    for step, native_time, perm, a_all in native_frames(native):
        assert step == len(times) + 1
        lookup = {int(nid): i for i, nid in enumerate(perm[:, 0])} if step == 1 else lookup
        which = np.array([lookup[int(nid)] for nid in used_ids]) if step == 1 else which
        a = a_all[which]
        data = circuits[step - 1]
        t = float(data[names["res: time"]]); times.append(t)
        max_dt_relative = max(max_dt_relative, abs((t - previous_time) / dt - 1))
        assert max_dt_relative < 1e-9
        native_time_delta = max(native_time_delta, abs(native_time - t))
        dadt = (a - last) / (t - previous_time) if step == 1 else (3 * a - 4 * last + last2) / (2 * (t - previous_time))
        voltage = np.array([data[names[f"res: v_component({c['component']})"]] for c in components])
        circuit_current = np.array([data[names[f"res: i_component({c['component']})"]] for c in components])
        imposed = np.array([c["amplitude_A"] * np.sin(2 * np.pi * frequency * t + np.deg2rad(c["phase_deg"])) for c in components])
        j = sigma * (voltage[bar, None] / length - dadt[tri])
        integrated = np.bincount(bar, weights=area * j.sum(axis=1) / 3, minlength=36)
        residual = j - (imposed / bar_area)[bar, None]
        total = scale * np.bincount(bar, weights=square_integral(area, j), minlength=36)
        dc = scale * imposed ** 2 / bar_area
        residual_loss = scale * np.bincount(bar, weights=square_integral(area, residual), minlength=36)
        cross = scale * 2 * imposed / bar_area * (integrated - imposed)
        max_identity = max(max_identity, float(np.max(np.abs(total - dc - residual_loss - cross))))
        delta = a[tri][:, 1:] - a[tri][:, :1]
        gx = (delta[:, 0] * v[:, 1] - delta[:, 1] * u[:, 1]) / det
        gy = (u[:, 0] * delta[:, 1] - v[:, 0] * delta[:, 0]) / det
        b = np.column_stack((gy, -gx))
        bm = np.array([np.bincount(bar, weights=area * b[:, k], minlength=36) / bar_area for k in [0, 1]]).T
        bs = np.array([np.bincount(bar, weights=area * vv, minlength=36) / bar_area for vv in [b[:, 0] ** 2, b[:, 1] ** 2, b[:, 0] * b[:, 1]]]).T
        err = 100 * np.abs(integrated - imposed) / (np.sqrt(2) * params["Irms"])
        scalar.append([total.sum(), dc.sum(), (total - dc).sum(), residual_loss.sum(), cross.sum(), err.max()])
        currents.append(integrated); imposed_all.append(imposed); circuit_all.append(circuit_current); voltages.append(voltage)
        bmean.append(bm); bsecond.append(bs); bar_loss.append(np.stack([total, dc, total - dc], axis=-1))
        if full_history_out is not None:
            a_history.append(a.copy())
        if step in sample_steps:
            for key, value in zip(snapshots, [a, b, j, residual]): snapshots[key].append(value.copy())
        if step in [1, 256, 512]:
            vp = raw / f"mesh/field_t{step:04d}.vtu"
            if vp.exists():
                checked(vp.relative_to(raw).as_posix()); vd = vtu_arrays(vp)
                # Explicit original mesh row/node ID mapping, checked by element connectivity and native A.
                vtk_ids = node_ids
                vlookup = {int(vv): i for i, vv in enumerate(vtk_ids)}
                conn = vd["connectivity"].ravel(); offsets = vd["offsets"].ravel()
                starts = np.r_[0, offsets[:-1]]; body_tags = vd["GeometryIds"].ravel(); types = vd["types"].ravel()
                body_to_physical = {int(body): int(ph) for ph, body in contract["body_id_by_physical"].items()}
                tags = np.array([body_to_physical.get(int(x), -1) for x in body_tags])
                found = {}
                for ci in np.flatnonzero((tags >= 2301) & (tags <= 2336) & (types == 5)):
                    ids = vtk_ids[conn[starts[ci]:offsets[ci]]]
                    found[(int(tags[ci]), tuple(sorted(map(int, ids))))] = True
                assert len(found) == len(cu) and all((int(ph), tuple(sorted(map(int, ns)))) in found for ph, ns in zip(physical, tri_ids))
                vi = np.array([vlookup[int(nid)] for nid in used_ids])
                aerr = float(np.max(np.abs(vd["a"].ravel()[vi] - a)))
                xyerr = float(np.max(np.abs(vd["points"][vi, :2] - xy)))
                assert aerr < 1e-14 and xyerr < 1e-11
                map_checks.append(dict(step=step, native_A_vs_VTU_max_abs=aerr, copper_coordinate_max_abs_m=xyerr,
                                       copper_element_connectivity_equal=True, method="native explicit node IDs + exact element connectivity; no nearest-neighbor mapping"))
                if vd["version"] is not None:
                    solver_version, solver_revision = vd["version"], vd["revision"]
        last2, last, previous_time = last, a, t
    assert len(times) == 512 and map_checks
    scalar = np.array(scalar); currents = np.array(currents); imposed_all = np.array(imposed_all)
    means = scalar[256:].mean(axis=0)
    errorA = currents - imposed_all
    ref_means = np.array([expected["P2_Cu_total_W"], expected["P2_DC_transport_W"], expected["P2_AC_excess_W"]])
    deltas = means[:3] - ref_means
    p2error = float(scalar[256:, 5].max()); wholeerror = float(scalar[:, 5].max())
    nl = []
    for fragment in re.split(r"MAIN: Time:", (raw / "solver.log").read_text(encoding="utf-8"))[1:]:
        vals = re.findall(r"ComputeChange: NS \(ITER=(\d+)\).*?\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s*\) :: mag", fragment)
        if vals: nl.append(float(vals[-1][2]))
    assert len(nl) == 512
    for rel, stamp in source_stats.items():
        assert ((raw / rel).stat().st_size, (raw / rel).stat().st_mtime_ns) == stamp, "Source changed during read"
    common = dict(node_ids=used_ids, node_xy_m=xy, element_ids=elem_ids, connectivity_node_ids=tri_ids,
                  connectivity_local=tri, physical=physical, bar_ids=np.arange(1, 37),
                  component_ids=np.array([c["component"] for c in components]), circuit_ids=np.array([c["circuit"] for c in components]),
                  slot_ids=(physical - 2301) // 6 + 1, conductor_in_slot=(physical - 2301) % 6 + 1,
                  element_area_m2=area, bar_area_m2=bar_area, sector_element_volume_m3=area * length,
                  body_ids=np.array([contract["body_id_by_physical"][str(int(p))] for p in physical]),
                  material_id=np.ones(len(physical),dtype=np.int32), material_names=np.array(["copper"]),
                  component_phase_deg=np.array([c["phase_deg"] for c in components]), symmetry_sectors=np.array(8),
                  active_length_m=np.array(length), sigma_S_m=np.array(sigma))
    snapshot_file = out / (expected["point_id"] + "_field_snapshots.npz")
    np.savez_compressed(snapshot_file, **common, steps=sample_steps, time_s=np.array(times)[sample_steps - 1],
                        electrical_phase_deg=(sample_steps % 256) * 360 / 256,
                        **{k: np.array(vv) for k, vv in snapshots.items()})
    waves_file = out / (expected["point_id"] + "_full_history_integrals.npz")
    np.savez_compressed(waves_file, steps=np.arange(1, 513), time_s=np.array(times), scalar=scalar,
                        current_integrated_A=currents, current_prescribed_A=imposed_all, current_circuit_A=np.array(circuit_all), component_voltage_V=np.array(voltages),
                        B_area_mean_T=np.array(bmean), B_area_second_moment_T2=np.array(bsecond), bar_loss_W=np.array(bar_loss))
    if full_history_out is not None:
        hp = Path(full_history_out); hp.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hp / (expected["point_id"] + "_copper_A_all512.npz"), **common,
                            A_node_Wb_per_m=np.array(a_history), time_s=np.array(times), initial_A_node_Wb_per_m=initial)
    sparse_mean = scalar[sample_steps - 1, :3].mean(axis=0)
    result = dict(case_id=expected["case_id"], point_id=expected["point_id"], model=model["machine"],
        steps=512, periods=2, dt_s=dt, P2_window_steps=[257, 512], P2_time_s=[times[256], times[-1]],
        P2_electrical_phase_unwrapped_deg=[361.40625, 720.0], P2_average="uniform post-step arithmetic mean",
        sigma_S_m=sigma, active_length_m=length, radial_factor=model["radial_factor"], symmetry_sectors=8,
        solver_version=solver_version, solver_revision=solver_revision, copper_nodes=len(used_ids), copper_elements=len(cu),
        source_contract_sha256=sha(raw / "INPUT_CONTRACT.json"), source_model_sha256=sha(raw / "MODEL.json"),
        quantities=["Joule_total_active_W", "prescribed_transport_DC_W", "AC_excess_W"],
        P2_reintegrated_W=means[:3].tolist(), P2_saved_W=ref_means.tolist(), delta_W=deltas.tolist(),
        relative_delta_percent=(100 * deltas / np.maximum(np.abs(ref_means), 1e-300)).tolist(),
        reconstruction_gate_W=1e-6, reconstruction_gate_source="pinned saved_elmer_fields.py main: abs(reintegrated_minus_summary_W) <= 1e-6",
        raw_loss_reintegration="PASS" if abs(deltas[2]) <= 1e-6 else "FAIL",
        current_gate_percent=0.001, current_gate_denominator="prescribed phase peak sqrt(2)*Irms, not instantaneous current",
        P2_max_current_error_percent=p2error, whole_max_current_error_percent=wholeerror,
        P2_current_status="PASS" if p2error <= 0.001 else "FAIL", whole_current_status="PASS" if wholeerror <= 0.001 else "FAIL",
        current_absolute_error_A=dict(max=float(np.abs(errorA).max()), mean=float(np.abs(errorA).mean()), P2max=float(np.abs(errorA[256:]).max())),
        field_vs_circuit_max_A=float(np.abs(currents - np.array(circuit_all)).max()),
        scalar_current_error_delta_percent=p2error - expected["P2_max_current_error_percent"],
        nonlinear_status="PASS" if max(nl) <= params["stop_criterion"] else "FAIL", nonlinear_max=max(nl),
        period_convergence_status="UNKNOWN", mesh_convergence_status="UNKNOWN", period_convergence_qualified=False, mesh_convergence_qualified=False,
        period_reason="Only P1 startup and P2 stored; P1/P2 is not approved P2/P3 convergence",
        P1_vs_P2_AC_percent=float(100 * (scalar[:256, 2].mean() / means[2] - 1)),
        P1_P2_wave_L2_percent=float(100 * np.linalg.norm(scalar[:256, 2] - scalar[256:, 2]) / np.linalg.norm(scalar[256:, 2])),
        max_decomposition_residual_W_per_bar=max_identity, max_dt_relative_error=max_dt_relative,
        native_time_rounding_max_abs_s=native_time_delta, node_mapping_checks=map_checks,
        saved_snapshot_phase_count=8, snapshot_loss_vs_fullP2_percent=(100 * (sparse_mean / means[:3] - 1)).tolist(),
        snapshot_use="Spatial reader/development examples only. Use full 256-sample P2 integral history for scalar labels.",
        inputs=consumed, source_unchanged_during_read=True, new_solver_calls=0)
    write_json(out / (expected["point_id"] + "_audit.json"), result)
    print(json.dumps({k: result[k] for k in ["point_id", "raw_loss_reintegration", "delta_W", "P2_current_status", "whole_current_status"]}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True); p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--full-history-out", type=Path)
    a = p.parse_args(); audit(a.raw, read_json(a.expected), a.out, a.full_history_out)


if __name__ == "__main__":
    main()
