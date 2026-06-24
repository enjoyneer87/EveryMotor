#!/usr/bin/env python
"""Generate object/property-focused UML and physics I/O catalog from AEDT inventory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

KEY_REL_LABELS = {
    "modeler",
    "mesh",
    "materials",
    "boundaries",
    "setups",
    "available_variations",
    "variable_manager",
    "post",
    "monitor",
}
APP_PROP_ORDER = [
    "modeler",
    "mesh",
    "boundaries",
    "materials",
    "setups",
    "setup_names",
    "active_setup",
    "available_variations",
    "variable_manager",
    "monitor",
    "post",
]
DOMAIN_BY_APP = {
    "Mechanical": "structure",
    "Icepak": "flow",
    "Hfss": "electromagnetic",
    "Hfss3dLayout": "electromagnetic",
    "Maxwell2d": "electromagnetic",
    "Maxwell3d": "electromagnetic",
    "Q2d": "electromagnetic",
    "Q3d": "electromagnetic",
    "Rmxprt": "electromagnetic",
    "Emit": "electromagnetic",
    "Circuit": "electromagnetic",
    "CircuitNetlist": "electromagnetic",
    "MaxwellCircuit": "electromagnetic",
    "TwinBuilder": "electromagnetic",
    "Simplorer": "electromagnetic",
}
DOMAIN_TITLE = {
    "structure": "1) Structure (Structural)",
    "flow": "2) Flow/Thermal (CFD)",
    "electromagnetic": "3) Electromagnetic",
    "multiphysics": "4) Coupled Multi-Physics",
}
MULTIPHYSICS_HINTS = (
    "assign_2way_coupling",
    "assign_em_losses",
    "assign_thermal_map",
    "create_em_target_design",
    "create_external_circuit",
)
GENERIC_TYPE_NOISE = {
    "Get",
    "Set",
    "Design",
    "Boundaries",
    "Post",
    "Object",
    "Available",
    "Setups",
    "References",
    "Property",
    "Postprocessor",
    "Modeler",
}


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _unique_apps(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for app in sorted(inventory.get("apps", []), key=lambda a: a["name"]):
        name = app["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(app)
    return out


def _known_object_names(inventory: dict[str, Any]) -> set[str]:
    names = {profile.get("name", "") for profile in inventory.get("class_profiles", [])}
    names.update(rel.get("to", "") for rel in inventory.get("relations", []))
    return {n for n in names if n and n not in GENERIC_TYPE_NOISE}


def _relation_index(inventory: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    index: dict[str, dict[str, list[str]]] = {}
    for rel in inventory.get("relations", []):
        source = rel.get("from", "")
        label = rel.get("label", "")
        target = rel.get("to", "")
        if rel.get("type") == "has_a" and label in KEY_REL_LABELS:
            index.setdefault(source, {}).setdefault(label, []).append(target)
        if rel.get("type") == "uses" and label == "get_solution_data":
            index.setdefault(source, {}).setdefault(label, []).append(target)
    for source in list(index.keys()):
        for label in list(index[source].keys()):
            index[source][label] = _dedupe(index[source][label])
    return index


def _type_of_property(app: dict[str, Any], prop_name: str, known_types: set[str]) -> str:
    for detail in app.get("property_details", []):
        if detail.get("name") != prop_name:
            continue
        hits = [t for t in detail.get("return_types", []) if t in known_types and t not in GENERIC_TYPE_NOISE]
        if hits:
            return " | ".join(_dedupe(hits)[:2])
        return ""
    return ""


def _build_app_node(app: dict[str, Any], rel_index: dict[str, dict[str, list[str]]], known_types: set[str]) -> dict[str, Any]:
    app_props = set(app.get("properties", []))
    app_rel = rel_index.get(app["name"], {})
    lines: list[str] = []
    for prop in APP_PROP_ORDER:
        if prop not in app_props:
            continue
        rel_targets = [t for t in app_rel.get(prop, []) if t in known_types]
        if rel_targets:
            lines.append(f"{prop} : {' | '.join(rel_targets[:2])}")
            continue
        ptype = _type_of_property(app, prop, known_types)
        if ptype:
            lines.append(f"{prop} : {ptype}")
        else:
            lines.append(prop)
    if not lines:
        lines = app.get("properties", [])[:12]
    return {"name": app["name"], "kind": "app", "properties": lines}


def _component_profile_map(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for profile in inventory.get("class_profiles", []):
        out[profile.get("name", "")] = profile
    return out


def _object_relations(inventory: dict[str, Any], known_types: set[str]) -> list[dict[str, str]]:
    rels: list[dict[str, str]] = []
    for rel in inventory.get("relations", []):
        source = rel.get("from", "")
        target = rel.get("to", "")
        label = rel.get("label", "")
        if target not in known_types:
            continue
        if rel.get("type") == "has_a" and label in KEY_REL_LABELS:
            rels.append(rel)
        elif rel.get("type") == "uses" and label == "get_solution_data":
            rels.append(rel)
    seen = set()
    out: list[dict[str, str]] = []
    for rel in rels:
        key = (rel["from"], rel["to"], rel["type"], rel["label"])
        if key not in seen:
            seen.add(key)
            out.append(rel)
    return out


def _build_object_nodes(inventory: dict[str, Any], selected_relations: list[dict[str, str]]) -> list[dict[str, Any]]:
    profile_map = _component_profile_map(inventory)
    targets = sorted({r["to"] for r in selected_relations})
    nodes: list[dict[str, Any]] = []
    for name in targets:
        profile = profile_map.get(name, {})
        props = profile.get("properties", [])[:24]
        nodes.append({"name": name, "kind": "object", "properties": props})
    return nodes


def _render_object_puml(app_nodes: list[dict[str, Any]], object_nodes: list[dict[str, Any]], relations: list[dict[str, str]]) -> str:
    lines = [
        "@startuml",
        "hide empty members",
        "skinparam classAttributeIconSize 0",
        "",
    ]
    for node in app_nodes + object_nodes:
        lines.append(f'class {node["name"]} <<{node["kind"]}>> {{')
        for prop in node.get("properties", []):
            lines.append(f"  +{prop}")
        lines.append("}")
        lines.append("")
    for rel in relations:
        if rel["type"] == "has_a":
            lines.append(f'{rel["from"]} --> {rel["to"]} : {rel["label"]}')
        else:
            lines.append(f'{rel["from"]} ..> {rel["to"]} : {rel["label"]}')
    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def _render_object_mmd(app_nodes: list[dict[str, Any]], object_nodes: list[dict[str, Any]], relations: list[dict[str, str]]) -> str:
    lines = ["classDiagram", ""]
    for node in app_nodes + object_nodes:
        lines.append(f'class {node["name"]} {{')
        for prop in node.get("properties", []):
            lines.append(f"  +{prop}")
        lines.append("}")
        lines.append("")
    for rel in relations:
        if rel["type"] == "has_a":
            lines.append(f'{rel["from"]} --> {rel["to"]} : {rel["label"]}')
        else:
            lines.append(f'{rel["from"]} ..> {rel["to"]} : {rel["label"]}')
    return "\n".join(lines)


def _app_domain(app_name: str) -> str:
    return DOMAIN_BY_APP.get(app_name, "electromagnetic")


def _multiphysics_clues(app: dict[str, Any]) -> list[str]:
    methods = app.get("methods", [])
    clues = [m for m in methods if any(hint in m for hint in MULTIPHYSICS_HINTS)]
    return _dedupe(clues)


def _fmt_list(items: list[str]) -> str:
    items = _dedupe(items)
    return ", ".join(items) if items else "-"


def _io_row(app: dict[str, Any], rel_index: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    app_rel = rel_index.get(app["name"], {})
    input_labels = [
        "modeler",
        "mesh",
        "boundaries",
        "materials",
        "setups",
        "available_variations",
        "variable_manager",
        "monitor",
    ]
    input_objects: list[str] = []
    for label in input_labels:
        input_objects.extend(app_rel.get(label, []))

    output_objects: list[str] = []
    output_objects.extend(app_rel.get("post", []))
    output_objects.extend(app_rel.get("get_solution_data", []))

    c_out = app.get("categorized_outputs", {})
    has_field = bool(c_out.get("field_data", {}).get("methods", []))
    has_graph = bool(c_out.get("graph_scalar_data", {}).get("methods", [])) or bool(
        c_out.get("graph_scalar_data", {}).get("solution_data_properties", [])
    )
    out_types: list[str] = []
    if has_field:
        out_types.append("Field data")
    if has_graph:
        out_types.append("Graph/Scalar data")

    return {
        "app": app["name"],
        "inputs": _fmt_list(input_objects),
        "outputs": _fmt_list(output_objects),
        "output_types": _fmt_list(out_types),
    }


def _render_catalog_md(inventory: dict[str, Any], apps: list[dict[str, Any]], rel_index: dict[str, dict[str, list[str]]]) -> str:
    domain_apps: dict[str, list[dict[str, Any]]] = {"structure": [], "flow": [], "electromagnetic": []}
    for app in apps:
        domain_apps[_app_domain(app["name"])].append(app)

    multiphysics_rows: list[tuple[dict[str, Any], list[str]]] = []
    for app in apps:
        clues = _multiphysics_clues(app)
        if clues:
            multiphysics_rows.append((app, clues))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = inventory.get("meta", {})

    lines: list[str] = []
    lines.append("# AEDT Input-Output Catalog for SIM-AI")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Package: `{meta.get('package', '-')}` / version `{meta.get('pyaedt_version', '-')}`")
    lines.append("- Basis: Static API introspection (object + properties), not GUI runtime state")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Object-level Input/Output inventory for AI data interface definition.")
    lines.append("")

    for domain in ["structure", "flow", "electromagnetic"]:
        lines.append(f"## {DOMAIN_TITLE[domain]}")
        lines.append("")
        lines.append("| App | Input Objects | Output Objects | Output Type |")
        lines.append("|---|---|---|---|")
        for app in sorted(domain_apps[domain], key=lambda x: x["name"]):
            row = _io_row(app, rel_index)
            lines.append(f"| {row['app']} | {row['inputs']} | {row['outputs']} | {row['output_types']} |")
        lines.append("")

    lines.append(f"## {DOMAIN_TITLE['multiphysics']}")
    lines.append("")
    lines.append("| App | Coupling Clues (API) | Input Objects | Output Objects |")
    lines.append("|---|---|---|---|")
    for app, clues in sorted(multiphysics_rows, key=lambda x: x[0]["name"]):
        row = _io_row(app, rel_index)
        lines.append(
            f"| {app['name']} | {_fmt_list(clues)} | {row['inputs']} | {row['outputs']} |"
        )
    lines.append("")

    lines.append("## AI Learning I/O Split")
    lines.append("")
    lines.append("### (1) Data-Driven AI")
    lines.append("- Input: normalized object features from Geometry/Boundary/Mesh/Material/Setup/Variation")
    lines.append("- Output: field tensors (vector/scalar) + report scalars/time series")
    lines.append("")
    lines.append("### (2) Model(Physics)-Based")
    lines.append("- Input: physics parameters + BC/source/setup objects")
    lines.append("- Output: solver response fields + report outputs + sensitivity/uncertainty metrics")
    lines.append("")
    lines.append("## Assumptions")
    for assumption in meta.get("assumptions", []):
        lines.append(f"- {assumption}")
    return "\n".join(lines)


def _validate(puml: str, mmd: str, md: str) -> None:
    if not puml.startswith("@startuml") or "@enduml" not in puml:
        raise RuntimeError("Invalid PlantUML output.")
    if not mmd.startswith("classDiagram"):
        raise RuntimeError("Invalid Mermaid output.")
    if "| App | Input Objects |" not in md:
        raise RuntimeError("Catalog markdown is missing required table header.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate object/property UML and domain I/O catalog from AEDT inventory.")
    parser.add_argument("--inventory", default="analysis_outputs/pyaedt_uml/aedt_inventory.json")
    parser.add_argument("--out", default="analysis_outputs/pyaedt_uml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inv_path = Path(args.inventory)
    out_dir = Path(args.out)
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))

    apps = _unique_apps(inventory)
    known_types = _known_object_names(inventory)
    rel_index = _relation_index(inventory)
    relations = _object_relations(inventory, known_types)

    app_nodes = [_build_app_node(app, rel_index, known_types) for app in apps]
    object_nodes = _build_object_nodes(inventory, relations)

    puml = _render_object_puml(app_nodes, object_nodes, relations)
    mmd = _render_object_mmd(app_nodes, object_nodes, relations)
    md = _render_catalog_md(inventory, apps, rel_index)
    _validate(puml, mmd, md)

    out_dir.mkdir(parents=True, exist_ok=True)
    puml_path = out_dir / "aedt_objects_properties.puml"
    mmd_path = out_dir / "aedt_objects_properties.mmd"
    md_path = out_dir / "aedt_io_catalog.md"

    puml_path.write_text(puml, encoding="utf-8")
    mmd_path.write_text(mmd, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    print(f"Generated object PlantUML: {puml_path}")
    print(f"Generated object Mermaid:  {mmd_path}")
    print(f"Generated I/O catalog:     {md_path}")
    print(f"App nodes: {len(app_nodes)} / Object nodes: {len(object_nodes)} / Relations: {len(relations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
