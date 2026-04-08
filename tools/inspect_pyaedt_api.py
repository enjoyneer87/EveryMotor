#!/usr/bin/env python
"""Static PyAEDT API inspector for UML and JSON inventory generation.

This script introspects the installed ``ansys.aedt.core`` package without
launching AEDT. It extracts application classes, key input/output API surfaces,
and writes:
1) JSON inventory
2) PlantUML class diagram
3) Mermaid class diagram
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_EXCLUDE = {"Desktop", "Quantity"}
INPUT_REL_PROPS = {
    "modeler",
    "mesh",
    "materials",
    "boundaries",
    "post",
    "setups",
    "available_variations",
    "variable_manager",
}
PROBLEM_DEF_PROPS = [
    "setups",
    "setup_names",
    "active_setup",
    "available_variations",
    "variable_manager",
]
COMPONENT_CLASS_PATHS = {
    "Modeler2D": ("ansys.aedt.core.modeler.modeler_2d", "Modeler2D"),
    "Modeler3D": ("ansys.aedt.core.modeler.modeler_3d", "Modeler3D"),
    "ModelerNexxim": ("ansys.aedt.core.modeler.schematic", "ModelerNexxim"),
    "Mesh": ("ansys.aedt.core.modules.mesh", "Mesh"),
    "IcepakMesh": ("ansys.aedt.core.modules.mesh_icepak", "IcepakMesh"),
    "Materials": ("ansys.aedt.core.modules.material_lib", "Materials"),
    "BoundaryObject": ("ansys.aedt.core.modules.boundary.common", "BoundaryObject"),
    "Setup": ("ansys.aedt.core.modules.solve_setup", "Setup"),
    "SolutionData": ("ansys.aedt.core.visualization.post.common", "SolutionData"),
    "PostProcessor3D": ("ansys.aedt.core.visualization.post.post_common_3d", "PostProcessor3D"),
    "PostProcessor3DLayout": ("ansys.aedt.core.visualization.post.post_3dlayout", "PostProcessor3DLayout"),
    "PostProcessorIcepak": ("ansys.aedt.core.visualization.post.post_icepak", "PostProcessorIcepak"),
    "PostProcessorCircuit": ("ansys.aedt.core.visualization.post.post_circuit", "PostProcessorCircuit"),
    "AvailableVariations": ("ansys.aedt.core.application.analysis", "AvailableVariations"),
}
GRAPH_OUTPUT_METHOD_FAMILIES = ("create_report", "get_solution_data", "export_report")
FIELD_OUTPUT_METHOD_EXACT = {"get_scalar_field_value", "get_efields_data"}


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    rel_type: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {
            "from": self.source,
            "to": self.target,
            "type": self.rel_type,
            "label": self.label,
        }


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_type_name(raw: str) -> str:
    text = raw.strip().strip("'").strip('"').replace("~", "")
    text = text.replace("typing.", "")
    if text.startswith("<class "):
        text = text.replace("<class ", "").strip(">").strip("'").strip('"')
    if "." in text:
        return text.split(".")[-1]
    return text


def _extract_type_names_from_text(text: str) -> list[str]:
    names: list[str] = []
    class_refs = re.findall(r":class:`([^`]+)`", text)
    for ref in class_refs:
        for part in re.split(r"\s+or\s+|,", ref):
            cleaned = part.strip()
            if cleaned:
                names.append(_normalize_type_name(cleaned))
    for dotted in re.findall(r"ansys\.aedt\.core(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text):
        names.append(_normalize_type_name(dotted))
    generic_refs = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", text)
    for token in generic_refs:
        if token in {"Returns", "List", "Dict", "Tuple", "Optional", "None", "True", "False"}:
            continue
        if token.startswith("Py"):
            continue
        names.append(token)
    return _dedupe_keep_order(names)


def _property_owner(cls: type, prop_name: str) -> type | None:
    for base in cls.__mro__:
        attr = base.__dict__.get(prop_name)
        if isinstance(attr, property):
            return base
    return None


def _get_public_properties(cls: type) -> list[str]:
    props = [name for name, value in inspect.getmembers(cls) if isinstance(value, property) and not name.startswith("_")]
    return sorted(_dedupe_keep_order(props))


def _matches_method(name: str, detail: str) -> bool:
    if name.startswith("_"):
        return False
    if detail == "exhaustive":
        starts = (
            "assign_",
            "create_",
            "set_",
            "analyze",
            "export_",
            "import_",
            "update_",
            "get_",
            "delete_",
            "plot_",
            "sample_",
            "change_",
        )
        contains = (
            "data",
            "report",
            "field",
            "mesh",
            "material",
            "boundary",
            "setup",
            "solution",
            "sweep",
            "variation",
            "source",
            "port",
            "matrix",
        )
        return name.startswith(starts) or any(token in name for token in contains)
    starts_core = ("assign_", "create_", "set_", "analyze", "export_", "get_")
    contains_core = ("data", "report", "field", "mesh", "material", "setup")
    return name.startswith(starts_core) or any(token in name for token in contains_core)


def _get_filtered_methods(cls: type, detail: str) -> list[str]:
    methods = [name for name, value in inspect.getmembers(cls, inspect.isfunction) if _matches_method(name, detail)]
    return sorted(_dedupe_keep_order(methods))


def _safe_import_class(module_name: str, class_name: str) -> type | None:
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None


def _load_component_classes() -> dict[str, type]:
    loaded: dict[str, type] = {}
    for short_name, (module_name, class_name) in COMPONENT_CLASS_PATHS.items():
        cls = _safe_import_class(module_name, class_name)
        if cls is not None:
            loaded[short_name] = cls
    return loaded


def _extract_property_details(cls: type) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for name in _get_public_properties(cls):
        owner = _property_owner(cls, name)
        owner_name = owner.__name__ if owner else cls.__name__
        prop = getattr(owner, name) if owner else getattr(cls, name)
        if not isinstance(prop, property):
            continue
        type_names: list[str] = []
        fget = prop.fget
        if fget is not None:
            ann = fget.__annotations__.get("return")
            if ann is not None:
                type_names.extend(_extract_type_names_from_text(str(ann)))
            doc = inspect.getdoc(fget) or ""
            type_names.extend(_extract_type_names_from_text(doc))
        type_names = _dedupe_keep_order(type_names)
        details.append({"name": name, "owner": owner_name, "return_types": type_names})
    return details


def _method_subset(methods: list[str], *, starts: tuple[str, ...] = (), contains: tuple[str, ...] = (), exact: set[str] | None = None) -> list[str]:
    exact = exact or set()
    out = []
    for method in methods:
        if method in exact:
            out.append(method)
            continue
        if starts and method.startswith(starts):
            out.append(method)
            continue
        if contains and any(token in method for token in contains):
            out.append(method)
    return sorted(_dedupe_keep_order(out))


def _component_methods(component_classes: dict[str, type], component_names: list[str], detail: str) -> list[str]:
    methods: list[str] = []
    for name in component_names:
        cls = component_classes.get(name)
        if cls is None:
            continue
        methods.extend(_get_filtered_methods(cls, detail))
    return sorted(_dedupe_keep_order(methods))


def _discover_app_classes(scope: str) -> list[type]:
    import ansys.aedt.core as aedt_core
    from ansys.aedt.core.application.design import Design

    classes = []
    for name, cls in inspect.getmembers(aedt_core, inspect.isclass):
        if not cls.__module__.startswith("ansys.aedt.core"):
            continue
        if name in APP_EXCLUDE:
            continue
        try:
            if not issubclass(cls, Design):
                continue
        except Exception:
            continue
        classes.append(cls)

    classes = sorted(classes, key=lambda c: c.__name__)
    if scope == "all":
        return classes

    motor_preferred = {
        "Hfss",
        "Hfss3dLayout",
        "Maxwell2d",
        "Maxwell3d",
        "Icepak",
        "Mechanical",
        "Q2d",
        "Q3d",
        "Rmxprt",
        "Circuit",
        "TwinBuilder",
        "MaxwellCircuit",
        "Emit",
    }
    return [cls for cls in classes if cls.__name__ in motor_preferred]


def _return_types_for_prop(prop_details: list[dict[str, Any]], prop_name: str) -> list[str]:
    for detail in prop_details:
        if detail["name"] == prop_name:
            return detail["return_types"]
    return []


def _build_app_record(cls: type, component_classes: dict[str, type], detail: str) -> tuple[dict[str, Any], list[Relation]]:
    prop_names = _get_public_properties(cls)
    prop_details = _extract_property_details(cls)
    method_names = _get_filtered_methods(cls, detail)

    modeler_types = _return_types_for_prop(prop_details, "modeler")
    mesh_types = _return_types_for_prop(prop_details, "mesh")
    material_types = _return_types_for_prop(prop_details, "materials") or ["Materials"]
    boundary_types = _return_types_for_prop(prop_details, "boundaries") or ["BoundaryObject"]
    post_types = _return_types_for_prop(prop_details, "post")
    setup_types = _return_types_for_prop(prop_details, "setups") or ["Setup"]

    geometry_methods = _method_subset(
        _component_methods(component_classes, modeler_types, detail),
        starts=("create_", "import_", "update_"),
        contains=("geometry",),
    )
    mesh_methods = _method_subset(
        _component_methods(component_classes, mesh_types, detail),
        starts=("assign_", "create_", "set_", "get_", "generate_"),
        contains=("mesh",),
        exact={"generate_mesh"},
    )
    boundary_condition_methods = _method_subset(
        method_names,
        starts=("assign_",),
        contains=("port", "source"),
    )
    material_methods = _method_subset(
        _component_methods(component_classes, material_types, detail),
        starts=("add_", "import_", "duplicate_", "remove_"),
        contains=("material",),
    )
    problem_definition_methods = _method_subset(
        method_names + _component_methods(component_classes, setup_types, detail),
        starts=("create_setup", "analyze", "set_"),
        contains=("setup", "solve", "variation"),
    )
    problem_definition_properties = [name for name in PROBLEM_DEF_PROPS if name in prop_names]

    post_methods = _component_methods(component_classes, post_types, detail)
    field_output_methods = _method_subset(
        post_methods,
        starts=("create_fieldplot_", "plot_field", "export_field_"),
        exact=FIELD_OUTPUT_METHOD_EXACT,
    )
    graph_output_methods = _method_subset(
        post_methods + method_names,
        starts=GRAPH_OUTPUT_METHOD_FAMILIES,
        contains=("report",),
    )

    solution_data_cls = component_classes.get("SolutionData")
    solution_data_methods: list[str] = []
    solution_data_properties: list[str] = []
    if solution_data_cls is not None:
        solution_data_methods = _method_subset(
            _get_filtered_methods(solution_data_cls, detail),
            starts=("get_", "export_", "plot_", "ifft", "set_"),
            contains=("data", "variation", "expression", "report", "plot"),
        )
        solution_data_properties = _method_subset(
            _get_public_properties(solution_data_cls),
            contains=("matrix", "variation", "sweep", "expression"),
        )

    categorized_inputs = {
        "geometry": {
            "properties": ["modeler"] if "modeler" in prop_names else [],
            "classes": modeler_types,
            "methods": geometry_methods,
        },
        "mesh": {
            "properties": ["mesh"] if "mesh" in prop_names else [],
            "classes": mesh_types,
            "methods": mesh_methods,
        },
        "boundary_and_excitation": {
            "properties": ["boundaries"] if "boundaries" in prop_names else [],
            "classes": boundary_types,
            "methods": boundary_condition_methods,
        },
        "materials": {
            "properties": ["materials"] if "materials" in prop_names else [],
            "classes": material_types,
            "methods": material_methods,
        },
        "problem_definition": {
            "properties": problem_definition_properties,
            "classes": setup_types,
            "methods": problem_definition_methods,
        },
    }

    categorized_outputs = {
        "field_data": {
            "properties": ["post"] if "post" in prop_names else [],
            "classes": post_types,
            "methods": field_output_methods,
        },
        "graph_scalar_data": {
            "properties": ["post"] if "post" in prop_names else [],
            "classes": post_types + (["SolutionData"] if solution_data_methods or solution_data_properties else []),
            "methods": graph_output_methods,
            "solution_data_methods": solution_data_methods,
            "solution_data_properties": solution_data_properties,
        },
    }

    app_relations: list[Relation] = []
    for prop_name in INPUT_REL_PROPS:
        target_types = _return_types_for_prop(prop_details, prop_name)
        for target_type in target_types:
            if target_type and target_type != cls.__name__:
                app_relations.append(Relation(cls.__name__, target_type, "has_a", prop_name))

    for idx in range(len(cls.__mro__) - 1):
        child = cls.__mro__[idx]
        parent = cls.__mro__[idx + 1]
        if not child.__module__.startswith("ansys.aedt.core"):
            continue
        if not parent.__module__.startswith("ansys.aedt.core"):
            continue
        app_relations.append(Relation(child.__name__, parent.__name__, "inheritance", "extends"))

    app_record = {
        "name": cls.__name__,
        "module": cls.__module__,
        "mro": [c.__name__ for c in cls.__mro__ if c.__module__.startswith("ansys.aedt.core")],
        "properties": prop_names,
        "methods": method_names,
        "property_details": prop_details,
        "categorized_inputs": categorized_inputs,
        "categorized_outputs": categorized_outputs,
    }
    return app_record, app_relations


def _build_component_profiles(component_classes: dict[str, type], detail: str) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for name, cls in sorted(component_classes.items()):
        profiles.append(
            {
                "name": name,
                "module": cls.__module__,
                "kind": "component",
                "properties": _get_public_properties(cls),
                "methods": _get_filtered_methods(cls, detail),
            }
        )
    return profiles


def _build_inventory(scope: str, detail: str) -> dict[str, Any]:
    import ansys.aedt.core as aedt_core

    component_classes = _load_component_classes()
    apps: list[dict[str, Any]] = []
    relations: list[Relation] = []
    class_profiles: list[dict[str, Any]] = []

    for cls in _discover_app_classes(scope):
        app_record, app_relations = _build_app_record(cls, component_classes, detail)
        apps.append(app_record)
        relations.extend(app_relations)
        class_profiles.append(
            {
                "name": app_record["name"],
                "module": app_record["module"],
                "kind": "app",
                "properties": app_record["properties"],
                "methods": app_record["methods"],
            }
        )

    class_profiles.extend(_build_component_profiles(component_classes, detail))

    if "SolutionData" in component_classes:
        relations.append(Relation("PostProcessor3D", "SolutionData", "uses", "get_solution_data"))
        relations.append(Relation("PostProcessorIcepak", "SolutionData", "uses", "get_solution_data"))
        relations.append(Relation("PostProcessorCircuit", "SolutionData", "uses", "get_solution_data"))
        relations.append(Relation("PostProcessor3DLayout", "SolutionData", "uses", "get_solution_data"))

    rel_set = set()
    unique_relations: list[Relation] = []
    for rel in relations:
        key = (rel.source, rel.target, rel.rel_type, rel.label)
        if key not in rel_set:
            rel_set.add(key)
            unique_relations.append(rel)

    all_names = {profile["name"] for profile in class_profiles}
    for rel in unique_relations:
        if rel.source not in all_names:
            class_profiles.append({"name": rel.source, "module": "", "kind": "derived", "properties": [], "methods": []})
            all_names.add(rel.source)
        if rel.target not in all_names:
            class_profiles.append({"name": rel.target, "module": "", "kind": "derived", "properties": [], "methods": []})
            all_names.add(rel.target)

    class_profiles = sorted(class_profiles, key=lambda item: (item["kind"], item["name"]))
    inventory = {
        "meta": {
            "inspector": "tools/inspect_pyaedt_api.py",
            "package": "ansys.aedt.core",
            "pyaedt_version": getattr(aedt_core, "__version__", "unknown"),
            "scope": scope,
            "detail": detail,
            "assumptions": [
                "Static introspection only. No AEDT GUI/session-dependent runtime values.",
                "API surface taken from installed package in current interpreter.",
            ],
        },
        "apps": sorted(apps, key=lambda item: item["name"]),
        "relations": [rel.to_dict() for rel in unique_relations],
        "class_profiles": class_profiles,
    }
    return inventory


def _plantuml_relation(rel: dict[str, str]) -> str:
    rel_type = rel["type"]
    if rel_type == "inheritance":
        return f'{rel["to"]} <|-- {rel["from"]} : {rel["label"]}'
    if rel_type == "has_a":
        return f'{rel["from"]} --> {rel["to"]} : {rel["label"]}'
    return f'{rel["from"]} ..> {rel["to"]} : {rel["label"]}'


def _mermaid_relation(rel: dict[str, str]) -> str:
    rel_type = rel["type"]
    if rel_type == "inheritance":
        return f'{rel["to"]} <|-- {rel["from"]} : {rel["label"]}'
    if rel_type == "has_a":
        return f'{rel["from"]} --> {rel["to"]} : {rel["label"]}'
    return f'{rel["from"]} ..> {rel["to"]} : {rel["label"]}'


def _render_plantuml(inventory: dict[str, Any]) -> str:
    lines = [
        "@startuml",
        "hide empty members",
        "skinparam classAttributeIconSize 0",
        "",
    ]
    for profile in inventory["class_profiles"]:
        stereotype = profile["kind"]
        lines.append(f'class {profile["name"]} <<{stereotype}>> {{')
        for prop in profile["properties"]:
            lines.append(f"  +{prop}")
        for method in profile["methods"]:
            lines.append(f"  +{method}()")
        lines.append("}")
        lines.append("")
    for rel in inventory["relations"]:
        lines.append(_plantuml_relation(rel))
    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def _render_mermaid(inventory: dict[str, Any]) -> str:
    lines = ["classDiagram", ""]
    for profile in inventory["class_profiles"]:
        lines.append(f"class {profile['name']} {{")
        for prop in profile["properties"]:
            lines.append(f"  +{prop}")
        for method in profile["methods"]:
            lines.append(f"  +{method}()")
        lines.append("}")
        lines.append("")
    for rel in inventory["relations"]:
        lines.append(_mermaid_relation(rel))
    return "\n".join(lines)


def _validate_outputs(plantuml_text: str, mermaid_text: str, inventory: dict[str, Any]) -> None:
    if not plantuml_text.startswith("@startuml") or "@enduml" not in plantuml_text:
        raise RuntimeError("PlantUML output missing header/footer tokens.")
    if not mermaid_text.startswith("classDiagram"):
        raise RuntimeError("Mermaid output missing classDiagram header.")
    if not inventory.get("apps") or not inventory.get("relations"):
        raise RuntimeError("Inventory is empty. Check introspection filters.")


def _write_outputs(inventory: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aedt_inventory.json"
    puml_path = out_dir / "aedt_full.puml"
    mmd_path = out_dir / "aedt_full.mmd"

    plantuml_text = _render_plantuml(inventory)
    mermaid_text = _render_mermaid(inventory)
    _validate_outputs(plantuml_text, mermaid_text, inventory)

    json_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    puml_path.write_text(plantuml_text, encoding="utf-8")
    mmd_path.write_text(mermaid_text, encoding="utf-8")
    return {"json": json_path, "plantuml": puml_path, "mermaid": mmd_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PyAEDT API and generate UML + JSON inventory.")
    parser.add_argument("--scope", choices=["all", "motor"], default="all", help="Application scope to inspect.")
    parser.add_argument(
        "--detail",
        choices=["core", "exhaustive"],
        default="exhaustive",
        help="Method inclusion depth for class members.",
    )
    parser.add_argument(
        "--out",
        default="analysis_outputs/pyaedt_uml",
        help="Output directory for generated artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = _build_inventory(scope=args.scope, detail=args.detail)
    paths = _write_outputs(inventory, Path(args.out))
    print(f"Generated inventory: {paths['json']}")
    print(f"Generated PlantUML:  {paths['plantuml']}")
    print(f"Generated Mermaid:   {paths['mermaid']}")
    print(f"Apps discovered:     {len(inventory['apps'])}")
    print(f"Relations discovered:{len(inventory['relations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
