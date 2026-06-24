from __future__ import annotations

from collections.abc import Iterable


MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP: dict[str, str] = {
    "TriIndex": "mesh/tri_index",
    "Node1": "mesh/node_1",
    "Node2": "mesh/node_2",
    "Node3": "mesh/node_3",
    "RegCode": "mesh/reg_code",
    "RegionCode": "regions/reg_code",
    "RegionName": "regions/name",
    "Bx": "fields/bx",
    "By": "fields/by",
    "A": "fields/a",
    "J": "fields/j",
    "Je": "fields/je",
    "NodeIndex": "mesh/node_id",
    "X": "mesh/node_x_mm",
    "Y": "mesh/node_y_mm",
    "A_node": "fields/a_node",
}


def map_header_tokens_to_h5_keys(
    header_tokens: Iterable[str],
) -> list[str]:
    """Map TXT header tokens to H5 keys using shared dictionary."""
    mapped: list[str] = []
    for token in header_tokens:
        normalized = token.replace(" ", "")
        if normalized in MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP:
            mapped.append(MOTORCAD_TXT_HEADER_TO_H5_KEY_MAP[normalized])
    return mapped
