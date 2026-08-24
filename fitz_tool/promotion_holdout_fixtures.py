"""Sealed v2 promotion registries created after all model selection work."""

from __future__ import annotations

import random
from typing import Any

from .generic_pilot_v3 import TARGET_CAPABILITIES
from .tool_registry import ToolRegistry


PROMOTION_HOLDOUT_VERSION = "nomos-promotion-holdout.v2"
PROMOTION_REGISTRY_STYLES = ("keystone", "mariner", "tessera", "umbra")

_RELEVANT_PURPOSES = {
    "plan_retrieval": "design a sequence of evidence-gathering moves while leaving the decision open",
    "list_sources": "return the collection of resource references the workspace currently exposes",
    "search_content": "discover passages about a concept anywhere inside the available resources",
    "exact_pattern_search": "locate verbatim occurrences of a supplied symbol, key, or identifier",
    "search_metadata": "query descriptive resource attributes without examining resource bodies",
    "inspect_structured_schema": "describe the names and data types available in a structured asset",
    "search_structured_records": "evaluate conditions against structured data and emit qualifying entries",
    "inspect_document_structure": "produce a navigational outline of a document's sections and landmarks",
    "search_document_pages": "retrieve document locations whose pages cover a requested subject",
    "read_content": "load the entire body of one already identified resource",
    "inspect_code_structure": "enumerate software definitions and the relationships among them",
    "inspect_evidence": "audit an observation for traceability, support, and inspection status",
    "expand_context": "extend a clipped observation with adjacent source material",
    "compare_evidence": "analyze two or more observations for consistency and disagreement",
    "update_requirements": "synchronize the ledger of satisfied and outstanding decision conditions",
    "assess_evidence": "determine whether current support is adequate to permit a conclusion",
    "finalize_selection": "record the justified answer and terminate the decision process",
}

_DISTRACTOR_PURPOSES = (
    ("compute a numeric aggregate from supplied values", "number"),
    ("translate prose between human languages", "text"),
    ("format prose according to a presentation style", "text"),
    ("render a bitmap preview from a visual specification", "image"),
    ("report service health and process uptime", "status"),
    ("convert a timestamp between time zones", "time"),
    ("classify sentiment in a message", "label"),
    ("transcribe an audio recording into text", "text"),
    ("resize an image to requested dimensions", "image"),
    ("encrypt a local payload for storage", "bytes"),
    ("look up a cached value by key", "value"),
    ("summarize numeric telemetry into percentiles", "statistics"),
    ("validate an email address syntax", "boolean"),
    ("generate a random identifier", "identifier"),
    ("parse a calendar expression into a date", "date"),
    ("detect the natural language of a passage", "label"),
    ("convert markdown content into HTML", "html"),
)


def _ids(style: str) -> tuple[str, ...]:
    offset = {"keystone": 31, "mariner": 47, "tessera": 61, "umbra": 79}[style]
    return tuple(
        f"{style[0]}p{(offset + index * 53) % 997:03d}"
        for index in range(len(TARGET_CAPABILITIES) + len(_DISTRACTOR_PURPOSES))
    )


PROMOTION_CANONICAL_CAPABILITY_BY_TOOL_ID: dict[str, str | None] = {
    tool_id: (TARGET_CAPABILITIES[index] if index < len(TARGET_CAPABILITIES) else None)
    for style in PROMOTION_REGISTRY_STYLES
    for index, tool_id in enumerate(_ids(style))
}


def _description(style: str, purpose: str) -> str:
    if style == "keystone":
        return f"This local interface exists to {purpose}."
    if style == "mariner":
        return f"Route work here whenever the immediate outcome is to {purpose}."
    if style == "tessera":
        return f"A read-only adapter that can {purpose}."
    return f"Its bounded responsibility is to {purpose}; it does not perform unrelated steps."


def build_promotion_registry(style: str) -> ToolRegistry:
    """Build a 34-tool registry whose canonical mapping stays evaluator-only."""

    if style not in PROMOTION_REGISTRY_STYLES:
        raise ValueError(f"unknown promotion registry style: {style}")
    rng = random.Random(7001 + PROMOTION_REGISTRY_STYLES.index(style) * 1009)
    aliases = list(range(34))
    rng.shuffle(aliases)
    ids = _ids(style)
    tools: list[dict[str, Any]] = []
    argument_names = ("intent", "payload", "selector", "value", "request")
    for index, tool_id in enumerate(ids):
        if index < len(TARGET_CAPABILITIES):
            canonical = TARGET_CAPABILITIES[index]
            purpose = _RELEVANT_PURPOSES[canonical]
            if canonical in {"inspect_structured_schema", "search_structured_records"}:
                input_modality, output_modality = "structured", "records"
            elif canonical in {"inspect_document_structure", "search_document_pages"}:
                input_modality, output_modality = "pdf", "passages"
            elif canonical == "inspect_code_structure":
                input_modality, output_modality = "code", "structured_summary"
            else:
                input_modality, output_modality = "text", "evidence"
            role = (
                "selection"
                if canonical == "finalize_selection"
                else "planning"
                if canonical == "plan_retrieval"
                else "observation"
            )
        else:
            canonical = None
            purpose, output_modality = _DISTRACTOR_PURPOSES[
                index - len(TARGET_CAPABILITIES)
            ]
            input_modality, role = "text", "utility"
        argument_name = argument_names[(index * 3 + len(style)) % len(argument_names)]
        tools.append(
            {
                "tool_id": tool_id,
                "tool_family": f"{style}_module_{(index * 7 + 2) % 13}",
                "description": _description(style, purpose),
                "capabilities": [f"bridge.op_{aliases[index]:02d}"],
                "input_modalities": [input_modality],
                "output_modalities": [output_modality],
                "evidence_roles": [role],
                "side_effect_class": "none",
                "argument_schema": {
                    "type": "object",
                    "properties": {argument_name: {"type": "string"}},
                    "required": [argument_name],
                    "additionalProperties": False,
                },
                "constraints": ["local_only", "read_only"],
                "prerequisites": ["none"],
            }
        )
        if canonical is not None and PROMOTION_CANONICAL_CAPABILITY_BY_TOOL_ID[tool_id] != canonical:
            raise RuntimeError("promotion mapping is inconsistent")
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"promotion_{style}_registry",
            "tools": tools,
        }
    )


def canonical_capability(tool_id: str) -> str | None:
    return PROMOTION_CANONICAL_CAPABILITY_BY_TOOL_ID[tool_id]
