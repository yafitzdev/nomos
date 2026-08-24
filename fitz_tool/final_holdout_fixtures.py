"""Frozen post-development registries for one-shot production evaluation.

The router never receives the canonical capability mapping below.  Final-suite
tools expose opaque capability labels plus ordinary descriptions and schemas,
which is closer to integrating an unseen third-party registry.
"""

from __future__ import annotations

from typing import Any

from .generic_pilot_v3 import TARGET_CAPABILITIES
from .tool_registry import ToolRegistry


FINAL_HOLDOUT_VERSION = "nomos-final-holdout.v1"
FINAL_REGISTRY_STYLES = ("cipher", "harvest", "switchboard", "monolith")

_PURPOSES = {
    "plan_retrieval": "lay out the next information-gathering actions before opening material",
    "list_sources": "show which local resources and handles can currently be addressed",
    "search_content": "find relevant wording across the body of available material",
    "exact_pattern_search": "match an exact token, identifier, or literal phrase without broadening it",
    "search_metadata": "narrow resources by catalog attributes rather than their body text",
    "inspect_structured_schema": "reveal columns, field names, and value types in structured input",
    "search_structured_records": "return rows or records satisfying structured conditions",
    "inspect_document_structure": "map headings, sections, and navigation landmarks in a document",
    "search_document_pages": "locate the pages or document ranges associated with a topic",
    "read_content": "open the complete content of one selected resource",
    "inspect_code_structure": "trace program symbols, definitions, and relationships",
    "inspect_evidence": "review an observation's support, origin, and verification status",
    "expand_context": "widen a partial result to include the surrounding material needed to interpret it",
    "compare_evidence": "place multiple observations side by side and identify agreement or conflict",
    "update_requirements": "record which requested conditions are covered and which remain open",
    "assess_evidence": "judge whether the collected observations are sufficient for a conclusion",
    "finalize_selection": "commit the supported answer after the decision conditions are satisfied",
}

_STYLE_IDS = {
    "cipher": tuple(f"cx_{97 + index * 13:03d}" for index in range(len(TARGET_CAPABILITIES))),
    "harvest": tuple(f"hv-{(index * 37 + 11) % 211:03d}" for index in range(len(TARGET_CAPABILITIES))),
    "switchboard": tuple(f"sw.node.{(index * 19 + 5) % 173:03d}" for index in range(len(TARGET_CAPABILITIES))),
    "monolith": tuple(f"mx{(index * 43 + 29) % 257:03x}" for index in range(len(TARGET_CAPABILITIES))),
}

FINAL_CANONICAL_CAPABILITY_BY_TOOL_ID = {
    tool_id: capability
    for style in FINAL_REGISTRY_STYLES
    for tool_id, capability in zip(_STYLE_IDS[style], TARGET_CAPABILITIES)
}


def _description(style: str, purpose: str) -> str:
    if style == "cipher":
        return f"Read-only endpoint used to {purpose}."
    if style == "harvest":
        return f"Choose this operation when the current job needs to {purpose}."
    if style == "switchboard":
        return f"Dispatch here to {purpose}; the response is safe to inspect locally."
    return f"Local analysis service whose responsibility is to {purpose}."


def build_final_registry(style: str) -> ToolRegistry:
    """Build one frozen, unseen registry without canonical capability leakage."""

    if style not in FINAL_REGISTRY_STYLES:
        raise ValueError(f"unknown final registry style: {style}")
    tools: list[dict[str, Any]] = []
    for index, (tool_id, canonical) in enumerate(
        zip(_STYLE_IDS[style], TARGET_CAPABILITIES)
    ):
        if canonical in {"inspect_structured_schema", "search_structured_records"}:
            input_modality, output_modality = "structured", "records"
        elif canonical in {"inspect_document_structure", "search_document_pages"}:
            input_modality, output_modality = "pdf", "passages"
        elif canonical == "inspect_code_structure":
            input_modality, output_modality = "code", "structured_summary"
        else:
            input_modality, output_modality = "text", "evidence"
        argument_name = ("needle", "request", "subject", "expression")[index % 4]
        tools.append(
            {
                "tool_id": tool_id,
                "tool_family": f"{style}_group_{(index * 3 + 1) % 8}",
                "description": _description(style, _PURPOSES[canonical]),
                # Deliberately opaque: the benchmark mapping is not serialized.
                "capabilities": [f"protocol.action_{(index * 29 + 7) % 101:02d}"],
                "input_modalities": [input_modality],
                "output_modalities": [output_modality],
                "evidence_roles": [
                    "selection"
                    if canonical == "finalize_selection"
                    else "planning"
                    if canonical == "plan_retrieval"
                    else "observation"
                ],
                "side_effect_class": "none",
                "argument_schema": {
                    "type": "object",
                    "properties": {argument_name: {"type": "string"}},
                    "required": [argument_name],
                    "additionalProperties": False,
                },
                "constraints": ["read_only", "local_only"],
                "prerequisites": ["none"],
            }
        )
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"final_{style}_registry",
            "tools": tools,
        }
    )


def canonical_capability(tool_id: str) -> str:
    """Return the evaluator-only capability for a final holdout tool."""

    return FINAL_CANONICAL_CAPABILITY_BY_TOOL_ID[tool_id]
