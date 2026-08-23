"""Hand-authored registry families for portability evaluation and augmentation.

These registries deliberately vary concrete identifiers and metadata while
keeping the normalized capability contract stable. They are fixtures for
testing the router's registry adapter boundary, not production tools.
"""

from __future__ import annotations

from typing import Any

from .generic_pilot_v3 import CAPABILITY_FOCUS, TARGET_CAPABILITIES
from .tool_registry import ToolRegistry


EXTERNAL_REGISTRY_STYLES = ("spectrum", "quarry", "weave", "relay")

EXTERNAL_TOOL_IDS = (
    "lumen_route",
    "quartz_catalog",
    "ember_probe",
    "velvet_match",
    "orbit_index",
    "harbor_schema",
    "cinder_rows",
    "atlas_outline",
    "ripple_pages",
    "meadow_reader",
    "forge_symbols",
    "mosaic_evidence",
    "canyon_context",
    "prism_compare",
    "ledger_requirements",
    "signal_assess",
    "northstar_commit",
)

EXTERNAL_STYLE_TOOL_IDS = {
    "spectrum": EXTERNAL_TOOL_IDS,
    "quarry": tuple(f"q_{index:02d}_op" for index in range(len(TARGET_CAPABILITIES))),
    "weave": tuple(f"w{index:02d}_unit" for index in range(len(TARGET_CAPABILITIES))),
    "relay": tuple(f"relay_{chr(97 + index)}" for index in range(len(TARGET_CAPABILITIES))),
}

DIRECT_DESCRIPTIONS = {
    "plan_retrieval": "Plan a staged retrieval route from the current research state.",
    "list_sources": "Enumerate the sources currently available for inspection.",
    "search_content": "Locate relevant passages in the available source content.",
    "exact_pattern_search": "Find exact identifiers, phrases, or literal patterns.",
    "search_metadata": "Filter and search source metadata and catalog fields.",
    "inspect_structured_schema": "Inspect fields, types, and structure of tabular data.",
    "search_structured_records": "Filter and retrieve matching structured records.",
    "inspect_document_structure": "Inspect headings, sections, and document organization.",
    "search_document_pages": "Locate relevant pages or page ranges in a document.",
    "read_content": "Read the full content of a selected source.",
    "inspect_code_structure": "Inspect symbols, definitions, and code relationships.",
    "inspect_evidence": "Inspect evidence support, provenance, and verification details.",
    "expand_context": "Expand a partial or ambiguous result with surrounding context.",
    "compare_evidence": "Compare conflicting evidence and identify meaningful differences.",
    "update_requirements": "Update requirement coverage and remaining obligations.",
    "assess_evidence": "Assess whether the available evidence is sufficient.",
    "finalize_selection": "Finalize the best-supported selection when requirements are met.",
}

INDIRECT_DESCRIPTIONS = {
    "plan_retrieval": "Select the next staged operation for an information task.",
    "list_sources": "Show the currently addressable material and source handles.",
    "search_content": "Find relevant wording inside the available material.",
    "exact_pattern_search": "Match a literal token or precisely specified phrase.",
    "search_metadata": "Narrow the catalog using descriptive source attributes.",
    "inspect_structured_schema": "Describe the fields and value types of structured input.",
    "search_structured_records": "Return structured entries satisfying the requested conditions.",
    "inspect_document_structure": "Map the organization and landmarks of a document.",
    "search_document_pages": "Locate the document ranges associated with a requested topic.",
    "read_content": "Open and return the complete selected material.",
    "inspect_code_structure": "Trace definitions and relationships in a source program.",
    "inspect_evidence": "Review support, origin, and verification of an observed result.",
    "expand_context": "Widen an incomplete result until its surrounding meaning is visible.",
    "compare_evidence": "Place competing observations side by side for reconciliation.",
    "update_requirements": "Record which requested conditions are covered or outstanding.",
    "assess_evidence": "Judge whether the current observations justify a conclusion.",
    "finalize_selection": "Commit the supported choice when the decision is ready.",
}


def build_external_registry(style: str) -> ToolRegistry:
    """Build one frozen evaluation registry style."""

    if style not in EXTERNAL_STYLE_TOOL_IDS:
        raise ValueError(f"unknown registry style: {style}")
    tool_ids = EXTERNAL_STYLE_TOOL_IDS[style]
    tools: list[dict[str, Any]] = []
    for index, (tool_id, capability) in enumerate(zip(tool_ids, TARGET_CAPABILITIES)):
        description = DIRECT_DESCRIPTIONS[capability] if style == "spectrum" else INDIRECT_DESCRIPTIONS[capability]
        if style == "quarry":
            family = f"quarry_channel_{index % 5}"
            input_modality = "json" if index % 3 == 0 else "text"
            output_modality = "json" if index % 3 == 0 else "passages"
            schema = {
                "type": "object",
                "properties": {"expression": {"type": "string"}, "scope": {"type": "string"}},
                "required": ["expression"],
            }
        elif style == "weave":
            family = f"weave_cluster_{index % 4}"
            input_modality = "code" if capability == "inspect_code_structure" else "text"
            output_modality = "records" if index % 2 else "structured_summary"
            schema = {
                "type": "object",
                "properties": {"request": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["request"],
            }
        else:
            family = f"relay_channel_{index % 6}"
            input_modality = "pdf" if capability in {"search_document_pages", "inspect_document_structure"} else "text"
            output_modality = "records" if "structured" in capability else "passages"
            schema = {
                "type": "object",
                "properties": {"query": {"type": "string"}, "source_ref": {"type": "string"}},
                "required": ["query"],
            }
        tools.append(
            {
                "tool_id": tool_id,
                "tool_family": family,
                "description": description,
                "capabilities": [capability],
                "input_modalities": [input_modality],
                "output_modalities": [output_modality],
                "evidence_roles": [
                    "planning"
                    if capability == "plan_retrieval"
                    else "selection"
                    if capability == "finalize_selection"
                    else "observation"
                ],
                "side_effect_class": "none",
                "argument_schema": schema,
                "constraints": ["read_only"],
                "prerequisites": ["none"],
            }
        )
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"external_{style}_registry",
            "tools": tools,
        }
    )


_AUGMENTATION_STYLES = ("atlas", "sable", "orbit")


def build_portability_augmentation_registry(style: str, registry_index: int) -> ToolRegistry:
    """Build a training-only registry style not reused by the external test."""

    if style not in _AUGMENTATION_STYLES:
        raise ValueError(f"unknown augmentation registry style: {style}")
    tools: list[dict[str, Any]] = []
    for index, capability in enumerate(TARGET_CAPABILITIES):
        tool_id = f"{style}_{registry_index % 997:03d}_{index:02d}"
        if style == "atlas":
            family = f"atlas_lane_{(index + registry_index) % 7}"
            description = (
                f"This operation handles {CAPABILITY_FOCUS[capability]} while preserving the current task state."
            )
            input_modality = "metadata" if capability in {"list_sources", "search_metadata"} else "text"
            output_modality = "records" if "structured" in capability else "evidence"
            properties = {"request": {"type": "string"}, "limit": {"type": "integer"}}
            required = ["request"]
        elif style == "sable":
            family = f"sable_group_{(index * 3 + registry_index) % 9}"
            description = (
                f"Use this operation when the next decision depends on {CAPABILITY_FOCUS[capability]}."
            )
            input_modality = "code" if capability == "inspect_code_structure" else "mixed"
            output_modality = "structured_summary" if index % 2 == 0 else "passages"
            properties = {"expression": {"type": "string"}, "context": {"type": "object"}}
            required = ["expression"]
        else:
            family = f"orbit_channel_{(index + 2 * registry_index) % 8}"
            description = f"Apply this candidate to determine {CAPABILITY_FOCUS[capability]} from available material."
            input_modality = "pdf" if capability in {"search_document_pages", "inspect_document_structure"} else "text"
            output_modality = "records" if index % 3 == 0 else "evidence"
            properties = {"query": {"type": "string"}, "scope": {"type": "string"}}
            required = ["query"]
        if registry_index % 4 == 0:
            properties[f"context_{(index + registry_index) % 5}"] = {"type": "string"}
        tools.append(
            {
                "tool_id": tool_id,
                "tool_family": family,
                "description": description,
                "capabilities": [capability],
                "input_modalities": [input_modality],
                "output_modalities": [output_modality],
                "evidence_roles": ["planning" if capability == "plan_retrieval" else "selection" if capability == "finalize_selection" else "observation"],
                "side_effect_class": "none" if capability != "plan_retrieval" else "local_state_write",
                "argument_schema": {"type": "object", "properties": properties, "required": required},
                "constraints": ["read_only"],
                "prerequisites": ["none"],
            }
        )
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"augmentation_{style}_{registry_index % 997:03d}",
            "tools": tools,
        }
    )
