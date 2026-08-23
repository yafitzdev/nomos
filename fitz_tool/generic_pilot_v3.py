"""Project-agnostic deterministic decision-state generation for Nomos."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .generic_contracts import ContractReport, validate_decision_state_v2
from .router_v2 import FEATURE_VERSION
from .tool_registry import ToolRegistry, ToolSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERIC_MATRIX_PATH = PROJECT_ROOT / "configs" / "matrix.generic.v3.json"
GENERIC_DATASET_VERSION = "nomos-generic.v3"
GENERIC_PILOT_VERSION = "generic-pilot.v3.50k"
GENERIC_PILOT_SEED = 20260824
GENERIC_GENERATED_AT = "2026-08-24T00:00:00+00:00"

GENERIC_COHORT_COUNTS = {
    "train": 34000,
    "validation": 4000,
    "familiar_registries": 2500,
    "unseen_tool_ids": 2000,
    "id_renames": 1000,
    "schema_variants": 1000,
    "modality_variants": 1000,
    "heldout_family": 2000,
    "heldout_sources": 1000,
    "heldout_questions": 1000,
    "alternate_registry": 500,
}

TARGET_CAPABILITIES = (
    "plan_retrieval",
    "list_sources",
    "search_content",
    "exact_pattern_search",
    "search_metadata",
    "inspect_structured_schema",
    "search_structured_records",
    "inspect_document_structure",
    "search_document_pages",
    "read_content",
    "inspect_code_structure",
    "inspect_evidence",
    "expand_context",
    "compare_evidence",
    "update_requirements",
    "assess_evidence",
    "finalize_selection",
)

TRAIN_TEMPLATE_IDS = (
    "generic_direct",
    "generic_paraphrase",
    "generic_indirect",
    "generic_compositional",
    "generic_constraint",
    "generic_stateful",
    "generic_contrastive",
    "generic_evidence",
    "generic_priority",
    "generic_failure",
    "generic_metadata",
    "generic_action",
)
HOLDOUT_TEMPLATE_IDS = (
    "generic_holdout_implicit",
    "generic_holdout_consequence",
    "generic_holdout_tradeoff",
    "generic_holdout_sparse",
    "generic_holdout_ambiguous",
    "generic_holdout_plan",
)

PHASE_TO_STATE = {
    "planning": "initial",
    "source_discovery": "source_discovery",
    "retrieval": "retrieval",
    "inspection": "partial_evidence",
    "synthesis": "contradiction",
    "governance": "insufficient",
    "terminal_readiness": "fresh_sufficient",
}

TASK_FOCUS = {
    "access_control": "access rules and permissions",
    "data_processing": "the required data transformation",
    "event_handling": "event delivery and retry behavior",
    "schema_management": "field definitions and compatibility",
    "failure_recovery": "failure handling and recovery",
    "versioning": "version compatibility",
    "observability": "telemetry and trace requirements",
    "consistency_checks": "consistency across sources",
    "resource_management": "resource limits and lifecycle",
    "workflow_automation": "the next workflow step",
    "document_analysis": "the relevant technical instruction",
    "data_quality": "validation and quality constraints",
}

CAPABILITY_FOCUS = {
    "plan_retrieval": "the best next information-gathering step",
    "list_sources": "the available resources",
    "search_content": "the relevant content",
    "exact_pattern_search": "an exact name, code, or token",
    "search_metadata": "resource metadata and filters",
    "inspect_structured_schema": "fields, types, and structure",
    "search_structured_records": "matching records",
    "inspect_document_structure": "sections and structural landmarks",
    "search_document_pages": "the relevant pages or ranges",
    "read_content": "the selected resource",
    "inspect_code_structure": "symbols and definitions",
    "inspect_evidence": "a candidate result",
    "expand_context": "missing surrounding context",
    "compare_evidence": "agreement or conflict between results",
    "update_requirements": "task requirements and coverage",
    "assess_evidence": "whether available information is sufficient",
    "finalize_selection": "the selected result or next action",
}

CAPABILITY_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "plan_retrieval": {
        "family": "planning",
        "inputs": ["agent_state", "text"],
        "outputs": ["plan"],
        "roles": ["none"],
        "side_effect": "local_state_write",
        "constraints": ["none"],
        "prerequisites": ["objective_available"],
        "schema": {"objective": "string", "steps": "array", "priority": "string"},
    },
    "list_sources": {
        "family": "source_discovery",
        "inputs": ["metadata"],
        "outputs": ["source_list"],
        "roles": ["source_discovery"],
        "side_effect": "read",
        "constraints": ["none"],
        "prerequisites": ["none"],
        "schema": {"scope": "string", "filters": "object"},
    },
    "search_content": {
        "family": "content_retrieval",
        "inputs": ["text", "pdf"],
        "outputs": ["evidence_candidates"],
        "roles": ["candidate_discovery"],
        "side_effect": "read",
        "constraints": ["indexed_sources_required"],
        "prerequisites": ["query_available"],
        "schema": {"query": "string", "limit": "integer"},
    },
    "exact_pattern_search": {
        "family": "exact_retrieval",
        "inputs": ["text", "code"],
        "outputs": ["evidence_candidates"],
        "roles": ["candidate_discovery"],
        "side_effect": "read",
        "constraints": ["textual_source_required"],
        "prerequisites": ["query_available"],
        "schema": {"pattern": "string", "case_sensitive": "boolean"},
    },
    "search_metadata": {
        "family": "metadata_retrieval",
        "inputs": ["metadata"],
        "outputs": ["source_list"],
        "roles": ["source_discovery"],
        "side_effect": "read",
        "constraints": ["metadata_catalog_required"],
        "prerequisites": ["none"],
        "schema": {"filters": "object", "sort": "string"},
    },
    "inspect_structured_schema": {
        "family": "structured_inspection",
        "inputs": ["csv", "excel", "sqlite"],
        "outputs": ["schema"],
        "roles": ["structure_discovery"],
        "side_effect": "read",
        "constraints": ["structured_source_required"],
        "prerequisites": ["source_selected"],
        "schema": {"resource": "string", "path": "string", "include_types": "boolean"},
    },
    "search_structured_records": {
        "family": "structured_retrieval",
        "inputs": ["csv", "excel", "sqlite"],
        "outputs": ["evidence_candidates"],
        "roles": ["candidate_discovery"],
        "side_effect": "read",
        "constraints": ["structured_source_required"],
        "prerequisites": ["schema_known"],
        "schema": {"resource": "string", "predicates": "array", "limit": "integer"},
    },
    "inspect_document_structure": {
        "family": "document_inspection",
        "inputs": ["pdf"],
        "outputs": ["document_structure"],
        "roles": ["structure_discovery"],
        "side_effect": "read",
        "constraints": ["pdf_source_required"],
        "prerequisites": ["source_selected"],
        "schema": {"resource": "string", "landmark": "string"},
    },
    "search_document_pages": {
        "family": "document_retrieval",
        "inputs": ["pdf"],
        "outputs": ["evidence_candidates"],
        "roles": ["candidate_discovery"],
        "side_effect": "read",
        "constraints": ["pdf_source_required"],
        "prerequisites": ["source_selected"],
        "schema": {"resource": "string", "query": "string", "page_range": "array"},
    },
    "read_content": {
        "family": "content_inspection",
        "inputs": ["text", "code"],
        "outputs": ["content"],
        "roles": ["evidence_inspection"],
        "side_effect": "read",
        "constraints": ["local_source_required"],
        "prerequisites": ["source_selected"],
        "schema": {"resource": "string", "offset": "integer"},
    },
    "inspect_code_structure": {
        "family": "code_inspection",
        "inputs": ["code"],
        "outputs": ["content", "code_structure"],
        "roles": ["evidence_inspection"],
        "side_effect": "read",
        "constraints": ["code_source_required"],
        "prerequisites": ["source_selected"],
        "schema": {"resource": "string", "symbol": "string", "depth": "integer"},
    },
    "inspect_evidence": {
        "family": "evidence_inspection",
        "inputs": ["evidence_candidates"],
        "outputs": ["evidence"],
        "roles": ["evidence_inspection"],
        "side_effect": "read",
        "constraints": ["candidate_evidence_required"],
        "prerequisites": ["evidence_candidate_selected"],
        "schema": {"result_id": "string", "claim": "string"},
    },
    "expand_context": {
        "family": "evidence_inspection",
        "inputs": ["evidence_candidates", "evidence"],
        "outputs": ["evidence"],
        "roles": ["context_expansion"],
        "side_effect": "read",
        "constraints": ["incomplete_context_required"],
        "prerequisites": ["evidence_candidate_selected"],
        "schema": {"result_id": "string", "window": "integer"},
    },
    "compare_evidence": {
        "family": "evidence_synthesis",
        "inputs": ["evidence"],
        "outputs": ["evidence_comparison"],
        "roles": ["evidence_synthesis"],
        "side_effect": "none",
        "constraints": ["multiple_evidence_required"],
        "prerequisites": ["evidence_inspected"],
        "schema": {"result_ids": "array", "comparison": "string"},
    },
    "update_requirements": {
        "family": "governance",
        "inputs": ["evidence", "agent_state"],
        "outputs": ["governance_state"],
        "roles": ["governance_update"],
        "side_effect": "local_state_write",
        "constraints": ["tracked_requirements_required"],
        "prerequisites": ["evidence_inspected"],
        "schema": {"requirement_id": "string", "result_ids": "array", "status": "string"},
    },
    "assess_evidence": {
        "family": "governance",
        "inputs": ["evidence", "governance_state"],
        "outputs": ["assessment"],
        "roles": ["governance_assessment"],
        "side_effect": "local_state_write",
        "constraints": ["canonical_evidence_set_required"],
        "prerequisites": ["requirements_updated"],
        "schema": {"result_ids": "array", "requirements": "array"},
    },
    "finalize_selection": {
        "family": "terminal",
        "inputs": ["assessment", "governance_state"],
        "outputs": ["terminal_result"],
        "roles": ["terminal_decision"],
        "side_effect": "local_state_write",
        "constraints": ["fresh_sufficient_assessment_required"],
        "prerequisites": ["requirements_complete"],
        "schema": {"selected_ids": "array", "decision": "string"},
    },
}

PROJECT_MARKER_RE = re.compile(r"(?<![a-z0-9_])(fitz|sage|bm25)(?![a-z0-9_])", re.IGNORECASE)

RELATED_CAPABILITIES = {
    "search_content": "retrieve_passages",
    "exact_pattern_search": "identifier_search",
    "search_metadata": "filter_sources",
    "list_sources": "discover_sources",
    "inspect_structured_schema": "discover_fields",
    "search_structured_records": "filter_records",
    "inspect_document_structure": "discover_sections",
    "search_document_pages": "retrieve_passages",
    "read_content": "inspect_full_source",
    "inspect_code_structure": "resolve_symbols",
    "inspect_evidence": "verify_provenance",
    "expand_context": "resolve_split_context",
    "compare_evidence": "detect_contradictions",
    "update_requirements": "track_coverage",
    "assess_evidence": "validate_sufficiency",
    "finalize_selection": "emit_terminal_result",
}


@dataclass(frozen=True)
class GenericSourceCard:
    source_id: str
    modality: str
    corpus: str
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "source-card.v3",
            "source_id": self.source_id,
            "document_id": self.source_id,
            "title": f"Synthetic resource {self.source_id}",
            "modality": self.modality,
            "corpus": self.corpus,
            "path": f"synthetic://{GENERIC_DATASET_VERSION}/{self.source_id}",
            "content_sha256": self.content_sha256,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def load_generic_matrix_spec(path: Path | str = GENERIC_MATRIX_PATH) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("matrix_version") != "matrix.generic.v3":
        raise ValueError("expected matrix.generic.v3")
    for group_name in ("dimensions", "observable_ranges"):
        group = spec.get(group_name)
        if not isinstance(group, Mapping) or not group:
            raise ValueError(f"{group_name} must be a non-empty object")
        for name, values in group.items():
            if not isinstance(values, list) or not values or len(values) != len(set(values)):
                raise ValueError(f"{group_name}.{name} must contain unique values")
    return spec


def generic_matrix_cell_id(values: Mapping[str, Any]) -> str:
    return _sha256(dict(values))


def _valid_generic_cell(values: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for group_name in ("dimensions", "observable_ranges"):
        for name, allowed in spec[group_name].items():
            if name not in values:
                errors.append(f"missing {name}")
            elif values[name] not in allowed:
                errors.append(f"invalid {name}: {values[name]!r}")
    if errors:
        return errors
    modality = values["source_modality"]
    target = values["target_capability"]
    inspection = values["evidence_inspection_state"]
    progress = values["requirement_progress"]
    freshness = values["assessment_freshness"]
    terminal = values["terminal_outcome"]
    topology = values["evidence_topology"]
    if target in {"inspect_document_structure", "search_document_pages"} and modality not in {"pdf", "mixed"}:
        errors.append(f"{target} requires pdf or mixed source modality")
    if target in {"inspect_structured_schema", "search_structured_records"} and modality not in {"csv", "excel", "sqlite", "mixed"}:
        errors.append(f"{target} requires a structured or mixed source modality")
    if target == "inspect_code_structure" and modality not in {"code", "mixed"}:
        errors.append("inspect_code_structure requires code or mixed source modality")
    if topology == "cross_format" and modality != "mixed":
        errors.append("cross_format evidence requires mixed source modality")
    if topology == "contradictory" and values["observed_evidence_count"] < 2:
        errors.append("contradictory evidence requires at least two observed evidence items")
    if topology == "absent_after_exhaustion" and values["prior_search_count"] < 2:
        errors.append("absence requires at least two prior searches")
    if freshness == "fresh_for_current_evidence" and inspection in {"none", "snippets_only"}:
        errors.append("fresh assessment requires inspected context")
    if freshness == "fresh_for_current_evidence" and progress not in {"complete", "disputed"}:
        errors.append("fresh assessment requires complete or disputed requirements")
    if target in {"inspect_evidence", "expand_context", "compare_evidence", "update_requirements", "assess_evidence", "finalize_selection"} and values["observed_evidence_count"] == 0:
        errors.append(f"{target} requires observed evidence")
    if target == "compare_evidence" and inspection != "multi_source_inspected":
        errors.append("compare_evidence requires multiple inspected sources")
    if target == "expand_context" and inspection not in {"snippets_only", "partial_context"}:
        errors.append("expand_context requires incomplete context")
    if target == "assess_evidence" and inspection in {"none", "snippets_only"}:
        errors.append("assess_evidence requires inspected evidence")
    if target == "finalize_selection":
        if terminal != "selection":
            errors.append("finalize_selection requires selection terminal outcome")
        if freshness != "fresh_for_current_evidence" or progress != "complete":
            errors.append("finalize_selection requires fresh complete state")
    elif terminal == "selection":
        errors.append("selection terminal outcome requires finalize_selection")
    if target == "list_sources" and values["source_inventory_state"] == "known":
        errors.append("list_sources is not optimal when inventory is known")
    if target == "plan_retrieval" and values["agent_phase"] not in {"planning", "retrieval"}:
        errors.append("plan_retrieval requires planning or retrieval")
    if values["remaining_steps"] == 0 and terminal == "ongoing":
        errors.append("zero remaining steps cannot be ongoing")
    return errors


def _sample_cell(rng: random.Random, spec: Mapping[str, Any], target: str, used: set[str]) -> dict[str, Any]:
    for _ in range(20000):
        values = {name: rng.choice(options) for name, options in spec["dimensions"].items()}
        values.update({name: rng.choice(options) for name, options in spec["observable_ranges"].items()})
        values["target_capability"] = target
        cell_id = generic_matrix_cell_id(values)
        if cell_id not in used and not _valid_generic_cell(values, spec):
            return values
    raise RuntimeError(f"could not sample a legal generic cell for {target}")


def _schema(properties: Mapping[str, str], variant: int) -> dict[str, Any]:
    type_map: dict[str, Any] = {
        "string": {"type": "string"},
        "integer": {"type": "integer"},
        "boolean": {"type": "boolean"},
        "array": {"type": "array", "items": {"type": "string"}},
        "object": {"type": "object"},
    }
    names = list(properties)
    required = [name for index, name in enumerate(names) if (index + variant) % 3 != 1]
    if not required:
        required = names[:1]
    return {
        "type": "object",
        "properties": {name: type_map[value] for name, value in properties.items()},
        "required": required,
    }


def _description(capability: str, style: int) -> str:
    focus = CAPABILITY_FOCUS[capability]
    templates = (
        "Use this candidate when the task requires {focus}.",
        "This operation handles {focus} while preserving the current task state.",
        "Invoke the candidate to determine {focus} from the available resources.",
        "The operation produces a reliable result for {focus} under current constraints.",
        "Apply this capability when the next decision depends on {focus}.",
        "This candidate supports controlled inspection of {focus}.",
    )
    return templates[style % len(templates)].format(focus=focus)


def build_generic_registry(
    registry_index: int,
    *,
    profile: str = "alpha",
    id_mode: str = "opaque",
    schema_variant: bool = False,
    modality_variant: bool = False,
    family_variant: bool = False,
) -> ToolRegistry:
    tools: list[dict[str, Any]] = []
    family_prefix = f"family_{profile}_" if family_variant else "family_"
    tool_index = 0
    for capability_index, capability in enumerate(TARGET_CAPABILITIES):
        blueprint = CAPABILITY_BLUEPRINTS[capability]
        count = 2 if (capability_index + registry_index) % 5 == 0 else 1
        for variant in range(count):
            if id_mode == "renamed":
                tool_id = f"candidate_{registry_index:03d}_{tool_index:02d}_{(registry_index + tool_index) % 17:02d}"
            elif id_mode == "unseen":
                tool_id = f"opaque_{(registry_index * 37 + tool_index * 11) % 100000:05d}"
            else:
                tool_id = f"op_{registry_index:03d}_{tool_index:02d}"
            family = f"{family_prefix}{blueprint['family']}"
            if profile in {"delta", "epsilon"}:
                family = f"{family_prefix}{blueprint['family']}_{profile}"
            inputs = list(blueprint["inputs"])
            if modality_variant and variant == 0:
                inputs = ["mixed"] if capability not in {"plan_retrieval", "finalize_selection"} else inputs
            capabilities = [capability]
            if variant == 1 and capability in RELATED_CAPABILITIES:
                capabilities.append(RELATED_CAPABILITIES[capability])
            side_effect = blueprint["side_effect"]
            if profile == "epsilon" and side_effect == "read" and variant == 0:
                side_effect = "none"
            properties = dict(blueprint["schema"])
            if schema_variant and variant == 0:
                properties[f"context_{(registry_index + capability_index) % 5}"] = "string"
            tools.append(
                {
                    "tool_id": tool_id,
                    "tool_family": family,
                    "description": _description(capability, registry_index + variant),
                    "capabilities": capabilities,
                    "input_modalities": inputs,
                    "output_modalities": list(blueprint["outputs"]),
                    "evidence_roles": list(blueprint["roles"]),
                    "side_effect_class": side_effect,
                    "argument_schema": _schema(properties, registry_index + variant if schema_variant else variant),
                    "constraints": list(blueprint["constraints"]),
                    "prerequisites": list(blueprint["prerequisites"]),
                }
            )
            tool_index += 1
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"registry_{profile}_{registry_index:03d}",
            "tools": tools,
        }
    )


def _source_cards() -> dict[str, GenericSourceCard]:
    cards: dict[str, GenericSourceCard] = {}
    modalities = ("text", "pdf", "csv", "excel", "sqlite", "code")
    for corpus, count in (("train_resources", 32), ("heldout_resources", 16)):
        prefix = "train" if corpus == "train_resources" else "holdout"
        for index in range(count):
            modality = modalities[index % len(modalities)]
            source_id = f"generic_{prefix}_source_{index:03d}"
            digest = hashlib.sha256(
                f"{GENERIC_DATASET_VERSION}|{corpus}|{source_id}|{modality}".encode("utf-8")
            ).hexdigest()
            cards[source_id] = GenericSourceCard(source_id, modality, corpus, digest)
    return cards


def _source_ids_for_cell(
    cell: Mapping[str, Any], cards: Mapping[str, GenericSourceCard], cohort: str, offset: int
) -> list[str]:
    holdout = cohort == "heldout_sources"
    pool = [card for card in cards.values() if (card.corpus == "heldout_resources") == holdout]
    modality = str(cell["source_modality"])
    matching = [card for card in pool if card.modality == modality]
    if modality == "mixed":
        matching = pool
    if not matching:
        matching = pool
    first = matching[offset % len(matching)]
    selected = [first]
    if modality == "mixed":
        second = next((card for card in matching if card.modality != first.modality), None)
        if second is not None:
            selected.append(second)
    return [card.source_id for card in selected]


def _question(template_id: str, task_domain: str, target: str, operation: str) -> str:
    task = TASK_FOCUS[task_domain]
    focus = CAPABILITY_FOCUS[target]
    if template_id == "generic_direct":
        return f"Which available operation should determine {focus} for the current task about {task}?"
    if template_id == "generic_paraphrase":
        return f"What should happen next to establish {focus} while resolving the task involving {task}?"
    if template_id == "generic_indirect":
        return f"The task needs a reliable answer about {task}. Which capability can move the work forward?"
    if template_id == "generic_compositional":
        return f"Given the {operation} step, current state, and available resources, choose the operation that determines {focus} without skipping required checks."
    if template_id == "generic_constraint":
        return f"Which candidate can establish {focus} while respecting the current evidence, modality, and side-effect constraints for {task}?"
    if template_id == "generic_stateful":
        return f"Considering what has already been inspected and what remains unresolved about {task}, what is the next reliable action for {focus}?"
    if template_id == "generic_contrastive":
        return f"Which operation is preferable for {focus} rather than a broader or less precise action on {task}?"
    if template_id == "generic_evidence":
        return f"What should the agent use to verify {focus} from the available results related to {task}?"
    if template_id == "generic_priority":
        return f"To avoid wasting the remaining steps, which operation should be prioritized for {focus} in the {task} task?"
    if template_id == "generic_failure":
        return f"The current attempt is incomplete or ambiguous. Which operation can recover the information needed for {focus} about {task}?"
    if template_id == "generic_metadata":
        return f"Which available capability can narrow or inspect the resources needed to determine {focus} for {task}?"
    if template_id == "generic_action":
        return f"Select the next tool action that gives the strongest support for {focus} while handling {task}."
    if template_id == "generic_holdout_implicit":
        return f"The agent must make the next dependable move on {task} with the evidence currently available. What kind of operation is needed?"
    if template_id == "generic_holdout_consequence":
        return f"If the current information is not enough to resolve {task}, what operation should be chosen before making a decision?"
    if template_id == "generic_holdout_tradeoff":
        return f"Which candidate best balances precision, available context, and safety while resolving {task}?"
    if template_id == "generic_holdout_sparse":
        return f"With limited context and an unresolved question about {task}, identify the operation that should come next."
    if template_id == "generic_holdout_ambiguous":
        return f"The task state is ambiguous. What kind of tool call would clarify the information needed about {task}?"
    return f"Before closing the task about {task}, what operation should establish the missing support?"


def _candidate_specs(
    registry: ToolRegistry, target: str, difficulty: str, rng: random.Random
) -> tuple[list[ToolSpec], list[ToolSpec]]:
    targets = [tool for tool in registry.tools if target in tool.capabilities]
    if not targets:
        raise ValueError(f"registry {registry.registry_id} lacks target {target}")
    rng.shuffle(targets)
    acceptable = targets[: min(2, len(targets))]
    target_families = {tool.tool_family for tool in acceptable}
    target_capabilities = {cap for tool in acceptable for cap in tool.capabilities}
    remaining = [tool for tool in registry.tools if tool not in acceptable]
    same_family = [tool for tool in remaining if tool.tool_family in target_families]
    overlap = [tool for tool in remaining if set(tool.capabilities) & target_capabilities]
    modality = [
        tool
        for tool in remaining
        if set(tool.input_modalities)
        & {modality for target_tool in acceptable for modality in target_tool.input_modalities}
    ]
    buckets = {
        "same_family_near_neighbor": same_family + overlap,
        "capability_overlap": overlap + same_family,
        "schema_only_difference": overlap + modality,
        "modality_constraint": modality,
        "side_effect_constraint": remaining,
        "cross_family_easy": remaining,
    }
    preferred: list[ToolSpec] = []
    seen: set[str] = set()
    for tool in [*buckets.get(difficulty, remaining), *remaining]:
        if tool.tool_id not in seen:
            preferred.append(tool)
            seen.add(tool.tool_id)
    rng.shuffle(preferred)
    desired = min(len(registry.tools), rng.randint(5, 8))
    selected = acceptable + preferred[: max(1, desired - len(acceptable))]
    rng.shuffle(selected)
    return selected, acceptable


def _observable_state(
    cell: Mapping[str, Any], source_ids: list[str], cards: Mapping[str, GenericSourceCard], question: str, rng: random.Random
) -> dict[str, Any]:
    phase = str(cell["agent_phase"])
    evidence_count = int(cell["observed_evidence_count"])
    evidence: list[dict[str, Any]] = []
    for index in range(evidence_count):
        source_id = source_ids[index % len(source_ids)]
        evidence.append(
            {
                "evidence_id": f"result-{rng.randrange(10**12):012d}",
                "source_id": source_id,
                "modality": cards[source_id].modality,
                "inspection_status": "inspected" if cell["evidence_inspection_state"] != "none" else "candidate",
                "claim_count": 1 + (index % 3),
            }
        )
    progress = str(cell["requirement_progress"])
    requirements = [
        {"requirement_id": "R1", "status": "complete" if progress == "complete" else "missing"},
        {"requirement_id": "R2", "status": "disputed" if progress == "disputed" else "missing"},
    ]
    if progress == "none":
        requirements = [{"requirement_id": "R1", "status": "missing"}]
    return {
        "question": question,
        "agent_state": {
            "state_name": PHASE_TO_STATE[phase],
            "phase": phase,
            "question_length_band": "short" if len(question) < 100 else "long",
        },
        "history": [
            {"step": index, "action_family": "tool_use", "result": "weak"}
            for index in range(int(cell["prior_search_count"]))
        ],
        "plan": {"active": phase in {"planning", "retrieval"}, "operation": cell["information_operation"]},
        "observed_evidence": evidence,
        "governance": {
            "assessment_fresh": cell["assessment_freshness"] == "fresh_for_current_evidence",
            "requirements": requirements,
            "allowed_side_effect_classes": ["none", "read", "local_state_write"],
        },
        "resource_state": {
            "remaining_steps": cell["remaining_steps"],
            "unresolved_requirement_count": cell["unresolved_requirement_count"],
            "observed_evidence_count": evidence_count,
            "distractor_count": cell["distractor_count"],
            "prior_search_count": cell["prior_search_count"],
        },
        "source_state": {
            "available_modalities": ["text", "pdf", "csv", "excel", "sqlite", "code", "metadata"]
            if cell["source_modality"] == "mixed"
            else [cell["source_modality"], "metadata"],
            "inventory_state": cell["source_inventory_state"],
            "inspection_state": cell["evidence_inspection_state"],
            "source_ids": source_ids,
            "source_card_hashes": [cards[source_id].content_sha256 for source_id in source_ids],
            "evidence_topology_observed": cell["evidence_topology"] if cell["evidence_topology"] != "absent_after_exhaustion" else "absent",
        },
        "query_state": {
            "operation": cell["information_operation"],
            "specificity": cell["query_specificity"],
            "match_strategy": cell["match_strategy"],
            "query_terms": [token for token in question.lower().split() if len(token) > 4][:10],
        },
    }


def _type_signature(state: Mapping[str, Any]) -> str:
    registry = ToolRegistry.from_dict(state["tool_registry"])
    payload = {
        "matrix_cell": state["matrix_cell"],
        "registry_semantics": sorted(tool.semantic_fingerprint for tool in registry.resolve(state["legal_candidate_ids"])),
        "source_card_ids": sorted(state.get("source_card_ids") or []),
        "question_template_id": state.get("question_template_id"),
        "evaluation_cohort": state.get("evaluation_cohort"),
    }
    return _sha256(payload)


def _instance_signature(state: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "type_signature": state.get("type_signature"),
            "question": state.get("question"),
            "teacher_paraphrase": state.get("teacher_paraphrase"),
            "history": state.get("history"),
            "observed_evidence": state.get("observed_evidence"),
            "seed": (state.get("provenance") or {}).get("seed"),
        }
    )


def _build_state(
    index: int,
    cohort: str,
    target: str,
    registry: ToolRegistry,
    cell: Mapping[str, Any],
    cards: Mapping[str, GenericSourceCard],
    template_id: str,
    seed: int,
    split_group_id: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    source_ids = _source_ids_for_cell(cell, cards, cohort, index)
    question = _question(template_id, str(cell["task_domain"]), target, str(cell["information_operation"]))
    legal_specs, acceptable_specs = _candidate_specs(registry, target, str(cell["candidate_set_difficulty"]), rng)
    legal_ids = [tool.tool_id for tool in legal_specs]
    acceptable_ids = [tool.tool_id for tool in acceptable_specs if tool.tool_id in legal_ids]
    hard_negatives = [tool_id for tool_id in legal_ids if tool_id not in acceptable_ids]
    state = _observable_state(cell, source_ids, cards, question, rng)
    state.update(
        {
            "schema_version": "decision-state.v2",
            "dataset_version": GENERIC_DATASET_VERSION,
            "decision_state_id": f"generic-v3-{index:05d}",
            "trajectory_id": f"generic-v3-trajectory-{index:05d}",
            "scenario_id": f"generic-v3-scenario-{index:05d}",
            "step": 0,
            "tool_registry": registry.as_dict(),
            "legal_candidate_ids": legal_ids,
            "label": {
                "acceptable_tools": acceptable_ids,
                "ranked_tools": acceptable_ids + hard_negatives,
                "hard_negative_tools": hard_negatives,
                "label_source": "generic_matrix_oracle.v3",
            },
            "accepted": True,
            "source_card_ids": source_ids,
            "source_kind": "deterministic_generic_oracle",
            "evaluation_cohort": cohort,
            "evaluation_partition": "train" if cohort == "train" else "validation" if cohort == "validation" else "test",
            "split_group_id": split_group_id,
            "question_template_id": template_id,
            "matrix_cell": dict(cell),
            "matrix_cell_id": generic_matrix_cell_id(cell),
            "sampling_context": {
                "matrix_version": "matrix.generic.v3",
                "target_capability": target,
                "candidate_set_difficulty": cell["candidate_set_difficulty"],
                "terminal_outcome": cell["terminal_outcome"],
                "retrieval_obstacle": cell["retrieval_obstacle"],
                "task_domain": cell["task_domain"],
                "agent_contract_profile": cell["agent_contract_profile"],
                "excluded_candidate_reasons": {
                    "modality_constraint": "recorded as an excluded illegal candidate when applicable",
                    "side_effect_constraint": "recorded as an excluded illegal candidate when applicable",
                },
            },
            "provenance": {
                "corpus": GENERIC_DATASET_VERSION,
                "source_card_hashes": [cards[source_id].content_sha256 for source_id in source_ids],
                "prompt_version": f"generic-v3-{template_id}",
                "model": "deterministic_generic_oracle",
                "artifact": "generic-matrix-oracle.v3",
                "teacher": "deterministic_generic_oracle",
                "seed": seed,
                "validator_version": "generic-validator.v3",
                "feature_version": FEATURE_VERSION,
                "registry_fingerprint": registry.fingerprint,
                "trajectory_hash": _sha256({"trajectory_id": f"generic-v3-trajectory-{index:05d}", "seed": seed}),
                "matrix_cell_id": generic_matrix_cell_id(cell),
                "generated_at": GENERIC_GENERATED_AT,
            },
        }
    )
    state["type_signature"] = _type_signature(state)
    state["instance_signature"] = _instance_signature(state)
    return state


def _registry_for_row(cohort: str, offset: int) -> ToolRegistry:
    if cohort == "unseen_tool_ids":
        return build_generic_registry(offset % 32, profile="alpha", id_mode="unseen")
    if cohort == "id_renames":
        return build_generic_registry(offset % 16, profile="beta", id_mode="renamed")
    if cohort == "schema_variants":
        return build_generic_registry(offset % 16, profile="gamma", schema_variant=True)
    if cohort == "modality_variants":
        return build_generic_registry(offset % 16, profile="gamma", modality_variant=True)
    if cohort == "heldout_family":
        return build_generic_registry(offset % 16, profile="delta", family_variant=True)
    if cohort == "alternate_registry":
        return build_generic_registry(offset % 8, profile="epsilon", family_variant=True, schema_variant=True)
    if cohort == "validation":
        return build_generic_registry(100 + offset % 12, profile="beta")
    if cohort == "familiar_registries":
        return build_generic_registry(200 + offset % 16, profile="gamma")
    return build_generic_registry(offset % 48, profile="alpha")


def generate_generic_states(
    *, count: int = sum(GENERIC_COHORT_COUNTS.values()), seed: int = GENERIC_PILOT_SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = sum(GENERIC_COHORT_COUNTS.values())
    if count != expected:
        raise ValueError(f"generic pilot requires exactly {expected} rows")
    spec = load_generic_matrix_spec()
    cards = _source_cards()
    rng = random.Random(seed)
    used_cells: set[str] = set()
    used_types: set[str] = set()
    used_instances: set[str] = set()
    rows: list[dict[str, Any]] = []
    index = 0
    for cohort, cohort_count in GENERIC_COHORT_COUNTS.items():
        targets = [TARGET_CAPABILITIES[offset % len(TARGET_CAPABILITIES)] for offset in range(cohort_count)]
        for offset, target in enumerate(targets):
            cell = _sample_cell(rng, spec, target, used_cells)
            cell["agent_contract_profile"] = {
                "train": "registry_alpha",
                "validation": "registry_beta",
                "familiar_registries": "registry_gamma",
                "unseen_tool_ids": "registry_alpha",
                "id_renames": "registry_beta",
                "schema_variants": "registry_gamma",
                "modality_variants": "registry_gamma",
                "heldout_family": "registry_delta",
                "heldout_sources": "registry_alpha",
                "heldout_questions": "registry_alpha",
                "alternate_registry": "registry_epsilon",
            }[cohort]
            cell_id = generic_matrix_cell_id(cell)
            if cell_id in used_cells or _valid_generic_cell(cell, spec):
                raise RuntimeError("generic cell uniqueness or validity failure")
            used_cells.add(cell_id)
            registry = _registry_for_row(cohort, offset)
            template_pool = HOLDOUT_TEMPLATE_IDS if cohort == "heldout_questions" else TRAIN_TEMPLATE_IDS
            template_id = template_pool[offset % len(template_pool)]
            split_group_id = f"{cohort}|registry-{offset % 32}|source-{offset % 16}|template-{offset % len(template_pool)}"
            row_seed = seed + index * 1009 + offset
            row = _build_state(index, cohort, target, registry, cell, cards, template_id, row_seed, split_group_id)
            if row["type_signature"] in used_types:
                raise RuntimeError(f"duplicate generic type signature at row {index}")
            if row["instance_signature"] in used_instances:
                raise RuntimeError(f"duplicate generic instance signature at row {index}")
            used_types.add(row["type_signature"])
            used_instances.add(row["instance_signature"])
            rows.append(row)
            index += 1
    target_counts = Counter(str(row["sampling_context"]["target_capability"]) for row in rows)
    template_counts = Counter(str(row["question_template_id"]) for row in rows)
    registry_fingerprints = {str(row["tool_registry"]["registry_id"]): row["tool_registry"]["registry_fingerprint"] for row in rows}
    training_families = sorted({tool["tool_family"] for row in rows if row["evaluation_cohort"] == "train" for tool in row["tool_registry"]["tools"]})
    heldout_families = sorted({tool["tool_family"] for row in rows if row["evaluation_cohort"] == "heldout_family" for tool in row["tool_registry"]["tools"]})
    manifest = {
        "dataset_version": GENERIC_DATASET_VERSION,
        "pilot_version": GENERIC_PILOT_VERSION,
        "feature_version": FEATURE_VERSION,
        "seed": seed,
        "count": len(rows),
        "cohort_counts": dict(Counter(row["evaluation_cohort"] for row in rows)),
        "target_capability_counts": dict(sorted(target_counts.items())),
        "question_template_counts": dict(sorted(template_counts.items())),
        "registry_fingerprints": dict(sorted(registry_fingerprints.items())),
        "source_cards": {source_id: card.as_dict() for source_id, card in cards.items()},
        "training_tool_families": training_families,
        "heldout_family_tool_families": heldout_families,
        "heldout_family_overlap": sorted(set(training_families) & set(heldout_families)),
        "matrix_cells": len(used_cells),
        "type_signatures": len(used_types),
        "instance_signatures": len(used_instances),
        "generic_registry_only": True,
    }
    return rows, manifest


def validate_generic_state(state: Mapping[str, Any]) -> ContractReport:
    report = validate_decision_state_v2(state)
    if state.get("dataset_version") != GENERIC_DATASET_VERSION:
        report.add("dataset_version", f"must equal {GENERIC_DATASET_VERSION}")
    for key in ("matrix_cell", "matrix_cell_id", "type_signature", "instance_signature", "evaluation_cohort", "question_template_id", "source_kind"):
        if key not in state:
            report.add(key, "missing generic dataset field")
    teacher_source_kinds = {
        "ninfer_generic_teacher": "ninfer",
        "deepseek_generic_teacher": "deepseek",
    }
    if state.get("source_kind") not in {"deterministic_generic_oracle", *teacher_source_kinds}:
        report.add("source_kind", "must identify the generic oracle or an approved external teacher")
    if state.get("label", {}).get("label_source") != "generic_matrix_oracle.v3":
        report.add("label.label_source", "must identify generic_matrix_oracle.v3")
    if state.get("source_kind") in teacher_source_kinds:
        provenance = state.get("provenance") or {}
        expected_teacher = teacher_source_kinds[str(state["source_kind"])]
        if provenance.get("teacher") != expected_teacher:
            report.add(
                "provenance.teacher",
                f"{state['source_kind']} rows must identify {expected_teacher} as teacher",
            )
    if PROJECT_MARKER_RE.search(_canonical_json(state)):
        report.add("genericity", "contains a project-specific marker")
    cell = state.get("matrix_cell")
    if isinstance(cell, Mapping) and generic_matrix_cell_id(cell) != state.get("matrix_cell_id"):
        report.add("matrix_cell_id", "does not match generic matrix cell")
    if "type_signature" in state and state.get("type_signature") != _type_signature(state):
        report.add("type_signature", "does not match canonical generic type signature")
    if "instance_signature" in state and state.get("instance_signature") != _instance_signature(state):
        report.add("instance_signature", "does not match canonical generic instance signature")
    return report


def write_generic_jsonl(rows: Iterable[Mapping[str, Any]], output: Path | str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
