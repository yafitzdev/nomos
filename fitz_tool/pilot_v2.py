"""Deterministic registry-aware pilot generation and strict provenance checks."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .generic_contracts import ContractReport, validate_decision_state_v2
from .matrix_v2 import MatrixV2Cell, load_matrix_v2_spec, validate_matrix_v2_cell
from .router_v2 import FEATURE_VERSION
from .tool_registry import ToolRegistry, ToolSpec, load_tool_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_VERSION = "pilot.v2"
PILOT_SEED = 20260823
FIXED_GENERATED_AT = "2026-08-23T00:00:00+00:00"
TARGET_CAPABILITIES = tuple(load_matrix_v2_spec()["dimensions"]["target_capability"])

COHORT_COUNTS = {
    "train": 3400,
    "validation": 400,
    "familiar_tools": 300,
    "unseen_tool_ids": 250,
    "id_renames": 100,
    "schema_variants": 100,
    "modality_variants": 100,
    "heldout_family": 150,
    "heldout_sources": 50,
    "heldout_questions": 50,
    "alternate_registry": 100,
}

SOURCE_BLUEPRINTS = (
    ("pilot-auth-text", "auth_reference.md", "text", "train_documents"),
    ("pilot-payments-text", "payments_reference.md", "text", "train_documents"),
    ("pilot-webhooks-text", "webhooks_reference.md", "text", "train_documents"),
    ("pilot-schema-csv", "schema_reference.csv", "csv", "train_documents"),
    ("pilot-migration-excel", "migration_reference.csv", "excel", "train_documents"),
    ("pilot-client-code", "client_reference.py", "code", "train_documents"),
    ("pilot-security-holdout", "holdout_security.md", "text", "heldout_documents"),
    ("pilot-reconciliation-holdout", "holdout_reconciliation.md", "text", "heldout_documents"),
    ("pilot-schema-holdout", "holdout_schema.csv", "csv", "heldout_documents"),
    ("pilot-migration-holdout", "holdout_migration.csv", "excel", "heldout_documents"),
    ("pilot-client-holdout", "holdout_client.py", "code", "heldout_documents"),
    ("pilot-pdf-holdout", "holdout_pdf.md", "pdf", "heldout_documents"),
)

TRAIN_TEMPLATE_IDS = ("train_direct", "train_paraphrase", "train_constraint")
HOLDOUT_TEMPLATE_IDS = ("holdout_indirect", "holdout_versioned")
PHASE_TO_STATE = {
    "planning": "initial",
    "source_discovery": "source_discovery",
    "retrieval": "retrieval",
    "inspection": "partial_evidence",
    "synthesis": "contradiction",
    "governance": "insufficient",
    "terminal_readiness": "fresh_sufficient",
}
DOMAIN_FOCUS = {
    "auth": "refresh-token behavior",
    "payments": "idempotent capture and refund rules",
    "webhooks": "signature verification and event replay",
    "schemas": "required API fields and compatibility",
    "errors": "documented error codes and recovery",
    "migrations": "version compatibility during migration",
    "security": "signed request verification",
    "reconciliation": "ledger and event consistency",
}
CAPABILITY_FOCUS = {
    "plan_retrieval": "a defensible retrieval plan",
    "list_sources": "the available research sources",
    "search_content": "the relevant document passage",
    "exact_pattern_search": "an exact identifier or phrase",
    "search_metadata": "source metadata and catalog filters",
    "inspect_structured_schema": "table fields and column types",
    "search_structured_records": "matching structured records",
    "inspect_document_structure": "document or page structure",
    "search_document_pages": "the relevant document pages",
    "read_content": "the selected source content",
    "inspect_code_structure": "source-code symbols and definitions",
    "inspect_evidence": "a candidate evidence item",
    "expand_context": "surrounding context for an incomplete snippet",
    "compare_evidence": "agreement or contradiction across evidence",
    "update_requirements": "tracked requirement coverage",
    "assess_evidence": "whether the evidence is sufficient",
    "finalize_selection": "the final document selection",
}


@dataclass(frozen=True)
class PilotSourceCard:
    source_id: str
    path: str
    modality: str
    corpus: str
    content_sha256: str
    document_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "source-card.v2",
            "source_id": self.source_id,
            "document_id": self.document_id,
            "title": self.document_id.replace("-", " ").title(),
            "modality": self.modality,
            "corpus": self.corpus,
            "path": self.path,
            "content_sha256": self.content_sha256,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def matrix_cell_id(cell: Mapping[str, Any]) -> str:
    return _sha256(dict(cell))


def pilot_type_signature(state: Mapping[str, Any]) -> str:
    registry = ToolRegistry.from_dict(state["tool_registry"])
    semantic_candidates = sorted(
        registry.require(tool_id).semantic_fingerprint
        for tool_id in state.get("legal_candidate_ids") or []
    )
    sampling = state.get("sampling_context") or {}
    payload = {
        "matrix_cell": state.get("matrix_cell"),
        "registry_semantics": semantic_candidates,
        "source_card_ids": sorted(state.get("source_card_ids") or []),
        "question_template_id": state.get("question_template_id"),
        "evaluation_cohort": state.get("evaluation_cohort"),
        "target_capability": sampling.get("target_capability"),
        "candidate_set_difficulty": sampling.get("candidate_set_difficulty"),
    }
    return _sha256(payload)


def pilot_instance_signature(state: Mapping[str, Any]) -> str:
    payload = {
        "type_signature": state.get("type_signature"),
        "question": state.get("question"),
        "history": state.get("history"),
        "observed_evidence": state.get("observed_evidence"),
        "seed": (state.get("provenance") or {}).get("seed"),
    }
    return _sha256(payload)


def annotate_pilot_state(state: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(state))
    output["matrix_version"] = "matrix.v2"
    output["matrix_cell_id"] = matrix_cell_id(output["matrix_cell"])
    output["pilot_version"] = PILOT_VERSION
    output["type_signature"] = pilot_type_signature(output)
    output["instance_signature"] = pilot_instance_signature(output)
    return output


def load_pilot_source_cards(
    root: Path | str = PROJECT_ROOT / "tests" / "fixtures" / "pilot_v2_corpus",
) -> dict[str, PilotSourceCard]:
    root = Path(root)
    cards: dict[str, PilotSourceCard] = {}
    for source_id, filename, modality, corpus in SOURCE_BLUEPRINTS:
        path = root / filename
        content = path.read_bytes()
        cards[source_id] = PilotSourceCard(
            source_id=source_id,
            path=str(path),
            modality=modality,
            corpus=corpus,
            content_sha256=hashlib.sha256(content).hexdigest(),
            document_id=source_id,
        )
    return cards


def load_pilot_registries(
    root: Path | str = PROJECT_ROOT,
) -> dict[str, ToolRegistry]:
    root = Path(root)
    return {
        "fitz_sage_v2": load_tool_registry(root / "configs" / "tool_registry.fitz_sage_v2.json"),
        "alternate_research_agent_v2": load_tool_registry(
            root / "configs" / "tool_registry.alternate_v2.json"
        ),
        "heldout_research_tools_v2": load_tool_registry(
            root / "configs" / "tool_registry.heldout_v2.json"
        ),
    }


def _registry_with_tools(registry_id: str, tools: Iterable[ToolSpec]) -> ToolRegistry:
    return ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": registry_id,
            "tools": [tool.as_dict() for tool in tools],
        }
    )


def renamed_registry(registry: ToolRegistry, prefix: str, *, alter_descriptions: bool = False) -> ToolRegistry:
    tools: list[ToolSpec] = []
    for tool in registry.tools:
        value = tool.as_dict()
        value["tool_id"] = f"{prefix}_{tool.tool_id}"
        if alter_descriptions:
            value["description"] = "A separate implementation that can " + tool.description[0].lower() + tool.description[1:]
        tools.append(ToolSpec.from_dict(value))
    return _registry_with_tools(f"{prefix}_registry", tools)


def modality_variant_registry(registry: ToolRegistry) -> ToolRegistry:
    tools: list[ToolSpec] = []
    for tool in registry.tools:
        value = tool.as_dict()
        if "search_content" in tool.capabilities:
            value["tool_id"] = f"modality_variant_{tool.tool_id}"
            value["input_modalities"] = sorted(set(tool.input_modalities) | {"mixed"})
            value["description"] = "A cross-format implementation that " + tool.description[0].lower() + tool.description[1:]
        tools.append(ToolSpec.from_dict(value))
    return _registry_with_tools("modality_variant_registry", tools)


def schema_variant_registry(registry: ToolRegistry) -> ToolRegistry:
    tools: list[ToolSpec] = []
    for tool in registry.tools:
        value = tool.as_dict()
        if "search_content" in tool.capabilities or "search_structured_records" in tool.capabilities:
            value["tool_id"] = f"schema_variant_{tool.tool_id}"
            schema = dict(value["argument_schema"])
            properties = dict(schema.get("properties") or {})
            properties["semantic_scope"] = {"type": "string"}
            schema["properties"] = properties
            value["argument_schema"] = schema
            value["description"] = "A schema-extended implementation that " + tool.description[0].lower() + tool.description[1:]
        tools.append(ToolSpec.from_dict(value))
    return _registry_with_tools("schema_variant_registry", tools)


def _source_ids_for_cell(
    cell: Mapping[str, Any], cards: Mapping[str, PilotSourceCard], cohort: str
) -> list[str]:
    modality = str(cell["source_modality"])
    holdout = cohort == "heldout_sources"
    pool = [
        card
        for card in cards.values()
        if (card.corpus == "heldout_documents") == holdout
    ]
    if not pool:
        raise ValueError(f"no source cards available for cohort {cohort}")
    matching = [card for card in pool if card.modality == modality]
    if modality == "mixed":
        matching = pool
    if not matching:
        matching = pool
    chosen = [matching[0]]
    if modality == "mixed":
        second = next((card for card in matching[1:] if card.modality != chosen[0].modality), None)
        if second is not None:
            chosen.append(second)
    return [card.source_id for card in chosen]


def _question(template_id: str, domain: str, target: str, operation: str) -> str:
    focus = DOMAIN_FOCUS[domain]
    capability_focus = CAPABILITY_FOCUS[target]
    if template_id == "train_direct":
        return f"Which documented action should the agent use to determine {capability_focus} while resolving {focus} for this {operation} request?"
    if template_id == "train_paraphrase":
        return f"What should the research agent do next to determine {capability_focus} while resolving the API question about {focus}?"
    if template_id == "train_constraint":
        return f"Given the available sources, which evidence operation can determine {capability_focus} and verify {focus} without guessing?"
    if template_id == "holdout_indirect":
        return f"Determine the next defensible research move for the versioned {domain} behavior described by the user, given the current evidence."
    return f"Resolve the integration question by finding the source evidence governing {focus}, taking the current research state into account."


def _valid_cell_for_target(
    rng: random.Random,
    target_capability: str,
    used_cell_ids: set[str],
    spec: Mapping[str, Any],
) -> MatrixV2Cell:
    dimensions = spec["dimensions"]
    ranges = spec["observable_ranges"]
    for _attempt in range(10000):
        values = {name: rng.choice(options) for name, options in dimensions.items()}
        values.update({name: rng.choice(options) for name, options in ranges.items()})
        values["target_capability"] = target_capability
        candidate = MatrixV2Cell(values)
        if candidate.cell_id not in used_cell_ids and not validate_matrix_v2_cell(values, spec):
            return candidate
    raise RuntimeError(f"could not sample a legal cell for target {target_capability}")


def _candidate_specs(
    registry: ToolRegistry,
    target_capability: str,
    difficulty: str,
    rng: random.Random,
) -> tuple[list[ToolSpec], list[ToolSpec]]:
    targets = [tool for tool in registry.tools if target_capability in tool.capabilities]
    if not targets:
        raise ValueError(f"registry {registry.registry_id} cannot provide {target_capability}")
    rng.shuffle(targets)
    acceptable = targets[: min(2, len(targets))]
    target_families = {tool.tool_family for tool in acceptable}
    target_capabilities = {capability for tool in acceptable for capability in tool.capabilities}
    remaining = [tool for tool in registry.tools if tool not in acceptable]
    same_family = [tool for tool in remaining if tool.tool_family in target_families]
    overlap = [
        tool for tool in remaining if set(tool.capabilities) & target_capabilities
    ]
    modality = [
        tool
        for tool in remaining
        if set(tool.input_modalities) & {modality for target in acceptable for modality in target.input_modalities}
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
    preferred_ids: set[str] = set()
    for tool in [*buckets.get(difficulty, remaining), *remaining]:
        if tool.tool_id not in preferred_ids:
            preferred.append(tool)
            preferred_ids.add(tool.tool_id)
    rng.shuffle(preferred)
    desired = min(len(registry.tools), rng.randint(5, 8))
    selected = acceptable + preferred[: max(1, desired - len(acceptable))]
    return selected, acceptable


def _observable_state(
    cell: Mapping[str, Any],
    source_ids: list[str],
    cards: Mapping[str, PilotSourceCard],
    question: str,
    rng: random.Random,
) -> dict[str, Any]:
    phase = str(cell["agent_phase"])
    evidence_count = int(cell["observed_evidence_count"])
    topology = str(cell["evidence_topology"])
    evidence: list[dict[str, Any]] = []
    for index in range(evidence_count):
        source_id = source_ids[index % len(source_ids)]
        evidence.append(
            {
                "evidence_id": f"ev-{rng.randrange(10**12):012d}",
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
            {"step": index, "action_family": "retrieval", "result": "weak"}
            for index in range(int(cell["prior_search_count"]))
        ],
        "plan": {
            "active": phase in {"planning", "retrieval"},
            "operation": cell["information_operation"],
        },
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
            "available_modalities": [
                "text", "pdf", "csv", "excel", "sqlite", "code", "metadata"
            ]
            if cell["source_modality"] == "mixed"
            else [cell["source_modality"], "metadata"],
            "inventory_state": cell["source_inventory_state"],
            "inspection_state": cell["evidence_inspection_state"],
            "source_ids": source_ids,
            "source_card_hashes": [cards[source_id].content_sha256 for source_id in source_ids],
            "evidence_topology_observed": topology if topology != "absent_after_exhaustion" else "absent",
        },
        "query_state": {
            "operation": cell["information_operation"],
            "specificity": cell["query_specificity"],
            "match_strategy": cell["match_strategy"],
            "query_terms": [token for token in question.lower().split() if len(token) > 4][:8],
        },
    }


def _build_state(
    *,
    index: int,
    cohort: str,
    target_capability: str,
    registry: ToolRegistry,
    cell: MatrixV2Cell,
    cards: Mapping[str, PilotSourceCard],
    template_id: str,
    split_group_id: str,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    source_ids = _source_ids_for_cell(cell.values, cards, cohort)
    domain = str(cell.values["integration_domain"])
    operation = str(cell.values["information_operation"])
    question = _question(template_id, domain, target_capability, operation)
    legal_specs, acceptable_specs = _candidate_specs(
        registry,
        target_capability,
        str(cell.values["candidate_set_difficulty"]),
        rng,
    )
    legal_ids = [tool.tool_id for tool in legal_specs]
    acceptable_ids = [tool.tool_id for tool in acceptable_specs if tool in legal_specs]
    hard_negatives = [tool_id for tool_id in legal_ids if tool_id not in acceptable_ids]
    state = _observable_state(cell.values, source_ids, cards, question, rng)
    state.update(
        {
            "schema_version": "decision-state.v2",
            "decision_state_id": f"pilot-v2-{index:05d}",
            "trajectory_id": f"pilot-v2-trajectory-{index:05d}",
            "scenario_id": f"pilot-v2-scenario-{index:05d}",
            "step": 0,
            "tool_registry": registry.as_dict(),
            "legal_candidate_ids": legal_ids,
            "label": {
                "acceptable_tools": acceptable_ids,
                "ranked_tools": acceptable_ids + hard_negatives,
                "hard_negative_tools": hard_negatives,
                "label_source": "deterministic_matrix_oracle.v2",
            },
            "accepted": True,
            "source_card_ids": source_ids,
            "source_kind": "deterministic_matrix_oracle",
            "evaluation_cohort": cohort,
            "evaluation_partition": "train" if cohort == "train" else "validation" if cohort == "validation" else "test",
            "split_group_id": split_group_id,
            "question_template_id": template_id,
            "matrix_cell": cell.as_dict(),
            "sampling_context": {
                "matrix_version": "matrix.v2",
                "target_capability": target_capability,
                "candidate_set_difficulty": cell.values["candidate_set_difficulty"],
                "terminal_outcome": cell.values["terminal_outcome"],
                "retrieval_obstacle": cell.values["retrieval_obstacle"],
                "integration_domain": cell.values["integration_domain"],
                "excluded_candidate_reasons": {
                    "modality_constraint": "recorded as an excluded illegal candidate when applicable",
                    "side_effect_constraint": "recorded as an excluded illegal candidate when applicable",
                },
            },
            "provenance": {
                "corpus": "nomos_router_v2_pilot",
                "source_card_hashes": [cards[source_id].content_sha256 for source_id in source_ids],
                "prompt_version": f"pilot-v2-{template_id}",
                "model": "deterministic_matrix_oracle",
                "artifact": "matrix-oracle.v2",
                "teacher": "deterministic_matrix_oracle",
                "seed": seed,
                "validator_version": "pilot-validator.v2",
                "feature_version": FEATURE_VERSION,
                "registry_fingerprint": registry.fingerprint,
                "trajectory_hash": _sha256({"trajectory_id": f"pilot-v2-trajectory-{index:05d}", "seed": seed}),
                "matrix_cell_id": cell.cell_id,
                "generated_at": FIXED_GENERATED_AT,
            },
        }
    )
    return annotate_pilot_state(state)


def _cohort_targets(cohort: str, count: int, registry: ToolRegistry) -> list[str]:
    available = [
        capability
        for capability in TARGET_CAPABILITIES
        if any(capability in tool.capabilities for tool in registry.tools)
    ]
    if not available:
        raise ValueError(f"no target capabilities available for {registry.registry_id}")
    return [available[index % len(available)] for index in range(count)]


def validate_pilot_state(state: Mapping[str, Any]) -> ContractReport:
    report = validate_decision_state_v2(state)
    required = {
        "matrix_version": "matrix.v2",
        "matrix_cell_id": None,
        "matrix_cell": None,
        "pilot_version": PILOT_VERSION,
        "type_signature": None,
        "instance_signature": None,
        "source_card_ids": None,
        "source_kind": None,
        "evaluation_cohort": None,
        "split_group_id": None,
        "question_template_id": None,
    }
    for key, expected in required.items():
        if key not in state:
            report.add(key, "missing pilot field")
        elif expected is not None and state.get(key) != expected:
            report.add(key, f"must equal {expected}")
    cell = state.get("matrix_cell")
    if isinstance(cell, Mapping):
        errors = validate_matrix_v2_cell(cell)
        for error in errors:
            report.add("matrix_cell", error)
        if state.get("matrix_cell_id") != matrix_cell_id(cell):
            report.add("matrix_cell_id", "does not match matrix_cell")
    if isinstance(state.get("source_card_ids"), list):
        if not state["source_card_ids"]:
            report.add("source_card_ids", "must not be empty")
        if len(state["source_card_ids"]) != len(set(state["source_card_ids"])):
            report.add("source_card_ids", "must not contain duplicates")
    provenance = state.get("provenance")
    if not isinstance(provenance, Mapping):
        report.add("provenance", "must be an object")
    else:
        for key in (
            "corpus", "source_card_hashes", "prompt_version", "model", "artifact", "teacher",
            "seed", "validator_version", "feature_version", "registry_fingerprint",
            "trajectory_hash", "matrix_cell_id", "generated_at",
        ):
            if key not in provenance:
                report.add(f"provenance.{key}", "missing pilot provenance field")
        if provenance.get("matrix_cell_id") != state.get("matrix_cell_id"):
            report.add("provenance.matrix_cell_id", "does not match state matrix_cell_id")
        if provenance.get("registry_fingerprint") != ToolRegistry.from_dict(state["tool_registry"]).fingerprint:
            report.add("provenance.registry_fingerprint", "does not match embedded registry")
    if "type_signature" in state and state.get("type_signature") != pilot_type_signature(state):
        report.add("type_signature", "does not match canonical pilot type signature")
    if "instance_signature" in state and state.get("instance_signature") != pilot_instance_signature(state):
        report.add("instance_signature", "does not match canonical pilot instance signature")
    if state.get("source_kind") == "deterministic_matrix_oracle" and state.get("label", {}).get("label_source") != "deterministic_matrix_oracle.v2":
        report.add("label.label_source", "matrix-oracle rows require deterministic_matrix_oracle.v2")
    return report


def generate_pilot_states(
    *,
    count: int = 5000,
    seed: int = PILOT_SEED,
    root: Path | str = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count != sum(COHORT_COUNTS.values()):
        raise ValueError(f"pilot.v2 requires exactly {sum(COHORT_COUNTS.values())} rows")
    root = Path(root)
    spec = load_matrix_v2_spec(root / "configs" / "matrix.v2.json")
    cards = load_pilot_source_cards(root / "tests" / "fixtures" / "pilot_v2_corpus")
    registries = load_pilot_registries(root)
    base = registries["fitz_sage_v2"]
    alternate = registries["alternate_research_agent_v2"]
    heldout = registries["heldout_research_tools_v2"]
    registry_by_cohort = {
        "train": base,
        "validation": base,
        "familiar_tools": base,
        "unseen_tool_ids": renamed_registry(base, "novel", alter_descriptions=True),
        "id_renames": renamed_registry(base, "renamed"),
        "schema_variants": schema_variant_registry(base),
        "modality_variants": modality_variant_registry(base),
        "heldout_family": heldout,
        "heldout_sources": base,
        "heldout_questions": base,
        "alternate_registry": alternate,
    }
    rng = random.Random(seed)
    used_cells: set[str] = set()
    used_types: set[str] = set()
    used_instances: set[str] = set()
    rows: list[dict[str, Any]] = []
    index = 0
    for cohort, cohort_count in COHORT_COUNTS.items():
        registry = registry_by_cohort[cohort]
        targets = _cohort_targets(cohort, cohort_count, registry)
        if cohort == "train":
            targets = [capability for capability in TARGET_CAPABILITIES for _ in range(200)]
        for offset, target in enumerate(targets):
            row_seed = seed + index * 1009 + offset
            cell = _valid_cell_for_target(rng, target, used_cells, spec)
            used_cells.add(cell.cell_id)
            template_pool = HOLDOUT_TEMPLATE_IDS if cohort == "heldout_questions" else TRAIN_TEMPLATE_IDS
            template_id = template_pool[offset % len(template_pool)]
            split_group = f"{cohort}|source-group-{offset % 8}|template-group-{offset % len(template_pool)}"
            row = _build_state(
                index=index,
                cohort=cohort,
                target_capability=target,
                registry=registry,
                cell=cell,
                cards=cards,
                template_id=template_id,
                split_group_id=split_group,
                seed=row_seed,
            )
            if row["type_signature"] in used_types:
                raise ValueError(f"duplicate pilot type signature at row {index}")
            if row["instance_signature"] in used_instances:
                raise ValueError(f"duplicate pilot instance signature at row {index}")
            used_types.add(row["type_signature"])
            used_instances.add(row["instance_signature"])
            rows.append(row)
            index += 1
    target_counts = Counter(
        str((row.get("sampling_context") or {}).get("target_capability")) for row in rows
    )
    manifest = {
        "pilot_version": PILOT_VERSION,
        "seed": seed,
        "count": len(rows),
        "cohort_counts": dict(Counter(row["evaluation_cohort"] for row in rows)),
        "target_capability_counts": dict(sorted(target_counts.items())),
        "registry_fingerprints": {
            registry_id: registry.fingerprint for registry_id, registry in registries.items()
        },
        "registry_variants": {
            "unseen_tool_ids": registry_by_cohort["unseen_tool_ids"].fingerprint,
            "id_renames": registry_by_cohort["id_renames"].fingerprint,
            "schema_variants": registry_by_cohort["schema_variants"].fingerprint,
            "modality_variants": registry_by_cohort["modality_variants"].fingerprint,
        },
        "source_cards": {source_id: card.as_dict() for source_id, card in cards.items()},
        "type_signatures": len(used_types),
        "instance_signatures": len(used_instances),
    }
    return rows, manifest


def write_pilot_jsonl(
    rows: Iterable[Mapping[str, Any]],
    output: Path | str,
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
