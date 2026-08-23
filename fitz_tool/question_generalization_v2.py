"""Question generalization transforms for the frozen router.v2 pilot.

The original 5,000-row pilot remains immutable.  This module creates a
derived training/evaluation view whose questions describe the required
evidence move indirectly, without copying the canonical target capability
label into the question.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .pilot_v2 import CAPABILITY_FOCUS, annotate_pilot_state
from .router_v2 import FEATURE_VERSION


QUESTION_GENERALIZATION_VERSION = "question-generalization.v1"
QUESTION_GENERALIZATION_TRAIN_TEMPLATE_IDS = (
    "qg_train_indirect",
    "qg_train_paraphrase",
    "qg_train_compositional",
    "qg_train_capability_implied",
    "qg_train_stateful",
    "qg_train_contrastive",
    "qg_train_evidence",
    "qg_train_constraint",
)
QUESTION_GENERALIZATION_HOLDOUT_TEMPLATE_IDS = (
    "qg_holdout_indirect",
    "qg_holdout_compositional",
    "qg_holdout_consequence",
    "qg_holdout_constraint",
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class IntentProfile:
    goal: str
    alternate_goal: str
    prerequisite: str
    contrast: str
    outcome: str


INTENT_PROFILES: dict[str, IntentProfile] = {
    "plan_retrieval": IntentProfile(
        "establish a deliberate sequence of evidence work",
        "decide how the investigation should begin",
        "the investigation has an explicit route before retrieval",
        "jumping into an unstructured lookup",
        "an ordered approach to the evidence",
    ),
    "list_sources": IntentProfile(
        "expose the available document set before choosing one",
        "learn what research material exists",
        "the source inventory is visible before selection",
        "opening an arbitrary document",
        "a complete source inventory",
    ),
    "search_content": IntentProfile(
        "locate the passage that contains the relevant claim",
        "find the right section of indexed material",
        "the relevant wording is found in the document body",
        "browsing unrelated source records",
        "a passage candidate",
    ),
    "exact_pattern_search": IntentProfile(
        "find a literal identifier with its spelling intact",
        "match the requested token exactly",
        "the token is preserved rather than normalized",
        "a broad semantic lookup",
        "an exact textual match",
    ),
    "search_metadata": IntentProfile(
        "narrow the document set using catalog information",
        "filter sources before reading their bodies",
        "the source constraints can be applied without body inspection",
        "searching full document text",
        "a filtered source set",
    ),
    "inspect_structured_schema": IntentProfile(
        "learn field names and data types before filtering",
        "map the shape of the structured source",
        "the table layout is known before a row query",
        "guessing field names in a row lookup",
        "a known field layout",
    ),
    "search_structured_records": IntentProfile(
        "retrieve rows matching field and value conditions",
        "find structured entries that satisfy the requested filters",
        "the needed fields and values are queryable",
        "inspecting the table definition again",
        "matching structured rows",
    ),
    "inspect_document_structure": IntentProfile(
        "map headings, sections, or page organization",
        "understand how the document is arranged before narrowing the search",
        "the document landmarks are known",
        "scanning page contents without a map",
        "a document outline",
    ),
    "search_document_pages": IntentProfile(
        "locate evidence within the selected page range",
        "search the relevant pages for the requested passage",
        "the page-level location of the evidence can be identified",
        "searching the entire source indiscriminately",
        "page-scoped evidence candidates",
    ),
    "read_content": IntentProfile(
        "open the selected source for direct inspection",
        "inspect the chosen source material itself",
        "the selected source contents are available",
        "choosing another source",
        "the source content",
    ),
    "inspect_code_structure": IntentProfile(
        "locate symbols, definitions, and relationships in the implementation",
        "trace how the relevant behavior is represented in code",
        "the implementation structure is visible",
        "reading prose without checking the implementation",
        "code-structure evidence",
    ),
    "inspect_evidence": IntentProfile(
        "check one candidate result deeply and trace its support",
        "verify the provenance of a candidate before synthesis",
        "the candidate's supporting evidence is inspected",
        "using a result on its label alone",
        "a verified evidence item",
    ),
    "expand_context": IntentProfile(
        "recover the surrounding material around a partial result",
        "resolve ambiguity by widening the evidence window",
        "the missing context is available around the snippet",
        "treating a clipped snippet as complete",
        "complete local context",
    ),
    "compare_evidence": IntentProfile(
        "check agreement or conflict across inspected claims",
        "reconcile multiple evidence items before deciding",
        "the claims can be examined side by side",
        "accepting the first claim without comparison",
        "an evidence comparison",
    ),
    "update_requirements": IntentProfile(
        "record evidence against each outstanding obligation",
        "track which requirements the current proof covers",
        "the current evidence can be attached to the open checklist",
        "leaving requirement progress implicit",
        "updated requirement coverage",
    ),
    "assess_evidence": IntentProfile(
        "decide whether collected proof is enough for every obligation",
        "test the evidence set against the research requirements",
        "sufficiency can be judged from the canonical evidence set",
        "declaring success from an unchecked snippet",
        "a sufficiency assessment",
    ),
    "finalize_selection": IntentProfile(
        "commit the chosen document set after a fresh sufficiency check",
        "close the investigation with the supported selection",
        "the evidence assessment is both fresh and sufficient",
        "finalizing while governance is stale",
        "a defensible final selection",
    ),
}


def _humanize(value: Any) -> str:
    return str(value).replace("_", " ").replace("-", " ")


def _question_terms(question: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(question)]


def _context(state: Mapping[str, Any]) -> dict[str, str]:
    cell = state.get("matrix_cell") or {}
    source = state.get("source_state") or {}
    agent = state.get("agent_state") or {}
    return {
        "domain": _humanize(cell.get("integration_domain", "API")),
        "operation": _humanize(cell.get("information_operation", "research")),
        "obstacle": _humanize(cell.get("retrieval_obstacle", "uncertainty")),
        "modality": _humanize(cell.get("source_modality", "mixed")),
        "phase": _humanize(agent.get("phase", "research")),
        "evidence_state": _humanize(source.get("inspection_state", "unknown")),
        "inventory": _humanize(source.get("inventory_state", "unknown")),
        "specificity": _humanize((state.get("query_state") or {}).get("specificity", "broad")),
    }


def render_question_generalization(
    state: Mapping[str, Any],
    target_capability: str,
    template_id: str,
) -> str:
    """Render a target-implied question without naming the target capability."""

    try:
        profile = INTENT_PROFILES[target_capability]
    except KeyError as exc:
        raise ValueError(f"unknown target capability: {target_capability}") from exc
    context = _context(state)
    if template_id == "qg_holdout_consequence":
        return (
            f"Given {context['evidence_state']} evidence in the {context['phase']} phase, "
            f"what would let the agent rely on {profile.outcome} without overclaiming?"
        )
    if template_id == "qg_holdout_constraint":
        return (
            f"With {context['modality']} material and a {context['inventory']} inventory, "
            f"which move handles the need to {profile.alternate_goal} without "
            f"{profile.contrast}?"
        )
    if template_id.endswith("indirect"):
        return (
            f"For the {context['domain']} request, what would help the agent "
            f"{profile.goal} when the task involves {context['operation']}?"
        )
    if template_id.endswith("paraphrase"):
        return (
            f"The user needs to {profile.alternate_goal}. Which evidence move fits "
            f"the current {context['phase']} situation?"
        )
    if template_id.endswith("compositional"):
        return (
            f"Given a {context['obstacle']} obstacle and {context['modality']} material, "
            f"what should happen next so the agent can {profile.goal}?"
        )
    if template_id.endswith("capability_implied"):
        return (
            f"The investigation is blocked until {profile.prerequisite}; "
            "which kind of step addresses that dependency?"
        )
    if template_id.endswith("stateful"):
        return (
            f"With {context['evidence_state']} evidence and a {context['inventory']} "
            f"source inventory, how should the agent proceed to {profile.goal}?"
        )
    if template_id.endswith("contrastive"):
        return (
            f"Which next action suits a need to {profile.goal}, rather than "
            f"{profile.contrast}?"
        )
    if template_id.endswith("evidence"):
        return (
            f"To avoid guessing about {context['domain']} behavior, what would "
            f"produce {profile.outcome} from the evidence already available?"
        )
    if template_id.endswith("constraint"):
        return (
            f"The task uses {context['modality']} material and a {context['specificity']} "
            f"query; what move can {profile.goal} while respecting the current "
            "evidence boundary?"
        )
    raise ValueError(f"unknown question-generalization template: {template_id}")


def canonical_question_leakage_markers(
    question: str,
    target_capability: str,
) -> list[str]:
    """Return canonical target phrases found verbatim in a question."""

    question_tokens = _question_terms(question)
    markers = {
        target_capability.casefold().replace("_", " "),
        target_capability.casefold(),
        CAPABILITY_FOCUS[target_capability].casefold(),
    }
    found: list[str] = []
    for marker in markers:
        marker_tokens = _question_terms(marker.replace("_", " "))
        if not marker_tokens:
            continue
        width = len(marker_tokens)
        if any(
            question_tokens[offset : offset + width] == marker_tokens
            for offset in range(len(question_tokens) - width + 1)
        ):
            found.append(marker)
    return sorted(found)


def _derived_seed(state: Mapping[str, Any], template_id: str, index: int) -> int:
    payload = f"{state.get('decision_state_id')}|{template_id}|{index}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _retag_state(
    state: Mapping[str, Any],
    *,
    question: str,
    template_id: str,
    index: int,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(state))
    target = str((output.get("sampling_context") or {}).get("target_capability"))
    leakage = canonical_question_leakage_markers(question, target)
    if leakage:
        raise ValueError(
            f"question template {template_id} leaks canonical target phrases: {leakage}"
        )
    derived_seed = _derived_seed(output, template_id, index)
    original_id = str(output.get("decision_state_id", f"row-{index:05d}"))
    output["decision_state_id"] = f"{original_id}-qg-{index:05d}"
    output["trajectory_id"] = f"{output.get('trajectory_id', original_id)}-qg-{index:05d}"
    output["scenario_id"] = f"{output.get('scenario_id', original_id)}-qg-{index:05d}"
    output["question"] = question
    output.setdefault("agent_state", {})["question_length_band"] = (
        "short" if len(question) < 100 else "long"
    )
    output.setdefault("query_state", {})["query_terms"] = _question_terms(question)[:8]
    output["question_template_id"] = template_id
    output["split_group_id"] = f"{output.get('split_group_id', 'unknown')}|{template_id}"
    provenance = dict(output.get("provenance") or {})
    provenance.update(
        {
            "prompt_version": f"{QUESTION_GENERALIZATION_VERSION}-{template_id}",
            "model": "deterministic_question_generalization_oracle",
            "artifact": "question-generalization-oracle.v1",
            "teacher": "deterministic_question_generalization_oracle",
            "seed": derived_seed,
            "feature_version": FEATURE_VERSION,
            "trajectory_hash": hashlib.sha256(
                f"{output['trajectory_id']}|{derived_seed}".encode("utf-8")
            ).hexdigest(),
        }
    )
    output["provenance"] = provenance
    return annotate_pilot_state(output)


def generate_question_generalization_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a derived dataset while leaving all non-question holdouts intact."""

    output: list[dict[str, Any]] = []
    template_counts: dict[str, int] = {}
    leakage: list[dict[str, Any]] = []
    transformed_train = transformed_holdout = 0
    for index, state in enumerate(rows):
        cohort = str(state.get("evaluation_cohort"))
        if cohort == "train":
            template_pool = QUESTION_GENERALIZATION_TRAIN_TEMPLATE_IDS
            transformed_train += 1
        elif cohort == "heldout_questions":
            template_pool = QUESTION_GENERALIZATION_HOLDOUT_TEMPLATE_IDS
            transformed_holdout += 1
        else:
            output.append(copy.deepcopy(dict(state)))
            continue
        target = str((state.get("sampling_context") or {}).get("target_capability"))
        template_id = template_pool[index % len(template_pool)]
        question = render_question_generalization(state, target, template_id)
        markers = canonical_question_leakage_markers(question, target)
        if markers:
            leakage.append(
                {
                    "decision_state_id": state.get("decision_state_id"),
                    "template_id": template_id,
                    "markers": markers,
                }
            )
        transformed = _retag_state(
            state,
            question=question,
            template_id=template_id,
            index=index,
        )
        output.append(transformed)
        template_counts[template_id] = template_counts.get(template_id, 0) + 1
    if leakage:
        raise ValueError(f"question-generalization leakage detected: {leakage[:3]}")
    return output, {
        "question_generalization_version": QUESTION_GENERALIZATION_VERSION,
        "rows": len(output),
        "transformed_train_rows": transformed_train,
        "transformed_heldout_question_rows": transformed_holdout,
        "template_counts": dict(sorted(template_counts.items())),
        "feature_version": FEATURE_VERSION,
        "training_template_ids": list(QUESTION_GENERALIZATION_TRAIN_TEMPLATE_IDS),
        "heldout_template_ids": list(QUESTION_GENERALIZATION_HOLDOUT_TEMPLATE_IDS),
    }


def build_question_generalization_training_view(
    frozen_rows: Iterable[Mapping[str, Any]],
    generalized_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Interleave original and generalized training questions.

    Evaluation rows occur once. Training rows occur twice, with the original
    and generalized wording adjacent so deterministic learning-curve subsets
    see both surfaces at every target capability.
    """

    frozen = [dict(row) for row in frozen_rows]
    generalized = [dict(row) for row in generalized_rows]
    if len(frozen) != len(generalized):
        raise ValueError("frozen and generalized views must have the same row count")
    output: list[dict[str, Any]] = []
    original_training = generalized_training = 0
    for original, transformed in zip(frozen, generalized):
        if original.get("evaluation_cohort") == "train":
            output.extend([copy.deepcopy(original), copy.deepcopy(transformed)])
            original_training += 1
            generalized_training += 1
        else:
            output.append(copy.deepcopy(transformed))
    return output, {
        "question_generalization_version": QUESTION_GENERALIZATION_VERSION,
        "rows": len(output),
        "original_training_rows": original_training,
        "generalized_training_rows": generalized_training,
        "evaluation_rows": len(output) - original_training - generalized_training,
        "training_rows": original_training + generalized_training,
        "feature_version": FEATURE_VERSION,
    }
