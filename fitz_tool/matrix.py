"""Matrix configuration, conditional validity rules, and balanced materialization."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .uniqueness import canonical_json, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "configs" / "matrix.v1.json"
MATRIX_VERSION = "matrix.v1"

DIMENSION_NAMES = (
    "integration_domain",
    "information_operation",
    "source_modality",
    "evidence_topology",
    "retrieval_obstacle",
    "agent_state",
    "next_tool_target",
    "governance_path",
    "terminal_condition",
    "resource_pressure_band",
)

PDF_TOOLS = {
    "list_pdf_sources",
    "inspect_pdf_structure",
    "search_pdf_pages",
}
TABLE_TOOLS = {
    "list_tabular_sources",
    "inspect_table_schema",
    "search_table_rows",
}
CODE_TOOLS = {"read_file", "inspect_code"}
RETRIEVAL_TOOLS = {
    "search_bm25",
    "grep_search",
    "search_metadata",
    "list_sources",
    "search_table_rows",
    "search_pdf_pages",
    "read_file",
    "inspect_code",
}
INITIAL_DISCOVERY_TOOLS = {
    "list_sources",
    "list_tabular_sources",
    "inspect_table_schema",
    "list_pdf_sources",
    "inspect_pdf_structure",
}
NON_TERMINAL_TOOLS = set(
    {
        "set_retrieval_plan",
        "search_bm25",
        "grep_search",
        "search_metadata",
        "list_sources",
        "list_tabular_sources",
        "inspect_table_schema",
        "search_table_rows",
        "list_pdf_sources",
        "inspect_pdf_structure",
        "search_pdf_pages",
        "read_file",
        "inspect_code",
        "inspect_evidence",
        "expand_context",
        "compare_evidence",
        "update_requirement_progress",
        "assess_evidence",
    }
)
TERMINAL_PROFILE = (
    "ongoing",
    "ongoing",
    "ongoing",
    "ongoing",
    "ongoing",
    "ongoing",
    "ongoing",
    "selection",
    "selection",
    "abstention",
    "abstention",
    "clarification",
    "unresolved_contradiction",
    "step_limit_termination",
)


@dataclass(frozen=True)
class MatrixCell:
    """One legal point in the generation matrix."""

    cell_id: str
    values: Mapping[str, str]

    @property
    def matrix_version(self) -> str:
        return MATRIX_VERSION

    def as_dict(self) -> dict[str, str]:
        return {"cell_id": self.cell_id, **dict(self.values)}


def load_matrix_spec(path: Path | str = DEFAULT_MATRIX_PATH) -> dict[str, Any]:
    """Load and minimally validate the versioned matrix configuration."""

    matrix_path = Path(path)
    spec = json.loads(matrix_path.read_text(encoding="utf-8"))
    if spec.get("matrix_version") != MATRIX_VERSION:
        raise ValueError(f"expected {MATRIX_VERSION}, got {spec.get('matrix_version')!r}")
    dimensions = spec.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("matrix dimensions must be an object")
    missing = [name for name in DIMENSION_NAMES if name not in dimensions]
    if missing:
        raise ValueError(f"matrix is missing dimensions: {', '.join(missing)}")
    for name in DIMENSION_NAMES:
        values = dimensions[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"dimension {name!r} must contain unique non-empty values")
    profile = spec.get("sampling_profile", {}).get("terminal_condition_profile", [])
    if profile and any(value not in dimensions["terminal_condition"] for value in profile):
        raise ValueError("sampling profile contains an unknown terminal condition")
    return spec


def validate_matrix_cell(
    values: Mapping[str, Any],
    spec: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return deterministic validity errors for one matrix cell."""

    spec = spec or load_matrix_spec()
    dimensions = spec["dimensions"]
    errors: list[str] = []

    for name in DIMENSION_NAMES:
        if name not in values:
            errors.append(f"missing dimension {name}")
            continue
        value = values[name]
        if value not in dimensions[name]:
            errors.append(f"{name}={value!r} is not in the controlled vocabulary")

    if errors:
        return errors

    modality = values["source_modality"]
    topology = values["evidence_topology"]
    state = values["agent_state"]
    target = values["next_tool_target"]
    governance = values["governance_path"]
    terminal = values["terminal_condition"]
    pressure = values["resource_pressure_band"]
    operation = values["information_operation"]

    if target in PDF_TOOLS and modality not in {"pdf", "mixed"}:
        errors.append(f"{target} requires pdf or mixed source_modality")
    if target in TABLE_TOOLS and modality not in {"csv", "excel", "sqlite", "mixed"}:
        errors.append(f"{target} requires a structured source_modality")
    if target in CODE_TOOLS and modality not in {"code", "mixed"}:
        errors.append(f"{target} requires code or mixed source_modality")
    if topology == "cross_format" and modality != "mixed":
        errors.append("cross_format evidence_topology requires mixed source_modality")
    if topology == "contradictory" and state not in {"contradiction", "disputed"}:
        errors.append("contradictory evidence must be represented by contradiction or disputed state")
    if topology == "absent" and state not in {"no_hits", "insufficient"}:
        errors.append("absent evidence must be represented by no_hits or insufficient state")

    if target == "compare_evidence" and operation not in {
        "compare",
        "join",
        "compatibility",
        "contradiction",
        "latest_value_selection",
    }:
        errors.append("compare_evidence requires a comparison-oriented information operation")
    if target == "assess_evidence" and state in {"initial", "no_hits"}:
        errors.append("assess_evidence requires observed or exhausted retrieval state")
    if target == "update_requirement_progress" and state in {"initial", "no_hits"}:
        errors.append("update_requirement_progress requires tracked evidence or progress")

    if governance == "sufficient" and state != "fresh_sufficient":
        errors.append("sufficient governance requires fresh_sufficient state")
    if governance in {"insufficient_to_sufficient", "disputed_to_sufficient"} and state != "fresh_sufficient":
        errors.append(f"{governance} requires fresh_sufficient state")
    if governance == "repeated_insufficient" and state not in {"no_hits", "insufficient"}:
        errors.append("repeated_insufficient requires no_hits or insufficient state")
    if governance == "sufficient_to_disputed" and state not in {"contradiction", "disputed"}:
        errors.append("sufficient_to_disputed requires contradiction or disputed state")

    if terminal == "selection":
        if target != "finalize_document_selection":
            errors.append("selection terminal condition requires finalize_document_selection")
        if state != "fresh_sufficient":
            errors.append("selection requires fresh_sufficient state")
        if governance not in {"sufficient", "insufficient_to_sufficient", "disputed_to_sufficient"}:
            errors.append("selection requires a sufficient governance path")
    elif terminal == "abstention":
        if target != "finalize_document_selection":
            errors.append("abstention terminal condition requires finalize_document_selection")
        if state not in {"no_hits", "insufficient"}:
            errors.append("abstention requires no_hits or insufficient state")
        if governance != "repeated_insufficient":
            errors.append("abstention requires repeated_insufficient governance")
    elif terminal == "clarification":
        if target != "finalize_document_selection":
            errors.append("clarification terminal condition requires finalize_document_selection")
        if state not in {"insufficient", "disputed"}:
            errors.append("clarification requires insufficient or disputed state")
    elif terminal == "unresolved_contradiction":
        if target != "finalize_document_selection":
            errors.append("unresolved_contradiction requires finalize_document_selection")
        if state not in {"contradiction", "disputed"}:
            errors.append("unresolved_contradiction requires contradiction or disputed state")
        if governance != "sufficient_to_disputed":
            errors.append("unresolved_contradiction requires sufficient_to_disputed governance")
    elif terminal == "step_limit_termination":
        if target != "finalize_document_selection":
            errors.append("step_limit_termination requires finalize_document_selection")
        if pressure != "high":
            errors.append("step_limit_termination requires high resource pressure")
    elif terminal == "ongoing":
        if target == "finalize_document_selection":
            errors.append("ongoing decision state cannot finalize")
        if state == "fresh_sufficient":
            errors.append("fresh_sufficient state must use a terminal condition")

    if state == "initial" and target not in {
        "set_retrieval_plan",
        *RETRIEVAL_TOOLS,
        *INITIAL_DISCOVERY_TOOLS,
    }:
        errors.append("initial state must plan or retrieve")
    return errors


def make_cell(values: Mapping[str, str]) -> MatrixCell:
    """Validate values and create a stable cell identifier."""

    errors = validate_matrix_cell(values)
    if errors:
        raise ValueError("invalid matrix cell: " + "; ".join(errors))
    normalized = {name: str(values[name]) for name in DIMENSION_NAMES}
    return MatrixCell(cell_id=f"cell-{stable_hash(normalized)[:16]}", values=normalized)


def _choice(rng: random.Random, dimensions: Mapping[str, list[str]], name: str) -> str:
    return rng.choice(dimensions[name])


def _modality_for_target(
    rng: random.Random,
    dimensions: Mapping[str, list[str]],
    target: str,
    topology: str,
) -> str:
    if topology == "cross_format":
        return "mixed"
    if target in PDF_TOOLS:
        return rng.choice(["pdf", "mixed"])
    if target in TABLE_TOOLS:
        return rng.choice(["csv", "excel", "sqlite", "mixed"])
    if target in CODE_TOOLS:
        return rng.choice(["code", "mixed"])
    return _choice(rng, dimensions, "source_modality")


def _operation_for_state(
    rng: random.Random,
    dimensions: Mapping[str, list[str]],
    state: str,
    topology: str,
) -> str:
    if state in {"contradiction", "disputed"}:
        return rng.choice(["compare", "join", "compatibility", "contradiction", "latest_value_selection"])
    if topology == "absent":
        return rng.choice(["absence", "lookup", "enumerate"])
    return _choice(rng, dimensions, "information_operation")


def _target_pool(
    state: str,
    operation: str,
    modality: str,
    terminal: str,
) -> list[str]:
    if terminal != "ongoing":
        return ["finalize_document_selection"]
    state_tools = {
        "initial": (
            "set_retrieval_plan",
            "search_bm25",
            "grep_search",
            "search_metadata",
            "list_sources",
            "list_tabular_sources",
            "inspect_table_schema",
            "list_pdf_sources",
            "inspect_pdf_structure",
            "read_file",
            "inspect_code",
        ),
        "no_hits": (
            "set_retrieval_plan",
            "search_bm25",
            "grep_search",
            "search_metadata",
            "list_tabular_sources",
            "list_pdf_sources",
            "read_file",
            "inspect_code",
        ),
        "noisy_hits": (
            "search_bm25",
            "grep_search",
            "search_metadata",
            "inspect_evidence",
            "expand_context",
            "compare_evidence",
            "assess_evidence",
        ),
        "partial_evidence": (
            "search_bm25",
            "grep_search",
            "search_metadata",
            "inspect_evidence",
            "expand_context",
            "compare_evidence",
            "update_requirement_progress",
            "assess_evidence",
        ),
        "expansion_needed": (
            "search_bm25",
            "grep_search",
            "search_metadata",
            "inspect_evidence",
            "expand_context",
            "compare_evidence",
            "update_requirement_progress",
            "assess_evidence",
        ),
        "contradiction": (
            "inspect_evidence",
            "expand_context",
            "compare_evidence",
            "update_requirement_progress",
            "assess_evidence",
        ),
        "insufficient": (
            "search_bm25",
            "grep_search",
            "search_metadata",
            "inspect_evidence",
            "expand_context",
            "compare_evidence",
            "update_requirement_progress",
            "assess_evidence",
        ),
        "disputed": (
            "inspect_evidence",
            "expand_context",
            "compare_evidence",
            "update_requirement_progress",
            "assess_evidence",
        ),
    }
    candidates = list(state_tools[state])
    if operation in {"compare", "join", "compatibility", "contradiction", "latest_value_selection"}:
        candidates = ["compare_evidence", *candidates]
    if operation in {"lookup", "enumerate", "absence"}:
        candidates = ["search_bm25", "grep_search", "search_metadata", *candidates]
    if modality == "pdf":
        candidates = ["list_pdf_sources", "inspect_pdf_structure", "search_pdf_pages", *candidates]
    elif modality in {"csv", "excel", "sqlite"}:
        candidates = ["list_tabular_sources", "inspect_table_schema", "search_table_rows", *candidates]
    elif modality == "code":
        candidates = ["read_file", "inspect_code", *candidates]
    return [
        tool
        for tool in dict.fromkeys(candidates)
        if tool in NON_TERMINAL_TOOLS and (
            (tool in PDF_TOOLS and modality in {"pdf", "mixed"})
            or (tool in TABLE_TOOLS and modality in {"csv", "excel", "sqlite", "mixed"})
            or (tool in CODE_TOOLS and modality in {"code", "mixed"})
            or tool not in PDF_TOOLS | TABLE_TOOLS | CODE_TOOLS
        )
        and not (tool == "compare_evidence" and operation not in {
            "compare", "join", "compatibility", "contradiction", "latest_value_selection"
        })
        and not (tool == "assess_evidence" and state in {"initial", "no_hits"})
        and not (tool == "update_requirement_progress" and state in {"initial", "no_hits"})
    ]


def _preferred_target(
    state: str,
    operation: str,
    modality: str,
    terminal: str,
    obstacle: str,
) -> str:
    if terminal != "ongoing":
        return "finalize_document_selection"
    if state == "initial" and operation == "enumerate":
        return "list_sources"
    if modality == "pdf":
        if state == "initial":
            return "list_pdf_sources"
        if state in {"no_hits", "expansion_needed"}:
            return "search_pdf_pages"
        if state == "partial_evidence":
            return "inspect_pdf_structure"
    if modality in {"csv", "excel", "sqlite"}:
        if state == "initial":
            return "list_tabular_sources"
        if state == "no_hits":
            return "search_table_rows"
        if state == "partial_evidence":
            return "inspect_table_schema"
    if modality == "code":
        if state in {"initial", "no_hits"}:
            return "read_file"
        if state in {"partial_evidence", "expansion_needed"}:
            return "inspect_code"
    if state == "initial" and obstacle == "none":
        return "list_sources"
    if modality == "mixed":
        if state == "initial" and operation in {"enumerate", "absence"}:
            return "list_tabular_sources"
        if state == "initial" and obstacle in {"acronym", "identifier_variation"}:
            return "list_tabular_sources"
        if state == "initial" and operation in {
            "compare", "join", "compatibility", "contradiction", "latest_value_selection"
        }:
            return "list_pdf_sources"
        if state == "initial" and obstacle in {"long_document_needle", "version_noise"}:
            return "list_pdf_sources"
        if state == "no_hits" and operation == "enumerate":
            return "search_table_rows"
        if state == "partial_evidence" and operation == "latest_value_selection":
            return "assess_evidence"
    if state == "initial":
        if operation == "enumerate":
            return "search_metadata"
        if operation in {"compare", "join", "compatibility", "contradiction", "latest_value_selection"}:
            return "set_retrieval_plan"
        return "search_bm25"
    if state == "no_hits":
        if obstacle in {"acronym", "identifier_variation"}:
            return "grep_search"
        if operation == "enumerate":
            return "search_metadata"
        return "search_bm25"
    if state == "noisy_hits":
        if operation in {"compare", "join", "compatibility", "contradiction", "latest_value_selection"}:
            return "compare_evidence"
        return "inspect_evidence"
    if state == "partial_evidence":
        if operation == "latest_value_selection":
            return "assess_evidence"
        if operation in {"compare", "join", "compatibility", "contradiction", "latest_value_selection"}:
            return "compare_evidence"
        if operation == "lookup":
            return "update_requirement_progress"
        if operation == "enumerate":
            return "search_metadata"
        return "inspect_evidence"
    if state == "expansion_needed":
        return "expand_context"
    if state in {"contradiction", "disputed"}:
        return "compare_evidence"
    if state == "insufficient":
        if operation in {"compare", "join", "compatibility", "contradiction", "latest_value_selection"}:
            return "compare_evidence"
        if obstacle in {"acronym", "identifier_variation"}:
            return "grep_search"
        if operation == "enumerate":
            return "search_metadata"
        return "search_bm25"
    raise ValueError(f"unknown state for target preference: {state}")


def _sample_candidate(
    rng: random.Random,
    dimensions: Mapping[str, list[str]],
    terminal: str,
    allowed_source_modalities: set[str] | None = None,
) -> dict[str, str]:
    domain = _choice(rng, dimensions, "integration_domain")
    obstacle = _choice(rng, dimensions, "retrieval_obstacle")
    pressure = _choice(rng, dimensions, "resource_pressure_band")

    if terminal == "ongoing":
        state = rng.choice(
            [
                "initial",
                "no_hits",
                "noisy_hits",
                "partial_evidence",
                "expansion_needed",
                "contradiction",
                "insufficient",
                "disputed",
            ]
        )
        if state in {"contradiction", "disputed"}:
            topology = rng.choice(["contradictory", "complementary_sources", "multiple_passages", "cross_format"])
        elif state == "no_hits":
            topology = rng.choice(["absent", "one_passage", "multiple_passages", "complementary_sources"])
        else:
            topology = rng.choice(["one_passage", "multiple_passages", "complementary_sources", "cross_format"])
        governance = (
            rng.choice(["sufficient_to_disputed", "stale_assessment"])
            if state in {"contradiction", "disputed"}
            else rng.choice(["repeated_insufficient", "stale_assessment"])
            if state in {"no_hits", "insufficient"}
            else "stale_assessment"
        )
    elif terminal == "selection":
        state = "fresh_sufficient"
        governance = rng.choice(["sufficient", "insufficient_to_sufficient", "disputed_to_sufficient"])
        topology = rng.choice(["one_passage", "multiple_passages", "complementary_sources", "cross_format"])
    elif terminal == "abstention":
        state = rng.choice(["no_hits", "insufficient"])
        governance = "repeated_insufficient"
        topology = rng.choice(["absent", "multiple_passages", "complementary_sources"])
    elif terminal == "clarification":
        state = rng.choice(["insufficient", "disputed"])
        governance = rng.choice(["repeated_insufficient", "stale_assessment"])
        topology = "contradictory" if state == "disputed" else rng.choice(
            ["absent", "multiple_passages", "complementary_sources"]
        )
    elif terminal == "unresolved_contradiction":
        state = rng.choice(["contradiction", "disputed"])
        governance = "sufficient_to_disputed"
        topology = "contradictory"
    elif terminal == "step_limit_termination":
        state = rng.choice(["noisy_hits", "partial_evidence", "expansion_needed", "contradiction", "disputed"])
        governance = "sufficient_to_disputed" if state in {"contradiction", "disputed"} else "stale_assessment"
        topology = "contradictory" if state in {"contradiction", "disputed"} else rng.choice(
            ["multiple_passages", "complementary_sources", "cross_format"]
        )
        pressure = "high"
    else:
        raise ValueError(f"unknown terminal profile: {terminal}")

    if topology == "cross_format":
        modality = "mixed"
    elif allowed_source_modalities:
        modality = rng.choice(sorted(allowed_source_modalities))
    else:
        modality = _choice(rng, dimensions, "source_modality")
    operation = _operation_for_state(rng, dimensions, state, topology)
    target_pool = _target_pool(state, operation, modality, terminal)
    if not target_pool:
        raise RuntimeError(f"no legal target pool for state={state}, operation={operation}, modality={modality}")
    preferred_target = _preferred_target(state, operation, modality, terminal, obstacle)
    target = preferred_target if preferred_target in target_pool else target_pool[0]
    return {
        "integration_domain": domain,
        "information_operation": operation,
        "source_modality": modality,
        "evidence_topology": topology,
        "retrieval_obstacle": obstacle,
        "agent_state": state,
        "next_tool_target": target,
        "governance_path": governance,
        "terminal_condition": terminal,
        "resource_pressure_band": pressure,
    }


def materialize_cells(
    count: int,
    *,
    seed: int = 20260823,
    spec: Mapping[str, Any] | None = None,
    max_attempts: int | None = None,
    excluded_cell_ids: set[str] | None = None,
    allowed_source_modalities: set[str] | None = None,
    allowed_evidence_topologies: set[str] | None = None,
    allowed_agent_states: set[str] | None = None,
    allowed_terminal_conditions: set[str] | None = None,
    ensure_target_coverage: bool = True,
) -> list[MatrixCell]:
    """Materialize unique, conditionally legal cells reproducibly.

    Dimensions such as next-tool target and terminal condition are coupled, so
    independent per-axis balancing creates starvation. Rejection sampling over
    the legal space preserves those constraints; coverage is reported and can
    be checked against batch quotas by the caller.
    """

    if count < 1:
        raise ValueError("count must be positive")
    spec = spec or load_matrix_spec()
    dimensions = spec["dimensions"]
    rng = random.Random(seed)
    cells: list[MatrixCell] = []
    seen: set[str] = set()
    excluded_cell_ids = excluded_cell_ids or set()
    attempts = 0
    limit = max_attempts or max(100_000, count * 2_000)

    profile = list(
        spec.get("sampling_profile", {}).get("terminal_condition_profile", TERMINAL_PROFILE)
    )
    if allowed_terminal_conditions:
        profile = [value for value in profile if value in allowed_terminal_conditions]
    if not profile:
        raise ValueError("sampling profile has no terminal conditions allowed by the filters")
    rng.shuffle(profile)

    def allowed(candidate: Mapping[str, str]) -> bool:
        if allowed_source_modalities and candidate["source_modality"] not in allowed_source_modalities:
            return False
        if allowed_evidence_topologies and candidate["evidence_topology"] not in allowed_evidence_topologies:
            return False
        if allowed_agent_states and candidate["agent_state"] not in allowed_agent_states:
            return False
        if allowed_terminal_conditions and candidate["terminal_condition"] not in allowed_terminal_conditions:
            return False
        return True

    def append_candidate(candidate: dict[str, str]) -> bool:
        if not allowed(candidate) or validate_matrix_cell(candidate, spec):
            return False
        signature = canonical_json(candidate)
        if signature in seen:
            return False
        cell = make_cell(candidate)
        if cell.cell_id in excluded_cell_ids:
            return False
        seen.add(signature)
        cells.append(cell)
        return True

    def target_can_use_allowed_modality(target: str) -> bool:
        if not allowed_source_modalities:
            return True
        if target in PDF_TOOLS:
            return bool(allowed_source_modalities & {"pdf", "mixed"})
        if target in TABLE_TOOLS:
            return bool(allowed_source_modalities & {"csv", "excel", "sqlite", "mixed"})
        if target in CODE_TOOLS:
            return bool(allowed_source_modalities & {"code", "mixed"})
        return True

    if ensure_target_coverage:
        # Seed one legal cell for each reachable tool target before filling the
        # remainder randomly.  Rejection is bounded because some caller
        # filters can make a target unreachable (for example PDF tools in a
        # text-only pilot).
        anchor_attempts = max(1_000, min(10_000, count * 20))
        for target in dimensions["next_tool_target"]:
            if len(cells) >= count or not target_can_use_allowed_modality(target):
                continue
            found = False
            for _ in range(anchor_attempts):
                for terminal in profile:
                    candidate = _sample_candidate(
                        rng,
                        dimensions,
                        terminal,
                        allowed_source_modalities,
                    )
                    if candidate["next_tool_target"] != target:
                        continue
                    if append_candidate(candidate):
                        found = True
                        break
                if found or len(cells) >= count:
                    break

    while len(cells) < count and attempts < limit:
        attempts += 1
        terminal = profile[attempts % len(profile)]
        candidate = _sample_candidate(rng, dimensions, terminal, allowed_source_modalities)
        append_candidate(candidate)

    if len(cells) != count:
        raise RuntimeError(
            f"could materialize only {len(cells)} of {count} legal unique cells "
            f"after {attempts} attempts"
        )
    return cells


def coverage(cells: list[MatrixCell]) -> dict[str, dict[str, int]]:
    """Return per-dimension value counts for a materialized slice."""

    report: dict[str, dict[str, int]] = {name: {} for name in DIMENSION_NAMES}
    for cell in cells:
        for name, value in cell.values.items():
            report[name][value] = report[name].get(value, 0) + 1
    return report
