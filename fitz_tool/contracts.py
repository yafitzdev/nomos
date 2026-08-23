"""Pure-Python validation for the versioned Fitz-Tool data contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .matrix import (
    CODE_TOOLS,
    DIMENSION_NAMES,
    PDF_TOOLS,
    TABLE_TOOLS,
    load_matrix_spec,
    validate_matrix_cell,
)
from .uniqueness import instance_signature, type_signature


TOOL_NAMES = set(load_matrix_spec()["dimensions"]["next_tool_target"])
TERMINAL_STATES = set(load_matrix_spec()["dimensions"]["terminal_condition"])
AGENT_STATES = set(load_matrix_spec()["dimensions"]["agent_state"])


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "severity": self.severity}


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(self, path: str, message: str, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(path, message, severity))

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "issues": [issue.as_dict() for issue in self.issues]}


def _required(report: ValidationReport, value: Mapping[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        if key not in value:
            report.add(key, "missing required field")


def _string(report: ValidationReport, value: Any, path: str, *, minimum: int = 1) -> None:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        report.add(path, f"must be a string with at least {minimum} characters")


def _list_of_strings(report: ValidationReport, value: Any, path: str, *, minimum: int = 0) -> None:
    if not isinstance(value, list) or len(value) < minimum or any(not isinstance(item, str) for item in value):
        report.add(path, f"must be a list of strings with at least {minimum} items")
        return
    if len(value) != len(set(value)):
        report.add(path, "must not contain duplicate values")


def _sha256(report: ValidationReport, value: Any, path: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        report.add(path, "must be a lowercase SHA-256 hex digest")


def validate_source_card(card: Mapping[str, Any]) -> ValidationReport:
    report = ValidationReport()
    _required(
        report,
        card,
        ("schema_version", "source_id", "document_id", "title", "modality", "content_sha256", "facts"),
    )
    if card.get("schema_version") != "source-card.v1":
        report.add("schema_version", "must equal source-card.v1")
    for key in ("source_id", "document_id", "title"):
        if key in card:
            _string(report, card[key], key)
    if card.get("modality") not in set(load_matrix_spec()["dimensions"]["source_modality"]):
        report.add("modality", "is not in the controlled source modality vocabulary")
    if "content_sha256" in card:
        _sha256(report, card["content_sha256"], "content_sha256")
    if "normalized_content_sha256" in card:
        _sha256(report, card["normalized_content_sha256"], "normalized_content_sha256")
    facts = card.get("facts")
    if not isinstance(facts, list) or not facts:
        report.add("facts", "must be a non-empty list")
    else:
        fact_ids: set[str] = set()
        for index, fact in enumerate(facts):
            path = f"facts[{index}]"
            if not isinstance(fact, Mapping):
                report.add(path, "must be an object")
                continue
            _required(report, fact, ("fact_id", "statement"))
            if "fact_id" in fact:
                _string(report, fact["fact_id"], f"{path}.fact_id")
                if fact["fact_id"] in fact_ids:
                    report.add(f"{path}.fact_id", "duplicate fact_id")
                fact_ids.add(fact["fact_id"])
            if "statement" in fact:
                _string(report, fact["statement"], f"{path}.statement", minimum=5)
    return report


def validate_scenario(scenario: Mapping[str, Any]) -> ValidationReport:
    report = ValidationReport()
    _required(
        report,
        scenario,
        (
            "schema_version",
            "scenario_id",
            "matrix_version",
            "matrix_cell",
            "source_card_ids",
            "state_setup",
            "question",
            "difficult_paraphrase",
            "expected_facts",
            "expected_tools",
            "expected_terminal_state",
            "type_signature",
            "instance_signature",
            "provenance",
        ),
    )
    if scenario.get("schema_version") != "scenario.v1":
        report.add("schema_version", "must equal scenario.v1")
    if scenario.get("matrix_version") != "matrix.v1":
        report.add("matrix_version", "must equal matrix.v1")
    for key in ("scenario_id", "question", "difficult_paraphrase"):
        if key in scenario:
            _string(report, scenario[key], key, minimum=10 if key != "scenario_id" else 1)

    source_ids = scenario.get("source_card_ids")
    _list_of_strings(report, source_ids, "source_card_ids", minimum=1)
    if isinstance(scenario.get("matrix_cell"), Mapping):
        cell = scenario["matrix_cell"]
        missing = [name for name in DIMENSION_NAMES if name not in cell]
        if missing:
            report.add("matrix_cell", f"missing dimensions: {', '.join(missing)}")
        else:
            for message in validate_matrix_cell(cell):
                report.add("matrix_cell", message)
    else:
        report.add("matrix_cell", "must be an object")

    state_setup = scenario.get("state_setup")
    if not isinstance(state_setup, Mapping):
        report.add("state_setup", "must be an object")
    else:
        _required(report, state_setup, ("state_name", "history", "observed_evidence", "requirements", "governance"))
        if "state_name" in state_setup:
            _string(report, state_setup["state_name"], "state_setup.state_name")
            if isinstance(scenario.get("matrix_cell"), Mapping) and state_setup["state_name"] != scenario["matrix_cell"].get("agent_state"):
                report.add("state_setup.state_name", "must match matrix_cell.agent_state")
        for key in ("history", "observed_evidence", "requirements"):
            if key in state_setup and not isinstance(state_setup[key], list):
                report.add(f"state_setup.{key}", "must be a list")
        governance_setup = state_setup.get("governance")
        if not isinstance(governance_setup, Mapping):
            report.add("state_setup.governance", "must be an object")
        else:
            if not isinstance(governance_setup.get("assessment_fresh"), bool):
                report.add("state_setup.governance.assessment_fresh", "must be boolean")
            _string(report, governance_setup.get("path"), "state_setup.governance.path")

    expected_tools = scenario.get("expected_tools")
    _list_of_strings(report, expected_tools, "expected_tools", minimum=1)
    if isinstance(expected_tools, list):
        for index, tool in enumerate(expected_tools):
            if tool not in TOOL_NAMES:
                report.add(f"expected_tools[{index}]", f"unknown or illegal tool: {tool}")
            if isinstance(scenario.get("matrix_cell"), Mapping):
                modality = scenario["matrix_cell"].get("source_modality")
                if tool in PDF_TOOLS and modality not in {"pdf", "mixed"}:
                    report.add(f"expected_tools[{index}]", "PDF tool is incompatible with the cell source modality")
                if tool in TABLE_TOOLS and modality not in {"csv", "excel", "sqlite", "mixed"}:
                    report.add(f"expected_tools[{index}]", "table tool is incompatible with the cell source modality")
                if tool in CODE_TOOLS and modality not in {"code", "mixed"}:
                    report.add(f"expected_tools[{index}]", "code tool is incompatible with the cell source modality")
        if isinstance(scenario.get("matrix_cell"), Mapping):
            target = scenario["matrix_cell"].get("next_tool_target")
            if target not in expected_tools:
                report.add("expected_tools", "must include the matrix cell next_tool_target")
            if (
                scenario["matrix_cell"].get("terminal_condition") == "ongoing"
                and "finalize_document_selection" in expected_tools
            ):
                report.add("expected_tools", "ongoing cells must not include finalization")

    terminal = scenario.get("expected_terminal_state")
    if terminal not in TERMINAL_STATES:
        report.add("expected_terminal_state", "is not in the terminal condition vocabulary")
    if isinstance(scenario.get("matrix_cell"), Mapping) and terminal != scenario["matrix_cell"].get(
        "terminal_condition"
    ):
        report.add("expected_terminal_state", "must match matrix_cell.terminal_condition")

    expected_facts = scenario.get("expected_facts")
    if not isinstance(expected_facts, list):
        report.add("expected_facts", "must be a list")
    elif isinstance(scenario.get("matrix_cell"), Mapping) and scenario["matrix_cell"].get(
        "evidence_topology"
    ) == "absent":
        if expected_facts:
            report.add("expected_facts", "absent evidence cells must not claim positive source facts")
    elif expected_facts or (
        isinstance(scenario.get("matrix_cell"), Mapping)
        and scenario["matrix_cell"].get("terminal_condition") not in {"abstention", "clarification"}
    ):
        for index, fact in enumerate(expected_facts):
            if not isinstance(fact, Mapping):
                report.add(f"expected_facts[{index}]", "must be an object")
                continue
            if not fact.get("source_id") or not fact.get("fact_id"):
                report.add(f"expected_facts[{index}]", "requires source_id and fact_id")
            elif fact["source_id"] not in source_ids:
                report.add(f"expected_facts[{index}].source_id", "not listed in source_card_ids")

    provenance = scenario.get("provenance")
    if not isinstance(provenance, Mapping):
        report.add("provenance", "must be an object")
    else:
        _required(
            report,
            provenance,
            ("teacher", "model", "prompt_version", "seed", "source_card_hashes", "generated_at"),
        )
        for key in ("teacher", "model", "prompt_version", "generated_at"):
            if key in provenance:
                _string(report, provenance[key], f"provenance.{key}")
        if "seed" in provenance and not isinstance(provenance["seed"], int):
            report.add("provenance.seed", "must be an integer")
        hashes = provenance.get("source_card_hashes")
        _list_of_strings(report, hashes, "provenance.source_card_hashes", minimum=1)
        if isinstance(hashes, list):
            for index, value in enumerate(hashes):
                _sha256(report, value, f"provenance.source_card_hashes[{index}]")

    if isinstance(scenario.get("type_signature"), str):
        if scenario["type_signature"] != type_signature(dict(scenario)):
            report.add("type_signature", "does not match the canonical scenario type signature")
    else:
        report.add("type_signature", "must be a SHA-256 hex digest")
    if isinstance(scenario.get("instance_signature"), str):
        if scenario["instance_signature"] != instance_signature(dict(scenario)):
            report.add("instance_signature", "does not match the canonical scenario instance signature")
    else:
        report.add("instance_signature", "must be a SHA-256 hex digest")
    return report


def validate_trajectory(trajectory: Mapping[str, Any]) -> ValidationReport:
    report = ValidationReport()
    _required(
        report,
        trajectory,
        ("schema_version", "trajectory_id", "scenario_id", "runner", "events", "terminal_result", "provenance"),
    )
    if trajectory.get("schema_version") != "trajectory.v1":
        report.add("schema_version", "must equal trajectory.v1")
    for key in ("trajectory_id", "scenario_id"):
        if key in trajectory:
            _string(report, trajectory[key], key)
    if "question" in trajectory:
        _string(report, trajectory["question"], "question")
    runner = trajectory.get("runner")
    if not isinstance(runner, Mapping):
        report.add("runner", "must be an object")
    else:
        _required(report, runner, ("name", "version", "contract_version"))
        if runner.get("contract_version") != "runner.v1":
            report.add("runner.contract_version", "must equal runner.v1")
    events = trajectory.get("events")
    if not isinstance(events, list):
        report.add("events", "must be a list")
    else:
        previous_step = -1
        for index, event in enumerate(events):
            if not isinstance(event, Mapping):
                report.add(f"events[{index}]", "must be an object")
                continue
            step = event.get("step")
            if not isinstance(step, int) or step < previous_step:
                report.add(f"events[{index}].step", "must be a non-decreasing integer")
            if isinstance(step, int):
                previous_step = step
            if event.get("kind") not in {
                "state",
                "decision",
                "tool_call",
                "tool_result",
                "governance",
                "terminal",
                "error",
            }:
                report.add(f"events[{index}].kind", "unknown event kind")
            if event.get("kind") == "decision":
                _list_of_strings(report, event.get("legal_tools"), f"events[{index}].legal_tools", minimum=1)
                agent_state = event.get("agent_state") or event.get("state", {}).get("agent_state")
                if not isinstance(agent_state, Mapping) or agent_state.get("state_name") not in AGENT_STATES:
                    report.add(f"events[{index}].agent_state.state_name", "must be in the agent state vocabulary")
                governance = event.get("governance")
                if not isinstance(governance, Mapping) or not isinstance(
                    governance.get("assessment_fresh"), bool
                ):
                    report.add(f"events[{index}].governance", "must contain boolean assessment_fresh")
    if not isinstance(trajectory.get("terminal_result"), Mapping):
        report.add("terminal_result", "must be an object")
    provenance = trajectory.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("captured_at"):
        report.add("provenance", "must contain captured_at")
    return report


def validate_decision_state(state: Mapping[str, Any]) -> ValidationReport:
    report = ValidationReport()
    _required(
        report,
        state,
        (
            "schema_version",
            "decision_state_id",
            "trajectory_id",
            "scenario_id",
            "step",
            "question",
            "agent_state",
            "legal_tools",
            "observed_evidence",
            "governance",
            "label",
            "provenance",
        ),
    )
    if state.get("schema_version") != "decision-state.v1":
        report.add("schema_version", "must equal decision-state.v1")
    for key in ("decision_state_id", "trajectory_id", "scenario_id", "question"):
        if key in state:
            _string(report, state[key], key)
    if "step" in state and (not isinstance(state["step"], int) or state["step"] < 0):
        report.add("step", "must be a non-negative integer")
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, Mapping) or agent_state.get("state_name") not in AGENT_STATES:
        report.add("agent_state.state_name", "must be in the agent state vocabulary")
    legal_tools = state.get("legal_tools")
    _list_of_strings(report, legal_tools, "legal_tools", minimum=1)
    if isinstance(legal_tools, list):
        for index, tool in enumerate(legal_tools):
            if tool not in TOOL_NAMES:
                report.add(f"legal_tools[{index}]", f"unknown tool: {tool}")
    governance = state.get("governance")
    if not isinstance(governance, Mapping):
        report.add("governance", "must be an object")
    else:
        if not isinstance(governance.get("assessment_fresh"), bool):
            report.add("governance.assessment_fresh", "must be boolean")
        if not isinstance(governance.get("requirements"), list):
            report.add("governance.requirements", "must be a list")
    label = state.get("label")
    if not isinstance(label, Mapping):
        report.add("label", "must be an object")
    else:
        _list_of_strings(report, label.get("acceptable_tools"), "label.acceptable_tools")
        _list_of_strings(report, label.get("ranked_tools"), "label.ranked_tools")
        _list_of_strings(report, label.get("hard_negative_tools"), "label.hard_negative_tools")
        if label.get("label_source") != "deterministic_execution":
            report.add("label.label_source", "must equal deterministic_execution")
        acceptable = set(label.get("acceptable_tools", []))
        ranked = set(label.get("ranked_tools", []))
        negatives = set(label.get("hard_negative_tools", []))
        if not acceptable <= set(legal_tools or []):
            report.add("label.acceptable_tools", "must be a subset of legal_tools")
        if not ranked <= set(legal_tools or []):
            report.add("label.ranked_tools", "must be a subset of legal_tools")
        if not negatives <= set(legal_tools or []):
            report.add("label.hard_negative_tools", "must be a subset of legal_tools")
        if acceptable & negatives:
            report.add("label", "acceptable and hard-negative tools must be disjoint")
        if not acceptable and not negatives:
            report.add("label", "must contain an acceptable tool or a hard-negative tool")
    if not isinstance(state.get("accepted"), bool):
        report.add("accepted", "must be boolean")
    provenance = state.get("provenance")
    if not isinstance(provenance, Mapping):
        report.add("provenance", "must be an object")
    else:
        _sha256(report, provenance.get("trajectory_hash"), "provenance.trajectory_hash")
        _string(report, provenance.get("validator_version"), "provenance.validator_version")
    return report
