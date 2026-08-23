"""Extract deterministic router rows from the external trajectory contract."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import ValidationReport, validate_decision_state, validate_trajectory
from .uniqueness import stable_hash


VALIDATOR_VERSION = "decision-labels.v1"


def _agent_state(event: Mapping[str, Any]) -> dict[str, Any]:
    value = event.get("agent_state")
    if not isinstance(value, Mapping):
        state = event.get("state")
        value = state.get("agent_state") if isinstance(state, Mapping) else None
    if isinstance(value, Mapping):
        return dict(value)
    return {"state_name": "unknown"}


def _governance(event: Mapping[str, Any]) -> dict[str, Any]:
    value = event.get("governance")
    if isinstance(value, Mapping):
        return dict(value)
    return {"assessment_fresh": False, "requirements": []}


def extract_decision_states(
    trajectory: Mapping[str, Any],
    *,
    question: str | None = None,
) -> list[dict[str, Any]]:
    """Extract one row per external ``decision`` event.

    Positive labels are copied only from the runner's deterministic
    ``acceptable_tools`` field on an accepted trajectory. Failed trajectories
    are retained only when the runner supplies a safe legal hard negative.
    """

    trajectory_report = validate_trajectory(trajectory)
    if not trajectory_report.valid:
        raise ValueError(trajectory_report.as_dict())
    trajectory_hash = stable_hash(dict(trajectory))
    validation = trajectory.get("validation") or {}
    trajectory_accepted = bool(validation.get("trajectory_accepted"))
    output: list[dict[str, Any]] = []
    for event in trajectory.get("events", []):
        if not isinstance(event, Mapping) or event.get("kind") != "decision":
            continue
        legal_tools = [str(tool) for tool in event.get("legal_tools") or []]
        supplied_acceptable = [
            str(tool) for tool in event.get("acceptable_tools") or [] if str(tool) in legal_tools
        ]
        acceptable_tools = supplied_acceptable if trajectory_accepted else []
        hard_negative_tools = [
            str(tool) for tool in event.get("hard_negative_tools") or [] if str(tool) in legal_tools
        ]
        if not acceptable_tools and not hard_negative_tools:
            proposed = str(event.get("executed_tool") or event.get("proposed_tool") or "")
            if proposed in legal_tools:
                hard_negative_tools = [proposed]
        accepted = bool(acceptable_tools)
        ranked_tools = [
            str(tool) for tool in event.get("ranked_tools") or [] if str(tool) in legal_tools
        ]
        if not ranked_tools:
            ranked_tools = acceptable_tools or hard_negative_tools
        state = {
            "schema_version": "decision-state.v1",
            "decision_state_id": "decision_"
            + stable_hash(
                {"trajectory_hash": trajectory_hash, "step": event.get("step")}
            )[:24],
            "trajectory_id": str(trajectory["trajectory_id"]),
            "scenario_id": str(trajectory["scenario_id"]),
            "step": int(event.get("step", 0)),
            "question": question or str(trajectory.get("question") or ""),
            "agent_state": _agent_state(event),
            "history": list(event.get("history") or []),
            "plan": dict(event.get("plan") or {}) if isinstance(event.get("plan"), Mapping) else {},
            "matrix_context": (
                dict(event.get("matrix_context") or {})
                if isinstance(event.get("matrix_context"), Mapping)
                else {}
            ),
            "legal_tools": legal_tools,
            "observed_evidence": list(event.get("observed_evidence") or []),
            "governance": _governance(event),
            "label": {
                "acceptable_tools": acceptable_tools,
                "ranked_tools": ranked_tools,
                "hard_negative_tools": hard_negative_tools,
                "label_source": "deterministic_execution",
            },
            "accepted": accepted,
            "provenance": {
                "trajectory_hash": trajectory_hash,
                "validator_version": VALIDATOR_VERSION,
            },
        }
        report = validate_decision_state(state)
        if not report.valid:
            raise ValueError(report.as_dict())
        output.append(state)
    return output


def validate_and_extract(
    trajectory: Mapping[str, Any],
    *,
    question: str | None = None,
) -> tuple[list[dict[str, Any]], ValidationReport]:
    """Return rows and a report without raising for batch-import tooling."""

    report = validate_trajectory(trajectory)
    if not report.valid:
        return [], report
    try:
        return extract_decision_states(trajectory, question=question), report
    except ValueError as exc:
        report.add("decision_states", str(exc))
        return [], report
