"""Pure-Python validation for the agent-agnostic runner.v2 contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .tool_registry import RegistryValidationError, ToolRegistry


RUNNER_REQUEST_VERSION = "runner-request.v2"
DECISION_STATE_VERSION = "decision-state.v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ContractIssue:
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "severity": "error"}


@dataclass
class ContractReport:
    issues: list[ContractIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def add(self, path: str, message: str) -> None:
        self.issues.append(ContractIssue(path, message))

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "issues": [issue.as_dict() for issue in self.issues]}


def _nonempty_string(report: ContractReport, value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        report.add(path, "must be a non-empty string")


def _object(report: ContractReport, value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        report.add(path, "must be an object")


def _list(report: ContractReport, value: Any, path: str) -> None:
    if not isinstance(value, list):
        report.add(path, "must be a list")


def registry_from_request(value: Mapping[str, Any]) -> ToolRegistry | None:
    raw_registry = value.get("tool_registry")
    if not isinstance(raw_registry, Mapping):
        return None
    try:
        return ToolRegistry.from_dict(raw_registry)
    except RegistryValidationError:
        return None


def validate_runner_request_v2(value: Mapping[str, Any]) -> ContractReport:
    """Validate one request without assuming any concrete agent or tool vocabulary."""

    report = ContractReport()
    if value.get("schema_version") != RUNNER_REQUEST_VERSION:
        report.add("schema_version", f"must equal {RUNNER_REQUEST_VERSION}")
    for key in ("request_id", "question"):
        _nonempty_string(report, value.get(key), key)
    for key in ("agent_state", "governance", "resource_state", "source_state", "query_state"):
        _object(report, value.get(key), key)
    for key in ("history", "observed_evidence"):
        _list(report, value.get(key), key)

    raw_registry = value.get("tool_registry")
    registry: ToolRegistry | None = None
    if not isinstance(raw_registry, Mapping):
        report.add("tool_registry", "must be an embedded tool-registry.v2 object")
    else:
        try:
            registry = ToolRegistry.from_dict(raw_registry)
        except RegistryValidationError as exc:
            for issue in exc.issues:
                report.add("tool_registry", issue)
        else:
            claimed = raw_registry.get("registry_fingerprint")
            if claimed is not None and claimed != registry.fingerprint:
                report.add("tool_registry.registry_fingerprint", "does not match registry content")

    candidate_ids = value.get("legal_candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        report.add("legal_candidate_ids", "must be a non-empty list")
    elif any(not isinstance(tool_id, str) or not tool_id for tool_id in candidate_ids):
        report.add("legal_candidate_ids", "must contain non-empty tool IDs")
    else:
        if len(candidate_ids) != len(set(candidate_ids)):
            report.add("legal_candidate_ids", "must not contain duplicate tool IDs")
        if registry is not None:
            unknown = sorted(set(candidate_ids) - set(registry.by_id))
            if unknown:
                report.add(
                    "legal_candidate_ids",
                    "contains IDs absent from the registry: " + ", ".join(unknown),
                )

    allowed_side_effects = (value.get("governance") or {}).get("allowed_side_effect_classes")
    if allowed_side_effects is not None and not isinstance(allowed_side_effects, list):
        report.add("governance.allowed_side_effect_classes", "must be a list when present")
    elif registry is not None and isinstance(candidate_ids, list) and allowed_side_effects:
        disallowed = [
            tool_id
            for tool_id in candidate_ids
            if tool_id in registry.by_id
            and registry.by_id[tool_id].side_effect_class not in set(allowed_side_effects)
        ]
        if disallowed:
            report.add(
                "legal_candidate_ids",
                "contains candidates disallowed by governance: " + ", ".join(disallowed),
            )
    return report


def validate_decision_state_v2(value: Mapping[str, Any]) -> ContractReport:
    report = validate_runner_request_v2(
        {**value, "schema_version": RUNNER_REQUEST_VERSION, "request_id": value.get("decision_state_id")}
    )
    if value.get("schema_version") != DECISION_STATE_VERSION:
        report.add("schema_version", f"must equal {DECISION_STATE_VERSION}")
    for key in ("decision_state_id", "trajectory_id", "scenario_id"):
        _nonempty_string(report, value.get(key), key)
    label = value.get("label")
    candidates = set(value.get("legal_candidate_ids") or [])
    if not isinstance(label, Mapping):
        report.add("label", "must be an object")
    else:
        for key in ("acceptable_tools", "ranked_tools", "hard_negative_tools"):
            items = label.get(key)
            if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                report.add(f"label.{key}", "must be a list of tool IDs")
                continue
            if len(items) != len(set(items)):
                report.add(f"label.{key}", "must not contain duplicates")
            if not set(items) <= candidates:
                report.add(f"label.{key}", "must be a subset of legal_candidate_ids")
        positives = set(label.get("acceptable_tools") or [])
        negatives = set(label.get("hard_negative_tools") or [])
        if positives & negatives:
            report.add("label", "acceptable and hard-negative tools must be disjoint")
        if not isinstance(label.get("label_source"), str) or not label.get("label_source"):
            report.add("label.label_source", "must identify the deterministic validator")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        report.add("provenance", "must be an object")
    else:
        for key in (
            "trajectory_hash",
            "registry_fingerprint",
            "matrix_cell_id",
            "feature_version",
            "validator_version",
        ):
            _nonempty_string(report, provenance.get(key), f"provenance.{key}")
        trajectory_hash = provenance.get("trajectory_hash")
        if isinstance(trajectory_hash, str) and not SHA256_RE.fullmatch(trajectory_hash):
            report.add("provenance.trajectory_hash", "must be a lowercase SHA-256 digest")
        registry = registry_from_request(value)
        if registry is not None and provenance.get("registry_fingerprint") != registry.fingerprint:
            report.add("provenance.registry_fingerprint", "does not match tool_registry")
    return report


def observable_router_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only fields that are observable at decision time.

    Sampling labels such as target capability, future governance path and terminal
    outcome are intentionally absent from this allowlist.
    """

    return {
        key: value.get(key)
        for key in (
            "question",
            "task_kind",
            "proposed_tool_call",
            "candidate_pool_size",
            "previous_candidate_ids",
            "expansion_context",
            "agent_state",
            "history",
            "plan",
            "observed_evidence",
            "governance",
            "resource_state",
            "source_state",
            "query_state",
        )
    }
