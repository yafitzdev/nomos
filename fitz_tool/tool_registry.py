"""Versioned, agent-agnostic tool registry contracts for router.v2."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


TOOL_REGISTRY_VERSION = "tool-registry.v2"
TOOL_ID_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,127}$")
TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
SIDE_EFFECT_CLASSES = {
    "none",
    "read",
    "local_state_write",
    "external_write",
    "irreversible_external_write",
}


class RegistryValidationError(ValueError):
    """Raised when a registry cannot be normalized safely."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(self.issues))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_strings(value: Any, field: str, issues: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        issues.append(f"{field} must be a non-empty list")
        return ()
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not TOKEN_RE.fullmatch(item):
            issues.append(f"{field}[{index}] must be a lowercase registry token")
            continue
        output.append(item)
    if len(output) != len(set(output)):
        issues.append(f"{field} must not contain duplicate values")
    return tuple(output)


def _validate_schema(schema: Any, path: str, issues: list[str]) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        issues.append(f"{path} must be an object")
        return {}
    normalized = dict(schema)
    if normalized.get("type") != "object":
        issues.append(f"{path}.type must equal object")
    properties = normalized.get("properties", {})
    if not isinstance(properties, Mapping):
        issues.append(f"{path}.properties must be an object")
    required = normalized.get("required", [])
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        issues.append(f"{path}.required must be a list of property names")
    elif isinstance(properties, Mapping):
        unknown = sorted(set(required) - set(properties))
        if unknown:
            issues.append(f"{path}.required references unknown properties: {', '.join(unknown)}")
    return normalized


@dataclass(frozen=True)
class ToolSpec:
    """Canonical metadata used to score one concrete external tool."""

    tool_id: str
    tool_family: str
    description: str
    capabilities: tuple[str, ...]
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    evidence_roles: tuple[str, ...]
    side_effect_class: str
    argument_schema: Mapping[str, Any]
    constraints: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolSpec":
        issues: list[str] = []
        tool_id = value.get("tool_id")
        if not isinstance(tool_id, str) or not TOOL_ID_RE.fullmatch(tool_id):
            issues.append("tool_id must be a stable registry identifier")
            tool_id = "invalid"
        family = value.get("tool_family")
        if not isinstance(family, str) or not TOKEN_RE.fullmatch(family):
            issues.append("tool_family must be a lowercase registry token")
            family = "invalid"
        description = value.get("description")
        if not isinstance(description, str) or len(description.strip()) < 12:
            issues.append("description must contain at least 12 characters")
            description = "invalid tool"
        capabilities = _normalized_strings(value.get("capabilities"), "capabilities", issues)
        inputs = _normalized_strings(value.get("input_modalities"), "input_modalities", issues)
        outputs = _normalized_strings(value.get("output_modalities"), "output_modalities", issues)
        roles = _normalized_strings(value.get("evidence_roles"), "evidence_roles", issues)
        side_effect = value.get("side_effect_class")
        if side_effect not in SIDE_EFFECT_CLASSES:
            issues.append(
                "side_effect_class must be one of " + ", ".join(sorted(SIDE_EFFECT_CLASSES))
            )
            side_effect = "none"
        schema = _validate_schema(value.get("argument_schema"), "argument_schema", issues)
        constraints = _normalized_strings(
            value.get("constraints", ["none"]), "constraints", issues
        )
        prerequisites = _normalized_strings(
            value.get("prerequisites", ["none"]), "prerequisites", issues
        )
        if issues:
            raise RegistryValidationError(issues)
        return cls(
            tool_id=tool_id,
            tool_family=family,
            description=description.strip(),
            capabilities=capabilities,
            input_modalities=inputs,
            output_modalities=outputs,
            evidence_roles=roles,
            side_effect_class=side_effect,
            argument_schema=schema,
            constraints=constraints,
            prerequisites=prerequisites,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "tool_family": self.tool_family,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "input_modalities": list(self.input_modalities),
            "output_modalities": list(self.output_modalities),
            "evidence_roles": list(self.evidence_roles),
            "side_effect_class": self.side_effect_class,
            "argument_schema": dict(self.argument_schema),
            "constraints": list(self.constraints),
            "prerequisites": list(self.prerequisites),
        }

    def semantic_dict(self) -> dict[str, Any]:
        """Return scoring metadata with the concrete tool identity removed."""

        value = self.as_dict()
        value.pop("tool_id")
        return value

    @property
    def semantic_fingerprint(self) -> str:
        return _fingerprint(self.semantic_dict())


@dataclass(frozen=True)
class ToolRegistry:
    """Validated collection of tool specifications supplied by an agent."""

    registry_id: str
    tools: tuple[ToolSpec, ...]
    schema_version: str = TOOL_REGISTRY_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolRegistry":
        issues: list[str] = []
        if value.get("schema_version") != TOOL_REGISTRY_VERSION:
            issues.append(f"schema_version must equal {TOOL_REGISTRY_VERSION}")
        registry_id = value.get("registry_id")
        if not isinstance(registry_id, str) or not TOOL_ID_RE.fullmatch(registry_id):
            issues.append("registry_id must be a stable registry identifier")
            registry_id = "invalid"
        raw_tools = value.get("tools")
        if not isinstance(raw_tools, list) or not raw_tools:
            issues.append("tools must be a non-empty list")
            raw_tools = []
        tools: list[ToolSpec] = []
        for index, raw_tool in enumerate(raw_tools):
            if not isinstance(raw_tool, Mapping):
                issues.append(f"tools[{index}] must be an object")
                continue
            try:
                tools.append(ToolSpec.from_dict(raw_tool))
            except RegistryValidationError as exc:
                issues.extend(f"tools[{index}].{issue}" for issue in exc.issues)
        tool_ids = [tool.tool_id for tool in tools]
        if len(tool_ids) != len(set(tool_ids)):
            issues.append("tools must not contain duplicate tool_id values")
        if issues:
            raise RegistryValidationError(issues)
        return cls(registry_id=registry_id, tools=tuple(tools))

    @property
    def by_id(self) -> dict[str, ToolSpec]:
        return {tool.tool_id: tool for tool in self.tools}

    def require(self, tool_id: str) -> ToolSpec:
        try:
            return self.by_id[tool_id]
        except KeyError as exc:
            raise RegistryValidationError([f"unknown tool_id: {tool_id}"]) from exc

    def resolve(self, tool_ids: Iterable[str]) -> tuple[ToolSpec, ...]:
        return tuple(self.require(tool_id) for tool_id in tool_ids)

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "tools": [tool.as_dict() for tool in self.tools],
        }
        if include_fingerprint:
            value["registry_fingerprint"] = self.fingerprint
        return value

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.as_dict(include_fingerprint=False))


def load_tool_registry(path: Path | str) -> ToolRegistry:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RegistryValidationError(["registry root must be an object"])
    registry = ToolRegistry.from_dict(value)
    claimed = value.get("registry_fingerprint")
    if claimed is not None and claimed != registry.fingerprint:
        raise RegistryValidationError(["registry_fingerprint does not match canonical registry"])
    return registry
