"""General, capability-oriented generation matrix for router.v2."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_V2_PATH = PROJECT_ROOT / "configs" / "matrix.v2.json"
MATRIX_VERSION = "matrix.v2"

PDF_CAPABILITIES = {"inspect_document_structure", "search_document_pages"}
TABLE_CAPABILITIES = {"inspect_structured_schema", "search_structured_records"}
CODE_CAPABILITIES = {"inspect_code_structure"}
EVIDENCE_CAPABILITIES = {
    "inspect_evidence",
    "expand_context",
    "compare_evidence",
    "update_requirements",
    "assess_evidence",
    "finalize_selection",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def matrix_v2_cell_id(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(values)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatrixV2Cell:
    values: Mapping[str, Any]

    @property
    def cell_id(self) -> str:
        return matrix_v2_cell_id(self.values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


def load_matrix_v2_spec(path: Path | str = DEFAULT_MATRIX_V2_PATH) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("matrix_version") != MATRIX_VERSION:
        raise ValueError(f"expected {MATRIX_VERSION}, got {spec.get('matrix_version')!r}")
    dimensions = spec.get("dimensions")
    ranges = spec.get("observable_ranges")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("matrix.v2 dimensions must be a non-empty object")
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError("matrix.v2 observable_ranges must be a non-empty object")
    for group_name, group in (("dimensions", dimensions), ("observable_ranges", ranges)):
        for name, values in group.items():
            if not isinstance(values, list) or not values or len(values) != len(set(values)):
                raise ValueError(f"{group_name}.{name} must contain unique values")
    policy = spec.get("model_input_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("matrix.v2 requires model_input_policy")
    forbidden = {"target_capability", "terminal_outcome"}
    runtime_fields = set(policy.get("runtime_observable_fields") or [])
    if forbidden & runtime_fields:
        raise ValueError("sampling labels must not appear in runtime_observable_fields")
    return spec


def validate_matrix_v2_cell(
    values: Mapping[str, Any], spec: Mapping[str, Any] | None = None
) -> list[str]:
    spec = spec or load_matrix_v2_spec()
    dimensions = spec["dimensions"]
    ranges = spec["observable_ranges"]
    errors: list[str] = []
    for name, allowed in dimensions.items():
        if name not in values:
            errors.append(f"missing dimension {name}")
        elif values[name] not in allowed:
            errors.append(f"invalid {name}: {values[name]!r}")
    for name, allowed in ranges.items():
        if name not in values:
            errors.append(f"missing observable range {name}")
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

    if target in PDF_CAPABILITIES and modality not in {"pdf", "mixed"}:
        errors.append(f"{target} requires pdf or mixed source modality")
    if target in TABLE_CAPABILITIES and modality not in {"csv", "excel", "sqlite", "mixed"}:
        errors.append(f"{target} requires a structured or mixed source modality")
    if target in CODE_CAPABILITIES and modality not in {"code", "mixed"}:
        errors.append(f"{target} requires code or mixed source modality")
    if topology == "cross_format" and modality != "mixed":
        errors.append("cross_format evidence requires mixed source modality")
    if topology == "contradictory" and values["observed_evidence_count"] < 2:
        errors.append("contradictory evidence requires at least two observed evidence items")
    if topology == "absent_after_exhaustion" and values["prior_search_count"] < 2:
        errors.append("absence requires at least two materially different prior searches")
    if freshness == "fresh_for_current_evidence" and inspection in {"none", "snippets_only"}:
        errors.append("fresh assessment requires inspected evidence context")
    if freshness == "fresh_for_current_evidence" and progress not in {"complete", "disputed"}:
        errors.append("fresh assessment requires complete or disputed requirements")
    if target in EVIDENCE_CAPABILITIES and values["observed_evidence_count"] == 0:
        errors.append(f"{target} requires observed evidence")
    if target == "compare_evidence" and inspection != "multi_source_inspected":
        errors.append("compare_evidence requires multiple inspected sources")
    if target == "expand_context" and inspection not in {"snippets_only", "partial_context"}:
        errors.append("expand_context requires incomplete inspected context")
    if target == "assess_evidence" and inspection in {"none", "snippets_only"}:
        errors.append("assess_evidence requires fully inspected evidence")
    if target == "finalize_selection":
        if terminal != "selection":
            errors.append("finalize_selection requires selection terminal outcome")
        if freshness != "fresh_for_current_evidence" or progress != "complete":
            errors.append("finalize_selection requires fresh assessment and complete requirements")
    elif terminal == "selection":
        errors.append("selection terminal outcome requires finalize_selection target")
    if target == "list_sources" and values["source_inventory_state"] == "known":
        errors.append("list_sources is not optimal when source inventory is already known")
    if target == "plan_retrieval" and values["agent_phase"] not in {"planning", "retrieval"}:
        errors.append("plan_retrieval requires planning or retrieval phase")
    if values["remaining_steps"] == 0 and terminal == "ongoing":
        errors.append("zero remaining steps cannot have an ongoing terminal outcome")
    return errors


def _sample_values(rng: random.Random, spec: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        name: rng.choice(options) for name, options in spec["dimensions"].items()
    }
    values.update(
        {name: rng.choice(options) for name, options in spec["observable_ranges"].items()}
    )
    return values


def _find_valid_with_override(
    rng: random.Random,
    spec: Mapping[str, Any],
    override_name: str,
    override_value: Any,
    excluded: set[str],
) -> MatrixV2Cell | None:
    for _attempt in range(5000):
        values = _sample_values(rng, spec)
        values[override_name] = override_value
        cell = MatrixV2Cell(values)
        if cell.cell_id not in excluded and not validate_matrix_v2_cell(values, spec):
            return cell
    return None


def materialize_matrix_v2_cells(
    count: int,
    *,
    seed: int = 20260823,
    excluded_cell_ids: Iterable[str] = (),
    spec: Mapping[str, Any] | None = None,
) -> list[MatrixV2Cell]:
    if count < 1:
        raise ValueError("count must be positive")
    spec = spec or load_matrix_v2_spec()
    rng = random.Random(seed)
    used = set(excluded_cell_ids)
    output: list[MatrixV2Cell] = []

    coverage_values = [
        (name, value)
        for group in (spec["dimensions"], spec["observable_ranges"])
        for name, options in group.items()
        for value in options
    ]
    if count >= len(coverage_values):
        for name, value in coverage_values:
            cell = _find_valid_with_override(rng, spec, name, value, used)
            if cell is None:
                raise ValueError(f"could not materialize reachable coverage for {name}={value!r}")
            output.append(cell)
            used.add(cell.cell_id)

    attempts = 0
    while len(output) < count:
        attempts += 1
        if attempts > count * 10000:
            raise ValueError("could not materialize enough unique legal matrix.v2 cells")
        values = _sample_values(rng, spec)
        cell = MatrixV2Cell(values)
        if cell.cell_id in used or validate_matrix_v2_cell(values, spec):
            continue
        output.append(cell)
        used.add(cell.cell_id)
    return output


def matrix_v2_coverage(
    cells: Iterable[MatrixV2Cell], spec: Mapping[str, Any] | None = None
) -> dict[str, dict[str, int]]:
    spec = spec or load_matrix_v2_spec()
    names = [*spec["dimensions"], *spec["observable_ranges"]]
    output = {name: {} for name in names}
    for cell in cells:
        for name in names:
            value = str(cell.values[name])
            output[name][value] = output[name].get(value, 0) + 1
    return output
