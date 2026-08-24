"""Independent, frozen matrix-cell sampling for Nomos agentic data v2."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "configs" / "matrix.agentic.v2.json"
MATRIX_VERSION = "matrix.agentic.v2"


def load_matrix_v2(path: Path | str = MATRIX_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("matrix_version") != MATRIX_VERSION:
        raise ValueError(f"expected {MATRIX_VERSION}")
    return value


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _weighted_values(
    count: int, weights: Mapping[str, float], *, seed: int, numeric: bool = False
) -> list[Any]:
    names = list(weights)
    raw = [count * float(weights[name]) for name in names]
    allocations = [int(value) for value in raw]
    for index in sorted(range(len(names)), key=lambda item: raw[item] - allocations[item], reverse=True)[
        : count - sum(allocations)
    ]:
        allocations[index] += 1
    output: list[Any] = []
    for name, allocation in zip(names, allocations):
        value: Any = int(name) if numeric else name
        output.extend([value] * allocation)
    random.Random(seed).shuffle(output)
    return output


def _balanced_values(count: int, values: list[Any], *, seed: int) -> list[Any]:
    output = [values[index % len(values)] for index in range(count)]
    random.Random(seed).shuffle(output)
    return output


def _digest(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_cell(cell: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    kind = cell.get("task_kind")
    if kind not in {"route", "recover", "verify"}:
        issues.append("invalid task_kind")
    if kind == "verify":
        if not cell.get("validation_case"):
            issues.append("verify requires validation_case")
        if cell.get("candidate_outcome") != "not_applicable":
            issues.append("verify candidate_outcome must be not_applicable")
    else:
        if cell.get("validation_case") != "not_applicable":
            issues.append("routing rows cannot have validation_case")
    if kind == "recover":
        if not cell.get("recovery_trigger") or int(cell.get("recovery_round", 0)) < 1:
            issues.append("recover requires trigger and positive round")
        if int(cell.get("prior_candidate_count", 0)) < 1:
            issues.append("recover requires prior candidates")
    elif cell.get("recovery_trigger") != "none" or int(cell.get("recovery_round", 0)) != 0:
        issues.append("non-recovery rows cannot contain recovery state")
    expected = cell.get("expected_action")
    if kind == "verify" and expected not in {"accept_tool_call", "reject_tool_call"}:
        issues.append("verify expected_action must accept or reject")
    if cell.get("candidate_outcome") == "no_suitable_candidate" and expected != "abstain":
        issues.append("no-suitable-candidate rows must abstain")
    return issues


def generate_matrix_cells(count: int, *, seed: int = 20260827) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate independently shuffled axes with split-specific freeze namespaces."""

    if count < 100:
        raise ValueError("v2 matrix slices require at least 100 rows")
    spec = load_matrix_v2()
    axes: dict[str, list[Any]] = {
        "evaluation_partition": _weighted_values(
            count, spec["split_ratios"], seed=_stable_seed(seed, "partition")
        ),
        "task_kind": _weighted_values(count, spec["task_mix"], seed=_stable_seed(seed, "task")),
    }
    for name, weights in spec["weighted_dimensions"].items():
        axes[name] = _weighted_values(
            count,
            weights,
            seed=_stable_seed(seed, name),
            numeric=name in {"candidate_pool_size", "top_k"},
        )
    for name, values in spec["dimensions"].items():
        axes[name] = _balanced_values(count, values, seed=_stable_seed(seed, name))

    recovery_rows = sum(value == "recover" for value in axes["task_kind"])
    verify_rows = sum(value == "verify" for value in axes["task_kind"])
    recovery_triggers = _balanced_values(
        recovery_rows, spec["recovery"]["triggers"], seed=_stable_seed(seed, "recovery_trigger")
    )
    recovery_rounds = _balanced_values(
        recovery_rows, spec["recovery"]["rounds"], seed=_stable_seed(seed, "recovery_round")
    )
    prior_counts = _balanced_values(
        recovery_rows,
        spec["recovery"]["prior_candidate_counts"],
        seed=_stable_seed(seed, "prior_count"),
    )
    validation_cases = _balanced_values(
        verify_rows, spec["verification_cases"], seed=_stable_seed(seed, "validation")
    )

    cells = []
    recovery_index = verify_index = 0
    split_ordinals: Counter[str] = Counter()
    for index in range(count):
        cell = {name: values[index] for name, values in axes.items()}
        partition = str(cell["evaluation_partition"])
        ordinal = split_ordinals[partition]
        split_ordinals[partition] += 1
        cell.update(
            {
                "registry_profile": f"{partition}_registry_{ordinal % 64:03d}",
                "question_template_group": f"{partition}_templates_{ordinal % 12:02d}",
                "scenario_group_id": f"{partition}_scenario_{ordinal:07d}",
            }
        )
        kind = cell["task_kind"]
        if kind == "recover":
            cell.update(
                {
                    "recovery_trigger": recovery_triggers[recovery_index],
                    "recovery_round": recovery_rounds[recovery_index],
                    "prior_candidate_count": prior_counts[recovery_index],
                    "validation_case": "not_applicable",
                }
            )
            recovery_index += 1
        elif kind == "verify":
            validation_case = validation_cases[verify_index]
            cell.update(
                {
                    "recovery_trigger": "none",
                    "recovery_round": 0,
                    "prior_candidate_count": 0,
                    "candidate_outcome": "not_applicable",
                    "validation_case": validation_case,
                    "expected_action": (
                        "accept_tool_call" if validation_case == "valid_call" else "reject_tool_call"
                    ),
                }
            )
            verify_index += 1
        else:
            cell.update(
                {
                    "recovery_trigger": "none",
                    "recovery_round": 0,
                    "prior_candidate_count": 0,
                    "validation_case": "not_applicable",
                }
            )
        if kind != "verify":
            cell["expected_action"] = (
                "abstain"
                if cell["candidate_outcome"] == "no_suitable_candidate"
                else "recommend_tools"
            )
        issues = validate_cell(cell)
        if issues:
            raise RuntimeError(f"invalid v2 matrix cell {index}: {issues}")
        cell["matrix_cell_id"] = _digest(cell)
        cells.append(cell)

    combinations = Counter(
        (str(cell["task_kind"]), int(cell["candidate_pool_size"])) for cell in cells
    )
    report = {
        "matrix_version": MATRIX_VERSION,
        "count": count,
        "seed": seed,
        "partition_counts": dict(Counter(str(cell["evaluation_partition"]) for cell in cells)),
        "task_counts": dict(Counter(str(cell["task_kind"]) for cell in cells)),
        "task_pool_combinations": {
            f"{kind}|{pool}": combinations[(kind, pool)]
            for kind in ("route", "recover", "verify")
            for pool in (5, 10, 30, 100)
        },
        "unique_cell_ids": len({cell["matrix_cell_id"] for cell in cells}),
    }
    return cells, report
