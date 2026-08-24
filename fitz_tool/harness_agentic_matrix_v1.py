"""Independent matrix sampling for heterogeneous agent-harness tools."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "configs" / "matrix.harness_agentic.v1.json"
MATRIX_VERSION = "matrix.harness-agentic.v1"


def load_matrix(path: Path | str = MATRIX_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("matrix_version") != MATRIX_VERSION:
        raise ValueError(f"expected {MATRIX_VERSION}")
    validate_matrix(value)
    return value


def validate_matrix(value: Mapping[str, Any]) -> None:
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("dimensions must be an object")
    capabilities = dimensions.get("target_capability")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("target_capability must be a non-empty list")
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("target_capability values must be unique")
    pairs = value.get("contrast_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("contrast_pairs must be a non-empty list")
    covered: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"contrast_pairs[{index}] must contain two distinct values")
        unknown = set(pair) - set(capabilities)
        if unknown:
            raise ValueError(f"contrast_pairs[{index}] contains unknown capabilities")
        covered.update(pair)
    missing = set(capabilities) - covered
    if missing:
        raise ValueError(f"target capabilities lack a contrast: {', '.join(sorted(missing))}")


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _balanced(count: int, values: list[Any], *, seed: int) -> list[Any]:
    output = [values[index % len(values)] for index in range(count)]
    random.Random(seed).shuffle(output)
    return output


def _weighted(
    count: int, weights: Mapping[str, float], *, seed: int, numeric: bool = False
) -> list[Any]:
    names = list(weights)
    raw = [count * float(weights[name]) for name in names]
    allocations = [int(value) for value in raw]
    remaining = count - sum(allocations)
    order = sorted(
        range(len(names)),
        key=lambda index: raw[index] - allocations[index],
        reverse=True,
    )
    for index in order[:remaining]:
        allocations[index] += 1
    output: list[Any] = []
    for name, allocation in zip(names, allocations):
        value: Any = int(name) if numeric else name
        output.extend([value] * allocation)
    random.Random(seed).shuffle(output)
    return output


def validate_cell(cell: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    kind = cell.get("task_kind")
    if kind not in {"route", "recover", "verify"}:
        issues.append("invalid task_kind")
    if cell.get("target_capability") == cell.get("hard_negative_capability"):
        issues.append("target and hard negative must differ")
    if kind == "recover":
        if cell.get("recovery_trigger") == "none":
            issues.append("recover requires a trigger")
        if int(cell.get("recovery_round", 0)) < 1:
            issues.append("recover requires a positive round")
        if int(cell.get("prior_candidate_count", 0)) < 1:
            issues.append("recover requires prior candidates")
        if cell.get("candidate_memory") != "accumulated_no_repeat":
            issues.append("recover requires accumulated no-repeat candidate memory")
    elif cell.get("recovery_trigger") != "none":
        issues.append("non-recovery rows cannot contain a recovery trigger")
    if kind == "verify" and cell.get("verification_case") == "not_applicable":
        issues.append("verify requires a verification case")
    if kind != "verify" and cell.get("verification_case") != "not_applicable":
        issues.append("routing rows cannot contain a verification case")
    return issues


def generate_matrix_cells(
    count: int, *, seed: int = 20260902
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate split-safe cells; this does not generate teacher-authored rows."""

    if count < 100:
        raise ValueError("harness-agentic matrix slices require at least 100 cells")
    spec = load_matrix()
    dimensions = spec["dimensions"]
    weighted = spec["weighted_dimensions"]
    axes: dict[str, list[Any]] = {
        "evaluation_partition": _weighted(
            count,
            spec["split_ratios"],
            seed=_stable_seed(seed, "evaluation_partition"),
        ),
        "task_kind": _weighted(
            count, spec["task_mix"], seed=_stable_seed(seed, "task_kind")
        ),
    }
    for name, values in dimensions.items():
        if name == "target_capability":
            continue
        axes[name] = _balanced(count, values, seed=_stable_seed(seed, name))
    for name, weights in weighted.items():
        axes[name] = _weighted(
            count,
            weights,
            seed=_stable_seed(seed, name),
            numeric=name in {"candidate_pool_size", "top_k"},
        )

    capabilities = dimensions["target_capability"]
    target_values = _balanced(
        count, capabilities, seed=_stable_seed(seed, "target_capability")
    )
    negative_options: dict[str, list[str]] = {value: [] for value in capabilities}
    for left, right in spec["contrast_pairs"]:
        negative_options[left].append(right)
        negative_options[right].append(left)

    recovery_count = axes["task_kind"].count("recover")
    verify_count = axes["task_kind"].count("verify")
    recovery_triggers = _balanced(
        recovery_count,
        spec["recovery"]["triggers"],
        seed=_stable_seed(seed, "recovery_trigger"),
    )
    recovery_rounds = _balanced(
        recovery_count,
        spec["recovery"]["rounds"],
        seed=_stable_seed(seed, "recovery_round"),
    )
    prior_counts = _balanced(
        recovery_count,
        spec["recovery"]["prior_candidate_counts"],
        seed=_stable_seed(seed, "prior_candidate_count"),
    )
    verification_cases = _balanced(
        verify_count,
        spec["verification_cases"],
        seed=_stable_seed(seed, "verification_case"),
    )

    cells: list[dict[str, Any]] = []
    split_ordinals: Counter[str] = Counter()
    negative_ordinals: Counter[str] = Counter()
    recovery_index = 0
    verify_index = 0
    for index in range(count):
        target = target_values[index]
        negatives = negative_options[target]
        negative = negatives[negative_ordinals[target] % len(negatives)]
        negative_ordinals[target] += 1
        cell = {name: values[index] for name, values in axes.items()}
        partition = str(cell["evaluation_partition"])
        ordinal = split_ordinals[partition]
        split_ordinals[partition] += 1
        cell.update(
            {
                "matrix_version": MATRIX_VERSION,
                "target_capability": target,
                "hard_negative_capability": negative,
                "registry_profile": f"{partition}_harness_registry_{ordinal % 96:03d}",
                "question_template_group": f"{partition}_harness_templates_{ordinal % 24:02d}",
                "scenario_group_id": f"{partition}_harness_scenario_{ordinal:07d}",
                "tool_identity_group": f"{partition}_opaque_tools_{ordinal % 128:03d}",
            }
        )
        if cell["task_kind"] == "recover":
            cell.update(
                {
                    "recovery_trigger": recovery_triggers[recovery_index],
                    "recovery_round": recovery_rounds[recovery_index],
                    "prior_candidate_count": prior_counts[recovery_index],
                    "candidate_memory": "accumulated_no_repeat",
                    "verification_case": "not_applicable",
                }
            )
            recovery_index += 1
        elif cell["task_kind"] == "verify":
            cell.update(
                {
                    "recovery_trigger": "none",
                    "recovery_round": 0,
                    "prior_candidate_count": 0,
                    "candidate_memory": "not_applicable",
                    "verification_case": verification_cases[verify_index],
                }
            )
            verify_index += 1
        else:
            cell.update(
                {
                    "recovery_trigger": "none",
                    "recovery_round": 0,
                    "prior_candidate_count": 0,
                    "candidate_memory": "not_applicable",
                    "verification_case": "not_applicable",
                }
            )
        issues = validate_cell(cell)
        if issues:
            raise RuntimeError(f"invalid harness-agentic cell {index}: {issues}")
        cell["matrix_cell_id"] = _digest(cell)
        cells.append(cell)

    combinations = Counter(
        (str(cell["task_kind"]), int(cell["candidate_pool_size"])) for cell in cells
    )
    return cells, {
        "matrix_version": MATRIX_VERSION,
        "count": count,
        "seed": seed,
        "partition_counts": dict(
            Counter(str(cell["evaluation_partition"]) for cell in cells)
        ),
        "task_counts": dict(Counter(str(cell["task_kind"]) for cell in cells)),
        "target_counts": dict(
            Counter(str(cell["target_capability"]) for cell in cells)
        ),
        "task_pool_combinations": {
            f"{kind}|{pool}": combinations[(kind, pool)]
            for kind in ("route", "recover", "verify")
            for pool in (10, 25, 50, 100)
        },
        "unique_cell_ids": len({cell["matrix_cell_id"] for cell in cells}),
    }
