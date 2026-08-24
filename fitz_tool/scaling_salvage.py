"""Deterministic selection rules for a balanced scaling-data salvage set."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .scaling_matrix_v1 import SCENARIOS


def canonical_scaling_target(row: dict[str, Any]) -> str | None:
    """Recover the fixed target from a row, including replacement assignments."""

    scenario = str((row.get("matrix_cell") or {}).get("scenario_family") or "")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scaling scenario family: {scenario or '<missing>'}")
    target = SCENARIOS[scenario].get("target")
    return str(target) if target else None


def select_balanced_hard_rows(
    scores: Iterable[dict[str, Any]],
    *,
    max_per_capability: int,
) -> tuple[set[str], dict[str, Any]]:
    """Keep the hardest rows per canonical target, preferring actual mistakes."""

    if max_per_capability <= 0:
        raise ValueError("max_per_capability must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_counts: Counter[str] = Counter()
    mistake_counts: Counter[str] = Counter()
    for score in scores:
        capability = score.get("target_capability")
        if not capability:
            continue
        capability = str(capability)
        grouped[capability].append(score)
        total_counts[capability] += 1
        if int(score["target_rank"]) > 1:
            mistake_counts[capability] += 1

    selected: set[str] = set()
    selected_counts: Counter[str] = Counter()
    selected_mistakes: Counter[str] = Counter()
    for capability, rows in grouped.items():
        # Wrongly ranked rows are always considered before merely low-margin rows.
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row["target_rank"]) == 1,
                float(row["margin"]),
                -int(row["target_rank"]),
                str(row["decision_state_id"]),
            ),
        )
        for row in ordered[:max_per_capability]:
            decision_id = str(row["decision_state_id"])
            selected.add(decision_id)
            selected_counts[capability] += 1
            if int(row["target_rank"]) > 1:
                selected_mistakes[capability] += 1

    return selected, {
        "selection_rule": "mistakes first, then ascending positive-minus-best-negative margin",
        "max_per_capability": max_per_capability,
        "available_counts": dict(sorted(total_counts.items())),
        "available_mistake_counts": dict(sorted(mistake_counts.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "selected_mistake_counts": dict(sorted(selected_mistakes.items())),
        "selected_rows": len(selected),
    }
