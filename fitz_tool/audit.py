"""Deterministic, stratified audit-sample selection."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable, Mapping


DEFAULT_STRATA_AXES = (
    "integration_domain",
    "source_modality",
    "next_tool_target",
    "terminal_condition",
)


def select_stratified_sample(
    records: Iterable[Mapping[str, Any]],
    size: int,
    *,
    seed: int = 20260823,
    axes: tuple[str, ...] = DEFAULT_STRATA_AXES,
) -> list[int]:
    """Select record indices by round-robin strata, reproducibly."""

    rows = list(records)
    if size < 1:
        raise ValueError("sample size must be positive")
    if not rows:
        return []
    size = min(size, len(rows))
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cell = row.get("matrix_cell", {})
        key = tuple(cell.get(axis) for axis in axes)
        groups[key].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    strata = list(groups.values())
    rng.shuffle(strata)

    selected: list[int] = []
    while len(selected) < size and strata:
        next_strata: list[list[int]] = []
        for indices in strata:
            if indices:
                selected.append(indices.pop())
                if len(selected) == size:
                    break
            if indices:
                next_strata.append(indices)
        strata = next_strata
    return selected
