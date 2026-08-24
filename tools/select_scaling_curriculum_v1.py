"""Create nested balanced subsets and broad-to-hard stages from the 25k cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


HARD_FAMILIES = {
    "assess_vs_finalize",
    "compare_vs_assess_evidence",
    "conflicting_irrelevant_history",
    "illegal_attractive_candidate",
    "inspect_vs_compare_evidence",
    "missing_prerequisite",
    "partial_context_vs_new_search",
    "recovery_after_rejection",
    "requirements_vs_assess",
    "requirements_vs_finalize",
    "side_effect_policy",
    "stale_terminal_history",
}


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line), line


def _balanced_order(path: Path, seed: int) -> list[str]:
    by_family: dict[str, list[tuple[str, str]]] = {}
    for row, _line in _iter_rows(path):
        identity = str(row["decision_state_id"])
        family = str(row["matrix_cell"]["scenario_family"])
        score = hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()
        by_family.setdefault(family, []).append((score, identity))
    for values in by_family.values():
        values.sort()
    order: list[str] = []
    offset = 0
    while len(order) < sum(map(len, by_family.values())):
        for family in sorted(by_family):
            if offset < len(by_family[family]):
                order.append(by_family[family][offset][1])
        offset += 1
    return order


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-5k", type=Path, required=True)
    parser.add_argument("--output-10k", type=Path, required=True)
    parser.add_argument("--output-broad", type=Path, required=True)
    parser.add_argument("--output-hard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    order = _balanced_order(args.input, args.seed)
    if len(order) != 25_000 or len(set(order)) != 25_000:
        raise ValueError("expected exactly 25,000 unique cohort rows")
    subset_5k = set(order[:5_000])
    subset_10k = set(order[:10_000])
    paths = (args.output_5k, args.output_10k, args.output_broad, args.output_hard)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = [path.open("w", encoding="utf-8") for path in paths]
    counts = {str(path): Counter() for path in paths}
    answer_present = Counter()
    try:
        for row, line in _iter_rows(args.input):
            identity = str(row["decision_state_id"])
            family = str(row["matrix_cell"]["scenario_family"])
            destinations = []
            if identity in subset_5k:
                destinations.append(0)
            if identity in subset_10k:
                destinations.append(1)
            destinations.append(3 if family in HARD_FAMILIES else 2)
            for destination in destinations:
                handles[destination].write(line if line.endswith("\n") else line + "\n")
                counts[str(paths[destination])][family] += 1
                answer_present[str(paths[destination])] += int(
                    bool((row.get("label") or {}).get("acceptable_tools"))
                )
    finally:
        for handle in handles:
            handle.close()
    output_counts = {path: sum(values.values()) for path, values in counts.items()}
    if output_counts[str(args.output_5k)] != 5_000 or output_counts[str(args.output_10k)] != 10_000:
        raise RuntimeError("nested subset counts are incorrect")
    if output_counts[str(args.output_broad)] + output_counts[str(args.output_hard)] != 25_000:
        raise RuntimeError("staged partition does not cover the full cohort")
    manifest: dict[str, Any] = {
        "input": str(args.input),
        "seed": args.seed,
        "nested": True,
        "hard_families": sorted(HARD_FAMILIES),
        "outputs": {
            path: {
                "rows": output_counts[path],
                "answer_present_rows": answer_present[path],
                "scenario_family_counts": dict(sorted(values.items())),
            }
            for path, values in counts.items()
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
