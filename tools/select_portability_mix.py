"""Build a smaller training mix from the frozen corpus and augmentation rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.generic_pilot_v3 import GENERIC_DATASET_VERSION, validate_generic_state
from fitz_tool.router_v2 import FEATURE_VERSION


UNIQUE_FIELDS = ("decision_state_id", "matrix_cell_id", "type_signature", "instance_signature", "question")
FOCUSED_TARGET_WEIGHTS = {
    "compare_evidence": 0.25,
    "search_content": 0.20,
    "search_metadata": 0.15,
    "exact_pattern_search": 0.15,
    "list_sources": 0.10,
}


def _quotas(count: int, targets: list[str]) -> dict[str, int]:
    if count < 1:
        raise ValueError("count must be positive")
    focused = {target: int(count * weight) for target, weight in FOCUSED_TARGET_WEIGHTS.items()}
    assigned = sum(focused.values())
    other_targets = [target for target in targets if target not in focused]
    remaining = count - assigned
    for index in range(remaining):
        focused[other_targets[index % len(other_targets)]] = (
            focused.get(other_targets[index % len(other_targets)], 0) + 1
        )
    return focused


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--augmentation-input", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        raise SystemExit("count must be positive")
    base_rows = _read_rows(args.base_input)
    augmentation_rows = _read_rows(args.augmentation_input)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmentation_rows:
        groups[str((row.get("sampling_context") or {}).get("target_capability"))].append(row)
    quotas = _quotas(args.count, sorted(groups))
    if sum(quotas.values()) != args.count:
        raise RuntimeError("augmentation quota calculation did not conserve count")
    selected: list[dict[str, Any]] = []
    for target, quota in quotas.items():
        if quota > len(groups.get(target, [])):
            raise SystemExit(f"augmentation has only {len(groups.get(target, []))} rows for {target}; need {quota}")
        selected.extend(groups[target][:quota])

    seen = {field: set() for field in UNIQUE_FIELDS}
    cohort_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in [*base_rows, *selected]:
            report = validate_generic_state(row)
            if not report.valid:
                raise RuntimeError(json.dumps(report.as_dict(), sort_keys=True))
            for field in UNIQUE_FIELDS:
                value = str(row.get(field) or "")
                if not value or value in seen[field]:
                    raise RuntimeError(f"duplicate or missing {field}: {value}")
                seen[field].add(value)
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            cohort_counts[str(row.get("evaluation_cohort"))] += 1
            teacher_counts[str((row.get("provenance") or {}).get("teacher"))] += 1
            target_counts[str((row.get("sampling_context") or {}).get("target_capability"))] += 1

    manifest = {
        "dataset_version": GENERIC_DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "count": len(base_rows) + len(selected),
        "base_count": len(base_rows),
        "augmentation_count": len(selected),
        "augmentation_quotas": dict(sorted(quotas.items())),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "target_capability_counts": dict(sorted(target_counts.items())),
        "teacher": "mixed" if len(teacher_counts) > 1 else next(iter(teacher_counts), "unknown"),
        "teacher_counts": dict(sorted(teacher_counts.items())),
        "inputs": [str(args.base_input), str(args.augmentation_input)],
        "output": str(args.output),
        "unique_fields": list(UNIQUE_FIELDS),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
