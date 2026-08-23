"""Evaluate a trained router against a hand-authored unseen tool registry."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.external_registry_fixtures import (
    EXTERNAL_REGISTRY_STYLES,
    EXTERNAL_STYLE_TOOL_IDS,
    TARGET_CAPABILITIES,
    build_external_registry,
)
from fitz_tool.router_v2 import load_router_v2, rank_tools_v2
from fitz_tool.tool_registry import ToolRegistry


DEFAULT_ARTIFACT = Path("artifacts/nomos_generic_ninfer_full.pt")
DEFAULT_INPUT = Path("data/generated/nomos_generic_ninfer_50000.jsonl")


def _read_external_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evaluation_cohort") != "heldout_questions":
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("input does not contain heldout_questions rows")
    return rows


def _external_state(
    row: dict[str, Any],
    registry: ToolRegistry,
    *,
    target_ids: tuple[str, ...],
    seed: int,
    index: int,
) -> tuple[dict[str, Any], str]:
    target = str((row.get("sampling_context") or {}).get("target_capability"))
    if target not in TARGET_CAPABILITIES:
        raise ValueError(f"unsupported target capability in row: {target}")
    target_id = target_ids[TARGET_CAPABILITIES.index(target)]
    distractors = [tool.tool_id for tool in registry.tools if tool.tool_id != target_id]
    random.Random(seed + index * 7919).shuffle(distractors)
    legal_ids = [target_id, *distractors[:6]]
    random.Random(seed + index * 104729).shuffle(legal_ids)
    output = copy.deepcopy(row)
    output["tool_registry"] = registry.as_dict()
    output["legal_candidate_ids"] = legal_ids
    output["accepted"] = True
    output["evaluation_cohort"] = "external_registry"
    output["evaluation_partition"] = "test"
    output["label"] = {
        "acceptable_tools": [target_id],
        "hard_negative_tools": [tool_id for tool_id in legal_ids if tool_id != target_id],
        "ranked_tools": [target_id, *[tool_id for tool_id in legal_ids if tool_id != target_id]],
        "label_source": "external_registry_test_oracle",
    }
    return output, target_id


def _metrics(rows: list[dict[str, Any]], model: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    recall_1 = recall_3 = 0
    reciprocal_rank = 0.0
    invalid = 0
    for row in rows:
        legal = set(row["legal_candidate_ids"])
        ranked = [item["tool_id"] for item in rank_tools_v2(model, metadata, row, top_k=len(legal))]
        invalid += int(not set(ranked).issubset(legal))
        target = set(row["label"]["acceptable_tools"])
        recall_1 += int(bool(set(ranked[:1]) & target))
        recall_3 += int(bool(set(ranked[:3]) & target))
        first = next((position for position, tool_id in enumerate(ranked, 1) if tool_id in target), None)
        if first is not None:
            reciprocal_rank += 1.0 / first
    count = len(rows)
    return {
        "states": count,
        "recall_at_1": recall_1 / count,
        "recall_at_3": recall_3 / count,
        "mrr": reciprocal_rank / count,
        "invalid_candidate_rate": invalid / count,
    }


def _candidate_order_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recall_1 = recall_3 = 0
    reciprocal_rank = 0.0
    for row in rows:
        ranked = list(row["legal_candidate_ids"])
        target = set(row["label"]["acceptable_tools"])
        recall_1 += int(bool(set(ranked[:1]) & target))
        recall_3 += int(bool(set(ranked[:3]) & target))
        first = next((position for position, tool_id in enumerate(ranked, 1) if tool_id in target), None)
        if first is not None:
            reciprocal_rank += 1.0 / first
    count = len(rows)
    return {
        "states": count,
        "recall_at_1": recall_1 / count,
        "recall_at_3": recall_3 / count,
        "mrr": reciprocal_rank / count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=sorted(EXTERNAL_REGISTRY_STYLES),
        default=list(EXTERNAL_REGISTRY_STYLES),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    model, metadata = load_router_v2(str(args.artifact))
    source_rows = _read_external_rows(args.input, args.limit)
    report: dict[str, Any] = {"artifact": str(args.artifact), "source_cohort": "heldout_questions", "registries": {}}
    for style in args.styles:
        registry = build_external_registry(style)
        target_ids = EXTERNAL_STYLE_TOOL_IDS[style]
        rows: list[dict[str, Any]] = []
        for index, source_row in enumerate(source_rows):
            state, _ = _external_state(
                source_row, registry, target_ids=target_ids, seed=args.seed, index=index
            )
            rows.append(state)
        registry_report: dict[str, Any] = {
            "registry_id": registry.registry_id,
            "registry_fingerprint": registry.fingerprint,
            "registry_tool_ids_are_unseen": True,
            "candidate_order_baseline": _candidate_order_metrics(rows),
            "metrics": _metrics(rows, model, metadata),
            "by_target_capability": {},
        }
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            target = str((row.get("sampling_context") or {}).get("target_capability"))
            groups[target].append(row)
        for target, target_rows in sorted(groups.items()):
            registry_report["by_target_capability"][target] = _metrics(target_rows, model, metadata)
        report["registries"][style] = registry_report
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(
        item["metrics"]["invalid_candidate_rate"] == 0.0
        for item in report["registries"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
