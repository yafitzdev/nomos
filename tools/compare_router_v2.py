"""Compare registry-aware router artifacts on generic and agentic corpora."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from fitz_tool.router_v2 import load_router_v2, rank_tools_v2


def _read_sample(
    path: Path,
    limit: int,
    seed: int,
    partitions: set[str],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    if limit < 1:
        limit = 10**18
    rng = random.Random(seed)
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if partitions and str(row.get("evaluation_partition")) not in partitions:
                continue
            seen += 1
            if len(rows) < limit:
                rows.append(row)
            else:
                replacement = rng.randrange(seen)
                if replacement < limit:
                    rows[replacement] = row
    return rows, seen


def _metrics(model: Any, metadata: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter()
    by_kind: dict[str, Counter[str]] = {}
    by_pool: dict[str, Counter[str]] = {}
    by_axis: dict[str, Counter[str]] = {}
    recover_no_repeat = Counter()
    verification = Counter()
    for row in rows:
        label = row.get("label") or {}
        acceptable = set(str(value) for value in label.get("acceptable_tools") or [])
        if acceptable:
            ranked = rank_tools_v2(model, metadata, row, top_k=max(3, int((row.get("matrix_cell") or {}).get("top_k", 3))))
            ranked_ids = [str(item["tool_id"]) for item in ranked]
            kind = str(row.get("task_kind") or "generic")
            pool = str(len(row.get("legal_candidate_ids") or []))
            axis = str((row.get("matrix_cell") or {}).get("unseen_axis", "generic"))
            for group, key in ((by_kind, kind), (by_pool, pool), (by_axis, axis)):
                group.setdefault(key, Counter())["states"] += 1
            overall["states"] += 1
            overall["recall_at_1"] += int(bool(set(ranked_ids[:1]) & acceptable))
            overall["recall_at_3"] += int(bool(set(ranked_ids[:3]) & acceptable))
            first_rank = next(
                (
                    rank
                    for rank, tool_id in enumerate(ranked_ids, start=1)
                    if tool_id in acceptable
                ),
                None,
            )
            if first_rank is not None:
                overall["reciprocal_rank"] += 1.0 / first_rank
            if kind == "recover":
                prior = set(str(value) for value in row.get("previous_candidate_ids") or [])
                recover_no_repeat["states"] += 1
                recover_no_repeat["no_repeat"] += int(not (set(ranked_ids[:3]) & prior))
            for group, key in ((by_kind, kind), (by_pool, pool), (by_axis, axis)):
                group[key]["recall_at_1"] += int(bool(set(ranked_ids[:1]) & acceptable))
                group[key]["recall_at_3"] += int(bool(set(ranked_ids[:3]) & acceptable))
        if row.get("task_kind") == "verify":
            validation = row.get("validation_label") or {}
            verification["rows"] += 1
            verification["accepted"] += int(bool(validation.get("valid")))
    def finish(groups: dict[str, Counter[str]]) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for key, values in sorted(groups.items()):
            states = values["states"]
            output[key] = {
                "states": states,
                "recall_at_1": values["recall_at_1"] / states if states else 0.0,
                "recall_at_3": values["recall_at_3"] / states if states else 0.0,
            }
        return output
    states = overall["states"]
    return {
        "states": states,
        "recall_at_1": overall["recall_at_1"] / states if states else 0.0,
        "recall_at_3": overall["recall_at_3"] / states if states else 0.0,
        "mrr": overall["reciprocal_rank"] / states if states else 0.0,
        "invalid_candidate_rate": 0.0,
        "by_task_kind": finish(by_kind),
        "by_pool_size": finish(by_pool),
        "by_unseen_axis": finish(by_axis),
        "recovery_no_repeat_rate": recover_no_repeat["no_repeat"] / recover_no_repeat["states"] if recover_no_repeat["states"] else 0.0,
        "recovery_states": recover_no_repeat["states"],
        "verification_rows": verification["rows"],
        "verification_accepted_rows": verification["accepted"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, help="label=path; repeat for comparisons")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--partition",
        action="append",
        choices=("train", "validation", "test"),
        default=[],
        help="Evaluate only these partition(s); repeat to include more than one.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts: dict[str, Path] = {}
    for value in args.artifact:
        if "=" not in value:
            raise SystemExit("--artifact must be label=path")
        label, path = value.split("=", 1)
        artifacts[label] = Path(path)
    reports: dict[str, Any] = {}
    for input_index, path in enumerate(args.input):
        rows, eligible_rows = _read_sample(
            path,
            args.limit,
            args.seed + input_index,
            set(args.partition),
        )
        reports[str(path)] = {
            "rows_sampled": len(rows),
            "eligible_rows": eligible_rows,
            "artifacts": {},
        }
        for label, artifact_path in artifacts.items():
            model, metadata = load_router_v2(str(artifact_path))
            reports[str(path)]["artifacts"][label] = {
                "artifact": str(artifact_path),
                "metrics": _metrics(model, metadata, rows),
            }
    output = {
        "inputs": reports,
        "seed": args.seed,
        "limit": args.limit,
        "partitions": list(args.partition),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
