"""Evaluate a fitted dense-router confidence layer on any frozen partition."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.calibration import calibration_metrics
from tools.calibrate_dense_router import _records, _selective_breakdown


def _load_rows(path: Path, partition: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                str(row.get("evaluation_partition")) == partition
                and row.get("task_kind") != "verify"
            ):
                rows.append(row)
    return rows


def _metrics(records: list[dict[str, Any]], calibration: dict[str, Any]) -> dict[str, Any]:
    present = [record for record in records if record["answer_present"]]
    ranks = [int(record["first_rank"]) for record in present if record.get("first_rank")]
    margins = [
        float(record["positive_margin"])
        for record in present
        if record.get("positive_margin") is not None
    ]
    return {
        **calibration_metrics(records, calibration),
        **_selective_breakdown(records, calibration),
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(present) if present else 0.0,
        "recall_at_2": sum(rank <= 2 for rank in ranks) / len(present) if present else 0.0,
        "recall_at_3": sum(rank <= 3 for rank in ranks) / len(present) if present else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks) / len(present) if present else 0.0,
        "mean_positive_margin": sum(margins) / len(margins) if margins else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--query-strategy", choices=("single", "multiview"), default="multiview"
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=("single", "multiview"),
        default="multiview",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from sentence_transformers import SentenceTransformer

    args = build_parser().parse_args(argv)
    calibration_path = args.model / "nomos_calibration.json"
    if not calibration_path.exists():
        raise SystemExit(f"missing fitted calibration: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    rows = _load_rows(args.input, args.partition)
    if not rows:
        raise SystemExit("no non-verification rows matched the requested partition")
    model = SentenceTransformer(
        str(args.model), local_files_only=True, device=args.device
    )
    records = _records(
        model,
        rows,
        args.batch_size,
        query_strategy=args.query_strategy,
        candidate_strategy=args.candidate_strategy,
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cell = row.get("matrix_cell") or {}
        grouped[f"task_kind:{row.get('task_kind') or 'generic'}"].append(index)
        grouped[f"pool_size:{len(row.get('legal_candidate_ids') or [])}"].append(index)
        grouped[f"scenario_family:{cell.get('scenario_family') or 'unspecified'}"].append(index)
    report = {
        "model": str(args.model),
        "input": str(args.input),
        "partition": args.partition,
        "query_strategy": args.query_strategy,
        "candidate_strategy": args.candidate_strategy,
        "calibration_source": str(calibration_path),
        "metrics": _metrics(records, calibration),
        "by_group": {
            key: _metrics([records[index] for index in indices], calibration)
            for key, indices in sorted(grouped.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
