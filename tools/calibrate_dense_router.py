"""Calibrate dense-router top-3 recommendation versus abstention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.calibration import (
    calibration_metrics,
    choose_abstention_threshold,
    fit_logistic_calibration,
    predict_confidence,
)
from fitz_tool.coprocessor import score_diagnostics
from fitz_tool.dense_router import (
    candidate_views,
    eligible_tools,
    query_document,
    weighted_query_views,
)


def _load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            partition = str(row.get("evaluation_partition"))
            if partition in output and row.get("task_kind") != "verify":
                output[partition].append(row)
    return output


def _records(
    model: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    *,
    query_strategy: str,
    candidate_strategy: str,
) -> list[dict[str, Any]]:
    import numpy as np

    tools_by_row = [eligible_tools(row) for row in rows]
    documents: dict[str, tuple[str, ...]] = {}
    for tools in tools_by_row:
        for tool in tools:
            views = candidate_views(tool)
            documents.setdefault(
                tool.semantic_fingerprint,
                views if candidate_strategy == "multiview" else (views[0],),
            )
    fingerprints = sorted(documents)
    embeddings = model.encode(
        [text for fingerprint in fingerprints for text in documents[fingerprint]],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    candidate_embeddings = {}
    offset = 0
    for fingerprint in fingerprints:
        count = len(documents[fingerprint])
        candidate_embeddings[fingerprint] = embeddings[offset : offset + count]
        offset += count

    query_views = [
        weighted_query_views(row)
        if query_strategy == "multiview"
        else ((query_document(row), 1.0),)
        for row in rows
    ]
    query_texts = list(
        dict.fromkeys(text for views in query_views for text, _weight in views)
    )
    queries = model.encode(
        query_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt_name="query" if "query" in getattr(model, "prompts", {}) else None,
    )
    query_by_text = dict(zip(query_texts, queries))
    records = []
    for row, tools, views in zip(rows, tools_by_row, query_views):
        scored = sorted(
            (
                (
                    sum(
                        weight
                        * float(
                            np.dot(
                                candidate_embeddings[tool.semantic_fingerprint],
                                query_by_text[text],
                            ).max()
                        )
                        for text, weight in views
                    ),
                    tool.tool_id,
                )
                for tool in tools
            ),
            reverse=True,
        )
        acceptable = {str(value) for value in row["label"]["acceptable_tools"]}
        top3_correct = bool({tool_id for _score, tool_id in scored[:3]} & acceptable)
        first_rank = next(
            (
                rank
                for rank, (_score, tool_id) in enumerate(scored, start=1)
                if tool_id in acceptable
            ),
            None,
        )
        positive_scores = [score for score, tool_id in scored if tool_id in acceptable]
        negative_scores = [score for score, tool_id in scored if tool_id not in acceptable]
        records.append(
            {
                "correct": top3_correct,
                "answer_present": bool(acceptable),
                "first_rank": first_rank,
                "positive_margin": (
                    max(positive_scores) - max(negative_scores)
                    if positive_scores and negative_scores
                    else None
                ),
                "features": score_diagnostics([score for score, _tool_id in scored]),
            }
        )
    return records


def _selective_breakdown(records: list[dict[str, Any]], calibration: dict[str, Any]) -> dict[str, Any]:
    threshold = float(calibration["abstention_threshold"])
    present = [record for record in records if record["answer_present"]]
    absent = [record for record in records if not record["answer_present"]]
    selected = [
        record
        for record in records
        if predict_confidence(calibration, record["features"]) >= threshold
    ]
    return {
        "answer_present": len(present),
        "raw_top3_recall": sum(record["correct"] for record in present) / len(present)
        if present
        else 0.0,
        "false_abstention_rate": sum(
            predict_confidence(calibration, record["features"]) < threshold
            for record in present
        )
        / len(present)
        if present
        else 0.0,
        "no_suitable_tool": len(absent),
        "abstention_recall": sum(
            predict_confidence(calibration, record["features"]) < threshold
            for record in absent
        )
        / len(absent)
        if absent
        else 0.0,
        "selected": len(selected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--query-strategy", choices=("single", "multiview"), default="multiview"
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=("single", "multiview"),
        default="multiview",
    )
    parser.add_argument("--maximum-selective-risk", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from sentence_transformers import SentenceTransformer

    args = build_parser().parse_args(argv)
    model = SentenceTransformer(str(args.model), local_files_only=True)
    rows = _load_rows(args.input)
    records = {
        partition: _records(
            model,
            partition_rows,
            args.batch_size,
            query_strategy=args.query_strategy,
            candidate_strategy=args.candidate_strategy,
        )
        for partition, partition_rows in rows.items()
    }
    calibration = fit_logistic_calibration(records["validation"])
    calibration["target"] = "top3_correct_and_suitable"
    threshold = choose_abstention_threshold(
        records["validation"],
        calibration,
        maximum_selective_risk=args.maximum_selective_risk,
    )
    calibration["abstention_threshold"] = threshold["threshold"]
    report = {
        "model": str(args.model),
        "input": str(args.input),
        "query_strategy": args.query_strategy,
        "candidate_strategy": args.candidate_strategy,
        "maximum_selective_risk": args.maximum_selective_risk,
        "calibration": calibration,
        "threshold_selection": threshold,
        "validation": {
            **calibration_metrics(records["validation"], calibration),
            **_selective_breakdown(records["validation"], calibration),
        },
        "test": {
            **calibration_metrics(records["test"], calibration),
            **_selective_breakdown(records["test"], calibration),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.model / "nomos_calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
