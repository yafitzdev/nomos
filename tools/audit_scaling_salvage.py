"""Score scaling rows against a checkpoint and write a balanced hard subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from fitz_tool.dense_router import candidate_document, eligible_tools, query_document
from fitz_tool.scaling_salvage import canonical_scaling_target, select_balanced_hard_rows


def _chunks(values: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    iterator = iter(values)
    while chunk := list(islice(iterator, size)):
        yield chunk


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _score_chunk(model: Any, rows: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    import numpy as np

    row_tools = [eligible_tools(row) for row in rows]
    documents: dict[str, str] = {}
    for tools in row_tools:
        for tool in tools:
            documents.setdefault(tool.semantic_fingerprint, candidate_document(tool))
    fingerprints = sorted(documents)
    embeddings = model.encode(
        [documents[value] for value in fingerprints],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    by_fingerprint = dict(zip(fingerprints, embeddings))
    query_embeddings = model.encode(
        [query_document(row) for row in rows],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        prompt_name="query" if "query" in getattr(model, "prompts", {}) else None,
    )
    results = []
    for row, tools, query in zip(rows, row_tools, query_embeddings):
        acceptable = {str(value) for value in (row.get("label") or {}).get("acceptable_tools") or []}
        if not acceptable:
            continue
        ranked = sorted(
            (
                (float(np.dot(query, by_fingerprint[tool.semantic_fingerprint])), tool)
                for tool in tools
            ),
            key=lambda value: value[0],
            reverse=True,
        )
        positives = [(score, tool) for score, tool in ranked if tool.tool_id in acceptable]
        negatives = [(score, tool) for score, tool in ranked if tool.tool_id not in acceptable]
        if not positives or not negatives:
            continue
        positive_score, positive = positives[0]
        negative_score, negative = negatives[0]
        target_rank = next(index for index, (_score, tool) in enumerate(ranked, 1) if tool.tool_id in acceptable)
        decision_id = str(row["decision_state_id"])
        results.append(
            {
                "decision_state_id": decision_id,
                "target_capability": canonical_scaling_target(row),
                "target_rank": target_rank,
                "positive_score": positive_score,
                "best_negative_score": negative_score,
                "margin": positive_score - negative_score,
                "positive_fingerprint": positive.semantic_fingerprint,
                "best_negative_fingerprint": negative.semantic_fingerprint,
                "candidate_count": len(ranked),
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--salvage-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--max-per-capability", type=int, default=300)
    parser.add_argument("--expected-input-rows", type=int, default=25_000)
    parser.add_argument("--row-chunk-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from sentence_transformers import SentenceTransformer

    args = build_parser().parse_args(argv)
    model = SentenceTransformer(str(args.base_model), local_files_only=True, device=args.device)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    input_rows = 0
    with args.scores_output.open("w", encoding="utf-8") as output:
        for chunk in _chunks(_iter_rows(args.input), args.row_chunk_size):
            input_rows += len(chunk)
            chunk_scores = _score_chunk(model, chunk, args.batch_size)
            scores.extend(chunk_scores)
            for score in chunk_scores:
                output.write(json.dumps(score, sort_keys=True) + "\n")
            print(f"scored {input_rows} rows", flush=True)
    if args.expected_input_rows > 0 and input_rows != args.expected_input_rows:
        raise ValueError(
            f"expected {args.expected_input_rows} input rows, found {input_rows}"
        )

    selected, selection = select_balanced_hard_rows(scores, max_per_capability=args.max_per_capability)
    args.salvage_output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.salvage_output.open("w", encoding="utf-8") as output:
        for row in _iter_rows(args.input):
            if str(row.get("decision_state_id")) in selected:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
    rank_counts = Counter(min(int(value["target_rank"]), 4) for value in scores)
    manifest = {
        "base_model": str(args.base_model),
        "input": str(args.input),
        "input_rows": input_rows,
        "scored_answer_present_rows": len(scores),
        "rank_counts_capped_at_4": dict(sorted(rank_counts.items())),
        "pretraining_top1_rate": rank_counts[1] / len(scores),
        "scores_output": str(args.scores_output),
        "salvage_output": str(args.salvage_output),
        **selection,
    }
    if written != len(selected):
        raise RuntimeError(f"selected {len(selected)} IDs but wrote {written} rows")
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
