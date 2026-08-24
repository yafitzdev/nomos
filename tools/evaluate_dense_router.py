"""Evaluate a local SentenceTransformer as an identity-free tool retriever."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.dense_router import (
    DENSE_TEXT_VERSION,
    candidate_document,
    eligible_tools,
    query_document,
)
from fitz_tool.embedding_backend import (
    encode_documents,
    encode_queries,
    load_embedding_model,
    similarity_matrix,
)


def _sample(path: Path, limit: int, seed: int, partitions: set[str]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    seen = 0
    rng = random.Random(seed)
    effective_limit = limit if limit > 0 else 10**18
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if partitions and str(row.get("evaluation_partition")) not in partitions:
                continue
            if not (row.get("label") or {}).get("acceptable_tools"):
                continue
            seen += 1
            if len(rows) < effective_limit:
                rows.append(row)
                continue
            replacement = rng.randrange(seen)
            if replacement < effective_limit:
                rows[replacement] = row
    return rows, seen


def _finish(counter: Counter[str]) -> dict[str, float | int]:
    states = int(counter["states"])
    return {
        "states": states,
        "recall_at_1": counter["recall_at_1"] / states if states else 0.0,
        "recall_at_2": counter["recall_at_2"] / states if states else 0.0,
        "recall_at_3": counter["recall_at_3"] / states if states else 0.0,
        "mrr": counter["reciprocal_rank"] / states if states else 0.0,
        "mean_positive_margin": counter["positive_margin"] / states if states else 0.0,
    }


def _score_rows(model: Any, rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    queries = [query_document(row) for row in rows]
    candidate_texts: dict[str, str] = {}
    row_tools = []
    for row in rows:
        tools = eligible_tools(row)
        row_tools.append(tools)
        for tool in tools:
            candidate_texts.setdefault(tool.semantic_fingerprint, candidate_document(tool))

    fingerprints = sorted(candidate_texts)
    candidate_embeddings = encode_documents(
        model,
        [candidate_texts[value] for value in fingerprints],
        batch_size=batch_size,
        show_progress_bar=True,
    )
    by_fingerprint = dict(zip(fingerprints, candidate_embeddings))
    query_embeddings = encode_queries(
        model,
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    overall: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = {}
    pool_groups: dict[str, Counter[str]] = {}
    capability_groups: dict[str, Counter[str]] = {}
    scenario_groups: dict[str, Counter[str]] = {}
    disagreements = []
    for row, tools, query_embedding in zip(rows, row_tools, query_embeddings):
        scores = similarity_matrix(
            model,
            [query_embedding],
            [by_fingerprint[tool.semantic_fingerprint] for tool in tools],
        )[0].tolist()
        ranked = [
            tool
            for _score, tool in sorted(
                zip(scores, tools),
                key=lambda item: (-item[0], item[1].semantic_fingerprint, item[1].tool_id),
            )
        ]
        ranked_ids = [tool.tool_id for tool in ranked]
        acceptable = {str(value) for value in (row.get("label") or {}).get("acceptable_tools") or []}
        first_rank = next(
            (index for index, tool_id in enumerate(ranked_ids, start=1) if tool_id in acceptable),
            None,
        )
        kind = str(row.get("task_kind") or "generic")
        group = groups.setdefault(kind, Counter())
        pool_group = pool_groups.setdefault(str(len(tools)), Counter())
        registry = {tool.tool_id: tool for tool in tools}
        expected_capabilities = {
            capability
            for tool_id in acceptable
            if tool_id in registry
            for capability in registry[tool_id].capabilities
        }
        capability_counters = [
            capability_groups.setdefault(capability, Counter())
            for capability in expected_capabilities
        ]
        scenario = str((row.get("matrix_cell") or {}).get("scenario_family") or "unspecified")
        scenario_group = scenario_groups.setdefault(scenario, Counter())
        score_by_id = dict(zip((tool.tool_id for tool in tools), scores))
        positive_scores = [score_by_id[tool_id] for tool_id in acceptable if tool_id in score_by_id]
        negative_scores = [score for tool_id, score in score_by_id.items() if tool_id not in acceptable]
        positive_margin = max(positive_scores) - max(negative_scores) if positive_scores and negative_scores else 0.0
        for counter in (overall, group, pool_group, scenario_group, *capability_counters):
            counter["states"] += 1
            counter["recall_at_1"] += int(first_rank == 1)
            counter["recall_at_2"] += int(first_rank is not None and first_rank <= 2)
            counter["recall_at_3"] += int(first_rank is not None and first_rank <= 3)
            counter["reciprocal_rank"] += 1.0 / first_rank if first_rank else 0.0
            counter["positive_margin"] += positive_margin
        if first_rank != 1 and len(disagreements) < 50:
            disagreements.append(
                {
                    "decision_state_id": row.get("decision_state_id"),
                    "task_kind": kind,
                    "question": row.get("question"),
                    "expected_capabilities": sorted(
                        {
                            capability
                            for tool_id in acceptable
                            if tool_id in registry
                            for capability in registry[tool_id].capabilities
                        }
                    ),
                    "predicted_capabilities": list(ranked[0].capabilities) if ranked else [],
                    "expected_rank": first_rank,
                }
            )
    return {
        "metrics": _finish(overall),
        "by_task_kind": {key: _finish(value) for key, value in sorted(groups.items())},
        "by_pool_size": {key: _finish(value) for key, value in sorted(pool_groups.items())},
        "by_scenario_family": {
            key: _finish(value) for key, value in sorted(scenario_groups.items())
        },
        "by_expected_capability": {
            key: _finish(value) for key, value in sorted(capability_groups.items())
        },
        "unique_candidate_semantics": len(candidate_texts),
        "disagreements": disagreements,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--partition", action="append", default=[])
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.model / "nomos_runtime.json").exists():
        if args.device != "cpu":
            raise SystemExit("the packaged ONNX runtime currently supports CPU only")
        from fitz_tool.onnx_encoder import OnnxSentenceEncoder

        model = OnnxSentenceEncoder(args.model)
    else:
        model = load_embedding_model(args.model, device=args.device)
    reports: dict[str, Mapping[str, Any]] = {}
    for index, path in enumerate(args.input):
        rows, eligible = _sample(path, args.limit, args.seed + index, set(args.partition))
        report = _score_rows(model, rows, args.batch_size)
        reports[str(path)] = {"eligible_rows": eligible, "rows_sampled": len(rows), **report}
    output = {
        "model": str(args.model),
        "text_version": DENSE_TEXT_VERSION,
        "partitions": list(args.partition),
        "device": args.device,
        "inputs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
