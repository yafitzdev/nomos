"""Audit Nomos target ranks on leakage-safe executed-session states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.dense_router import candidate_views, query_document, weighted_query_views
from fitz_tool.embedding_backend import (
    encode_documents,
    encode_queries,
    load_embedding_model,
    similarity_matrix,
)
from fitz_tool.external_registry_fixtures import EXTERNAL_REGISTRY_STYLES, build_external_registry
from fitz_tool.final_holdout_fixtures import (
    FINAL_HOLDOUT_VERSION,
    FINAL_REGISTRY_STYLES,
    canonical_capability as final_canonical_capability,
    build_final_registry,
)
from fitz_tool.promotion_holdout_fixtures import (
    PROMOTION_HOLDOUT_VERSION,
    PROMOTION_REGISTRY_STYLES,
    canonical_capability as promotion_canonical_capability,
    build_promotion_registry,
)
from tools.evaluate_real_agent_sessions import (
    FINAL_WORKFLOWS,
    PROMOTION_WORKFLOWS,
    WORKFLOWS,
    _request,
)


def _metrics(ranks: list[int], margins: list[float] | None = None) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "states": len(ranks),
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_2": sum(rank <= 2 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
    }
    if margins is not None:
        metrics["mean_positive_margin"] = sum(margins) / len(margins)
    return metrics


def audit(
    model: Any,
    *,
    batch_size: int,
    strategy: str,
    candidate_strategy: str,
    suite: str = "development",
) -> dict[str, Any]:
    if suite == "final":
        styles = FINAL_REGISTRY_STYLES
        workflows = FINAL_WORKFLOWS
        registry_builder = build_final_registry
        canonical = final_canonical_capability
        suite_version = FINAL_HOLDOUT_VERSION
    elif suite == "promotion":
        styles = PROMOTION_REGISTRY_STYLES
        workflows = PROMOTION_WORKFLOWS
        registry_builder = build_promotion_registry
        canonical = promotion_canonical_capability
        suite_version = PROMOTION_HOLDOUT_VERSION
    else:
        styles = EXTERNAL_REGISTRY_STYLES
        workflows = WORKFLOWS
        registry_builder = build_external_registry
        canonical = None
        suite_version = "development.v1"
    cases = []
    candidate_texts: dict[str, tuple[str, ...]] = {}
    view_texts: dict[str, None] = {}
    for style in styles:
        registry = registry_builder(style)
        legal_ids = [tool.tool_id for tool in registry.tools]
        for tool in registry.tools:
            candidate_texts.setdefault(
                tool.semantic_fingerprint,
                candidate_views(tool)
                if candidate_strategy == "multiview"
                else (candidate_views(tool)[0],),
            )
        for workflow in workflows:
            for position, stage in enumerate(workflow["stages"]):
                request = _request(
                    workflow,
                    registry,
                    legal_ids,
                    stage=stage,
                    previous_ids=[],
                    completed=list(workflow["stages"][:position]),
                )
                views = (
                    weighted_query_views(request)
                    if strategy == "multiview"
                    else ((query_document(request), 1.0),)
                )
                for text, _weight in views:
                    view_texts[text] = None
                cases.append((style, workflow["name"], stage, registry, views))

    candidate_keys = sorted(candidate_texts)
    flat_candidate_texts = [
        text for key in candidate_keys for text in candidate_texts[key]
    ]
    flat_candidate_embeddings = encode_documents(
        model,
        flat_candidate_texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    candidate_by_key = {}
    offset = 0
    for key in candidate_keys:
        count = len(candidate_texts[key])
        candidate_by_key[key] = flat_candidate_embeddings[offset : offset + count]
        offset += count
    view_keys = list(view_texts)
    view_embeddings = encode_queries(
        model,
        view_keys,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    view_by_text = dict(zip(view_keys, view_embeddings))

    records = []
    for style, workflow, stage, registry, views in cases:
        scored = []
        for tool in registry.tools:
            score = sum(
                weight
                * float(
                    similarity_matrix(
                        model,
                        [view_by_text[text]],
                        candidate_by_key[tool.semantic_fingerprint],
                    ).max()
                )
                for text, weight in views
            )
            scored.append((score, tool))
        ranked = [
            tool
            for _score, tool in sorted(
                scored,
                key=lambda item: (-item[0], item[1].semantic_fingerprint),
            )
        ]
        target_scores = [
            score
            for score, tool in scored
            if (canonical(tool.tool_id) == stage if canonical else stage in tool.capabilities)
        ]
        negative_scores = [
            score
            for score, tool in scored
            if not (canonical(tool.tool_id) == stage if canonical else stage in tool.capabilities)
        ]
        target_rank = next(
            index
            for index, tool in enumerate(ranked, start=1)
            if (canonical(tool.tool_id) == stage if canonical else stage in tool.capabilities)
        )
        predicted_capability = (
            canonical(ranked[0].tool_id) or "irrelevant"
            if canonical
            else ranked[0].capabilities[0]
        )
        records.append(
            {
                "registry_style": style,
                "workflow": workflow,
                "target_capability": stage,
                "target_rank": target_rank,
                "positive_margin": max(target_scores) - max(negative_scores),
                "predicted_capability": predicted_capability,
            }
        )

    groups: dict[str, Counter[int]] = {}
    for record in records:
        groups.setdefault(record["target_capability"], Counter())[record["target_rank"]] += 1
    return {
        "suite": suite,
        "suite_version": suite_version,
        "strategy": strategy,
        "candidate_strategy": candidate_strategy,
        "metrics": _metrics(
            [record["target_rank"] for record in records],
            [record["positive_margin"] for record in records],
        ),
        "by_capability": {
            capability: _metrics(
                [rank for rank, count in counts.items() for _ in range(count)]
            )
            for capability, counts in sorted(groups.items())
        },
        "misses": [record for record in records if record["target_rank"] > 3],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("development", "final", "promotion"),
        default="development",
    )
    parser.add_argument("--strategy", choices=("single", "multiview"), default="multiview")
    parser.add_argument(
        "--candidate-strategy", choices=("single", "multiview"), default="single"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
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
    report = {
        "model": str(args.model),
        "device": args.device,
        **audit(
            model,
            batch_size=args.batch_size,
            strategy=args.strategy,
            candidate_strategy=args.candidate_strategy,
            suite=args.suite,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    print(f"top3_misses={len(report['misses'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
