"""Evaluate a local Nomos embedding model on the independent ToolRet benchmark."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.hybrid_retrieval import bm25_scores, reciprocal_rank_fusion


TASK_CATEGORY = {
    "apigen": "web",
    "toolace": "web",
    "toolbench": "web",
    "rotbench": "web",
    "ultratool": "web",
    "craft-math-algebra": "code",
    "craft-tabmwp": "code",
    "craft-vqa": "code",
    "gorilla-huggingface": "code",
    "toolink": "code",
    "appbench": "customized",
    "gpt4tools": "customized",
    "metatool": "customized",
    "toolalpaca": "customized",
    "toolbench-sam": "customized",
}


def _label_ids(row: dict[str, Any]) -> set[str]:
    labels = row.get("labels") or []
    if isinstance(labels, str):
        labels = json.loads(labels)
    return {
        str(label["id"])
        for label in labels
        if isinstance(label, dict) and int(label.get("relevance") or 0) > 0
    }


def _sample_rows(dataset: Any, limit: int, seed: int) -> list[dict[str, Any]]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if limit > 0:
        indices = indices[:limit]
    return [dict(dataset[index]) for index in indices]


def _documentation_views(documentation: str) -> tuple[str, ...]:
    """Expose clean semantic views without relying on a benchmark-specific ID."""

    views = [documentation.strip()]
    try:
        parsed = json.loads(documentation)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        name = str(parsed.get("name") or "").strip()
        description = str(parsed.get("description") or "").strip()
        if name and description:
            views.append(f"{name}. {description}")
        profile = parsed.get("tool_profile")
        if isinstance(profile, dict):
            profile_parts = []
            for key in ("function", "tags", "when_to_use", "limitation", "limitations"):
                value = profile.get(key)
                if value:
                    rendered = (
                        ", ".join(map(str, value))
                        if isinstance(value, list)
                        else str(value)
                    )
                    profile_parts.append(f"{key.replace('_', ' ').title()}: {rendered}")
            if profile_parts:
                views.append(". ".join(([name] if name else []) + profile_parts))
        if description:
            views.append(description)
        if name:
            views.append(name)
    return tuple(dict.fromkeys(view for view in views if view))


def _finish(counter: Counter[str]) -> dict[str, float | int]:
    states = int(counter["states"])
    return {
        "states": states,
        "recall_at_1": counter["r1"] / states if states else 0.0,
        "recall_at_3": counter["r3"] / states if states else 0.0,
        "recall_at_5": counter["r5"] / states if states else 0.0,
        "recall_at_10": counter["r10"] / states if states else 0.0,
        "recall_at_20": counter["r20"] / states if states else 0.0,
        "recall_at_50": counter["r50"] / states if states else 0.0,
        "mrr": counter["mrr"] / states if states else 0.0,
    }


def evaluate(
    model: Any,
    *,
    tasks: list[str],
    limit_per_task: int,
    seed: int,
    batch_size: int,
    cache_dir: Path,
    use_instruction: bool,
    candidate_strategy: str,
    expanded_tools_dir: Path | None,
    reranker: Any | None,
    rerank_top_k: int,
    rerank_batch_size: int,
    lexical_weight: float,
) -> dict[str, Any]:
    import numpy as np
    from datasets import load_dataset

    by_category: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for task_index, task in enumerate(tasks):
        if task not in TASK_CATEGORY:
            raise ValueError(f"unknown ToolRet task: {task}")
        query_rows = load_dataset(
            "mangopy/ToolRet-Queries",
            task,
            split="queries",
            cache_dir=str(cache_dir),
        )
        sampled = _sample_rows(query_rows, limit_per_task, seed + task_index)
        by_category.setdefault(TASK_CATEGORY[task], []).extend(
            (task, row) for row in sampled
        )

    overall: Counter[str] = Counter()
    task_counters: dict[str, Counter[str]] = {}
    category_reports = {}
    misses = []
    for category, task_rows in sorted(by_category.items()):
        if expanded_tools_dir is None:
            tools = load_dataset(
                "mangopy/ToolRet-Tools",
                category,
                split="tools",
                cache_dir=str(cache_dir),
            )
        else:
            tools = load_dataset(
                "parquet",
                data_files=str(
                    expanded_tools_dir
                    / category
                    / "tools-00000-of-00001.parquet"
                ),
                split="train",
            )
        tool_ids = [str(row["id"]) for row in tools]
        tool_views = [
            _documentation_views(str(row["documentation"]))
            if candidate_strategy == "multiview"
            else (str(row["documentation"]),)
            for row in tools
        ]
        flat_tool_views = [view for views in tool_views for view in views]
        flat_tool_embeddings = model.encode(
            flat_tool_views,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        query_texts = []
        for _task, row in task_rows:
            query = str(row["query"])
            if use_instruction and row.get("instruction"):
                query = f"Routing objective: {row['instruction']}\nUser request: {query}"
            query_texts.append(query)
        query_embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            prompt_name="query" if "query" in getattr(model, "prompts", {}) else None,
        )
        flat_scores = np.asarray(query_embeddings) @ np.asarray(flat_tool_embeddings).T
        if candidate_strategy == "multiview":
            score_columns = []
            offset = 0
            for views in tool_views:
                score_columns.append(
                    flat_scores[:, offset : offset + len(views)].max(axis=1)
                )
                offset += len(views)
            scores = np.stack(score_columns, axis=1)
        else:
            scores = flat_scores
        if lexical_weight > 0.0:
            lexical_documents = [
                "\n".join(views[1:]) if len(views) > 1 else views[0]
                for views in tool_views
            ]
            scores = np.asarray(
                [
                    reciprocal_rank_fusion(
                        list(map(float, row_scores)),
                        bm25_scores(query, lexical_documents),
                        lexical_weight=lexical_weight,
                    )
                    for query, row_scores in zip(query_texts, scores)
                ]
            )
        rankings = np.argsort(-scores, axis=1)
        if reranker is not None:
            pool_size = min(rerank_top_k, len(tools))
            pairs = []
            for query, ranking in zip(query_texts, rankings):
                for index in ranking[:pool_size]:
                    views = tool_views[int(index)]
                    document = "\n".join(views[1:]) if len(views) > 1 else views[0]
                    pairs.append((query, document))
            rerank_scores = np.asarray(
                reranker.predict(
                    pairs,
                    batch_size=rerank_batch_size,
                    show_progress_bar=True,
                )
            ).reshape(len(task_rows), pool_size)
            reranked = []
            for ranking, row_scores in zip(rankings, rerank_scores):
                order = np.argsort(-row_scores)
                reranked.append(
                    np.concatenate((ranking[:pool_size][order], ranking[pool_size:]))
                )
            rankings = np.stack(reranked)
        category_counter: Counter[str] = Counter()
        for row_index, ((task, row), ranked_indices) in enumerate(
            zip(task_rows, rankings)
        ):
            labels = _label_ids(row)
            first_rank = next(
                (
                    rank
                    for rank, index in enumerate(ranked_indices, start=1)
                    if tool_ids[int(index)] in labels
                ),
                None,
            )
            task_counter = task_counters.setdefault(task, Counter())
            for counter in (overall, category_counter, task_counter):
                counter["states"] += 1
                counter["r1"] += int(first_rank == 1)
                counter["r3"] += int(first_rank is not None and first_rank <= 3)
                counter["r5"] += int(first_rank is not None and first_rank <= 5)
                counter["r10"] += int(first_rank is not None and first_rank <= 10)
                counter["r20"] += int(first_rank is not None and first_rank <= 20)
                counter["r50"] += int(first_rank is not None and first_rank <= 50)
                counter["mrr"] += 1.0 / first_rank if first_rank else 0.0
            if (first_rank is None or first_rank > 3) and len(misses) < 100:
                misses.append(
                    {
                        "task": task,
                        "query_id": row.get("id"),
                        "query": row.get("query"),
                        "target_rank": first_rank,
                        "predicted_tool_ids": [
                            tool_ids[int(index)] for index in ranked_indices[:3]
                        ],
                    }
                )
        category_reports[category] = {
            "candidate_tools": len(tools),
            **_finish(category_counter),
        }
    return {
        "source": (
            "mangopy/ToolRet-Queries + Lux1997/Tool-REX-Tools"
            if expanded_tools_dir is not None
            else "mangopy/ToolRet-Queries + mangopy/ToolRet-Tools"
        ),
        "use_instruction": use_instruction,
        "candidate_strategy": candidate_strategy,
        "reranker": getattr(reranker, "model_name", None),
        "rerank_top_k": rerank_top_k if reranker is not None else None,
        "lexical_weight": lexical_weight,
        "tasks": tasks,
        "metrics": _finish(overall),
        "by_category": category_reports,
        "by_task": {
            task: _finish(counter) for task, counter in sorted(task_counters.items())
        },
        "misses": misses,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--limit-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=0,
        help="Optional runtime truncation for CPU and latency comparisons.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/toolret-cache"))
    parser.add_argument(
        "--expanded-tools-dir",
        type=Path,
        help="Optional local Lux1997/Tool-REX-Tools checkout for expansion ablations.",
    )
    parser.add_argument("--use-instruction", action="store_true")
    parser.add_argument(
        "--candidate-strategy", choices=("single", "multiview"), default="single"
    )
    parser.add_argument("--reranker-model", type=Path)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--lexical-weight", type=float, default=0.0)
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
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(args.model), local_files_only=True, device=args.device
        )
    if args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length
    reranker = None
    if args.reranker_model is not None:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(
            str(args.reranker_model), local_files_only=True, device=args.device
        )
    report = {
        "model": str(args.model),
        "device": args.device,
        "max_seq_length": model.max_seq_length,
        **evaluate(
            model,
            tasks=args.task,
            limit_per_task=args.limit_per_task,
            seed=args.seed,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            use_instruction=args.use_instruction,
            candidate_strategy=args.candidate_strategy,
            expanded_tools_dir=args.expanded_tools_dir,
            reranker=reranker,
            rerank_top_k=args.rerank_top_k,
            rerank_batch_size=args.rerank_batch_size,
            lexical_weight=args.lexical_weight,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
