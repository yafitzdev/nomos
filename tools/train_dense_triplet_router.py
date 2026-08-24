"""Continue a compact router with mined, within-registry hard negatives."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.dense_router import candidate_document, eligible_tools, query_document


def _states(paths: list[Path], limit_per_input: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("evaluation_partition") != "train" or row.get("task_kind") == "verify":
                    continue
                if not (row.get("label") or {}).get("acceptable_tools"):
                    continue
                rows.append(row)
                counts[str(path)] += 1
                if limit_per_input > 0 and counts[str(path)] >= limit_per_input:
                    break
    if not rows:
        raise ValueError("no answer-present train rows found")
    return rows, counts


def _mine_triplets(model: Any, rows: list[dict[str, Any]], batch_size: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    import numpy as np

    queries = [query_document(row) for row in rows]
    candidate_text: dict[str, str] = {}
    row_candidates = []
    for row in rows:
        tools = eligible_tools(row)
        row_candidates.append(tools)
        for tool in tools:
            candidate_text.setdefault(tool.semantic_fingerprint, candidate_document(tool))
    fingerprints = sorted(candidate_text)
    candidate_embeddings = model.encode(
        [candidate_text[value] for value in fingerprints],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    by_fingerprint = dict(zip(fingerprints, candidate_embeddings))
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt_name="query" if "query" in getattr(model, "prompts", {}) else None,
    )

    triplets: list[dict[str, str]] = []
    capability_counts: Counter[str] = Counter()
    already_correct = 0
    for row, tools, query, anchor in zip(rows, row_candidates, query_embeddings, queries):
        acceptable = {str(value) for value in (row.get("label") or {}).get("acceptable_tools") or []}
        positives = [tool for tool in tools if tool.tool_id in acceptable]
        negatives = [tool for tool in tools if tool.tool_id not in acceptable]
        if not positives or not negatives:
            continue
        positive = max(
            positives,
            key=lambda tool: float(np.dot(query, by_fingerprint[tool.semantic_fingerprint])),
        )
        negative = max(
            negatives,
            key=lambda tool: float(np.dot(query, by_fingerprint[tool.semantic_fingerprint])),
        )
        positive_score = float(np.dot(query, by_fingerprint[positive.semantic_fingerprint]))
        negative_score = float(np.dot(query, by_fingerprint[negative.semantic_fingerprint]))
        already_correct += int(positive_score > negative_score)
        capability_counts.update(positive.capabilities)
        triplets.append(
            {
                "anchor": anchor,
                "positive": candidate_document(positive),
                "negative": candidate_document(negative),
            }
        )
    return triplets, {
        "training_triplets": len(triplets),
        "pretraining_top1_rate": already_correct / len(triplets) if triplets else 0.0,
        "capability_counts": dict(sorted(capability_counts.items())),
        "unique_candidate_semantics": len(candidate_text),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-input", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mining-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )

    args = build_parser().parse_args(argv)
    model = SentenceTransformer(str(args.base_model), local_files_only=True, device=args.device)
    rows, input_counts = _states(args.input, args.limit_per_input)
    triplets, manifest = _mine_triplets(model, rows, args.mining_batch_size)
    loss = losses.TripletLoss(
        model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output.parent / f".{args.output.name}-checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        bf16=args.device == "cuda",
        tf32=args.device == "cuda",
        use_cpu=args.device == "cpu",
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(triplets),
        loss=loss,
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    manifest.update(
        {
            "base_model": str(args.base_model),
            "inputs": [str(path) for path in args.input],
            "input_state_counts": dict(sorted(input_counts.items())),
            "output": str(args.output),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "margin": args.margin,
            "seed": args.seed,
            "device": args.device,
            "loss": "TripletLoss(COSINE)",
            "negative_mining": "highest-scoring legal non-positive within each registry",
        }
    )
    (args.output / "nomos_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
