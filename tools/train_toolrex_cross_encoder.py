"""Fine-tune a compact pairwise reranker on tool-specific relevance pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from tools.train_toolrex_dense_router import _clean_document, _excluded_queries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_pair(row: dict[str, Any]) -> tuple[str, str, float] | None:
    value = str(row.get("input") or "")
    query_prefix = "Query: "
    separator = "\nTool doc: "
    if not value.startswith(query_prefix) or separator not in value:
        return None
    query, document = value[len(query_prefix) :].split(separator, 1)
    label = 1.0 if str(row.get("output") or "").strip().endswith("true") else 0.0
    return query.strip(), _clean_document(document.strip()), label


def load_pairs(
    path: Path,
    *,
    limit: int,
    seed: int,
    excluded_queries: set[str] | None = None,
) -> tuple[list[dict[str, str | float]], dict[str, int]]:
    excluded_queries = excluded_queries or set()
    rng = random.Random(seed)
    reservoir: list[dict[str, str | float]] = []
    source_rows = rejected = excluded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_rows += 1
            try:
                parsed = _parse_pair(json.loads(line))
            except json.JSONDecodeError:
                parsed = None
            if parsed is None:
                rejected += 1
                continue
            query, document, label = parsed
            if query in excluded_queries:
                excluded += 1
                continue
            item: dict[str, str | float] = {
                "query": query,
                "document": document,
                "label": label,
            }
            if limit <= 0 or len(reservoir) < limit:
                reservoir.append(item)
            else:
                replacement = rng.randrange(source_rows - rejected - excluded)
                if replacement < limit:
                    reservoir[replacement] = item
    rng.shuffle(reservoir)
    if not reservoir:
        raise ValueError("no valid Tool-REX reranker pairs")
    return reservoir, {
        "source_rows": source_rows,
        "training_pairs": len(reservoir),
        "positive_pairs": sum(float(row["label"]) == 1.0 for row in reservoir),
        "negative_pairs": sum(float(row["label"]) == 0.0 for row in reservoir),
        "rejected_rows": rejected,
        "excluded_rows": excluded,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-query-file", action="append", type=Path, default=[])
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--positive-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder import losses

    args = build_parser().parse_args(argv)
    excluded = _excluded_queries(args.exclude_query_file)
    pairs, counts = load_pairs(
        args.input,
        limit=args.limit,
        seed=args.seed,
        excluded_queries=excluded,
    )
    model = CrossEncoder(
        str(args.base_model),
        num_labels=1,
        max_length=args.max_length,
        local_files_only=True,
        device=args.device,
    )
    loss = losses.BinaryCrossEntropyLoss(
        model,
        pos_weight=torch.tensor(args.positive_weight, device=model.device),
    )
    training_args = CrossEncoderTrainingArguments(
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
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(pairs),
        loss=loss,
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    manifest = {
        **counts,
        "base_model": str(args.base_model),
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "excluded_query_files": [str(path) for path in args.exclude_query_file],
        "excluded_query_count": len(excluded),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "positive_weight": args.positive_weight,
        "seed": args.seed,
        "device": args.device,
        "loss": "BinaryCrossEntropyLoss",
        "source": "Lux1997/Tool-REX_train_reranker_200k",
        "source_license": "Apache-2.0",
    }
    (args.output / "nomos_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
