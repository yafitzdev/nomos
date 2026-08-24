"""Fine-tune a compact bi-encoder on frozen train-partition query/tool pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.dense_router import candidate_document, query_document
from fitz_tool.tool_registry import ToolRegistry


def _pairs(paths: list[Path], *, limit_per_input: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    pairs = []
    capability_counts: Counter[str] = Counter()
    registry_fingerprints = set()
    input_counts = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("evaluation_partition") != "train" or row.get("task_kind") == "verify":
                    continue
                acceptable = [str(value) for value in (row.get("label") or {}).get("acceptable_tools") or []]
                if not acceptable:
                    continue
                registry = ToolRegistry.from_dict(row["tool_registry"])
                tool = registry.require(acceptable[0])
                pairs.append(
                    {
                        "anchor": query_document(row),
                        "positive": candidate_document(tool),
                    }
                )
                input_counts[str(path)] += 1
                capability_counts.update(tool.capabilities)
                registry_fingerprints.add(registry.fingerprint)
                if limit_per_input > 0 and input_counts[str(path)] >= limit_per_input:
                    break
    if not pairs:
        raise ValueError("no answer-present train-partition pairs found")
    return pairs, {
        "training_pairs": len(pairs),
        "input_pair_counts": dict(sorted(input_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
        "registry_fingerprints": sorted(registry_fingerprints),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-input", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    args = build_parser().parse_args(argv)
    pairs, manifest = _pairs(args.input, limit_per_input=args.limit_per_input)
    model = SentenceTransformer(str(args.base_model), local_files_only=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output.parent / f".{args.output.name}-checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, round(len(pairs) / args.batch_size * args.epochs * 0.1)),
        bf16=True,
        tf32=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(pairs),
        loss=loss,
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    manifest.update(
        {
            "base_model": str(args.base_model),
            "inputs": [str(path) for path in args.input],
            "output": str(args.output),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "loss": "MultipleNegativesRankingLoss",
            "batch_sampler": "NO_DUPLICATES",
        }
    )
    (args.output / "nomos_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
