"""Fine-tune a compact bi-encoder on frozen train-partition query/tool pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.dense_router import candidate_document, query_document
from fitz_tool.embedding_backend import artifact_model_kind, copy_remote_code_files
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
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--model-kind", choices=("auto", "dense", "multivector"), default="auto"
    )
    parser.add_argument("--loss", choices=("mnrl", "cached-mnrl"), default="mnrl")
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-query-length", type=int, default=512)
    parser.add_argument(
        "--query-expansion", choices=("fixed", "min"), default="min"
    )
    return parser


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(
        value
        for value in path.rglob("*")
        if value.is_file() and value.name != "nomos_training_manifest.json"
    ):
        digest.update(str(file_path.relative_to(path)).replace("\\", "/").encode())
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    from datasets import Dataset
    from sentence_transformers import (
        MultiVectorEncoder,
        MultiVectorEncoderTrainer,
        MultiVectorEncoderTrainingArguments,
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.multi_vector_encoder import losses as multivector_losses
    from sentence_transformers.sentence_transformer import losses as dense_losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    args = build_parser().parse_args(argv)
    pairs, manifest = _pairs(args.input, limit_per_input=args.limit_per_input)
    kind = artifact_model_kind(args.base_model) if args.model_kind == "auto" else args.model_kind
    common_training_args = {
        "output_dir": str(args.output.parent / f".{args.output.name}-checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": max(1, round(len(pairs) / args.batch_size * args.epochs * 0.1)),
        "bf16": args.device == "cuda",
        "tf32": args.device == "cuda",
        "use_cpu": args.device == "cpu",
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "save_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
    }
    if kind == "multivector":
        model = MultiVectorEncoder(
            str(args.base_model),
            local_files_only=True,
            trust_remote_code=True,
            device=args.device,
        )
        if getattr(model[0], "query_expansion", None) is not None:
            model[0]._query_expansion["strategy"] = args.query_expansion
        prompts = {
            "anchor": str(getattr(model, "prompts", {}).get("query") or ""),
            "positive": str(getattr(model, "prompts", {}).get("document") or ""),
        }
        if args.loss == "cached-mnrl":
            loss = multivector_losses.CachedMultiVectorMultipleNegativesRankingLoss(
                model,
                mini_batch_size=args.mini_batch_size,
            )
        else:
            loss = multivector_losses.MultiVectorMultipleNegativesRankingLoss(model)
        training_args = MultiVectorEncoderTrainingArguments(
            **common_training_args,
            prompts=prompts,
            router_mapping={"anchor": "query", "positive": "document"},
            max_length={
                "anchor": args.max_query_length,
                "positive": args.max_seq_length,
            },
        )
        trainer = MultiVectorEncoderTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list(pairs),
            loss=loss,
        )
    else:
        model = SentenceTransformer(
            str(args.base_model),
            local_files_only=True,
            trust_remote_code=True,
            device=args.device,
        )
        model.max_seq_length = args.max_seq_length
        prompts = {
            "anchor": str(getattr(model, "prompts", {}).get("query") or ""),
            "positive": str(
                getattr(model, "prompts", {}).get("document")
                or getattr(model, "prompts", {}).get("positive")
                or ""
            ),
        }
        if args.loss == "cached-mnrl":
            loss = dense_losses.CachedMultipleNegativesRankingLoss(
                model,
                mini_batch_size=args.mini_batch_size,
            )
        else:
            loss = dense_losses.MultipleNegativesRankingLoss(model)
        training_args = SentenceTransformerTrainingArguments(
            **common_training_args,
            prompts=prompts,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list(pairs),
            loss=loss,
        )
    started = time.perf_counter()
    train_result = trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    copied_remote_code = copy_remote_code_files(args.base_model, args.output)
    manifest.update(
        {
            "base_model": str(args.base_model),
            "inputs": [str(path) for path in args.input],
            "output": str(args.output),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": args.device,
            "model_kind": kind,
            "loss": args.loss,
            "mini_batch_size": args.mini_batch_size if args.loss == "cached-mnrl" else None,
            "max_seq_length": args.max_seq_length,
            "max_query_length": args.max_query_length if kind == "multivector" else None,
            "query_expansion": args.query_expansion if kind == "multivector" else None,
            "prompts": prompts,
            "batch_sampler": "NO_DUPLICATES",
            "training_duration_seconds": time.perf_counter() - started,
            "training_loss": float(train_result.training_loss),
            "training_script": "tools.train_dense_router.v2",
            "copied_remote_code_files": copied_remote_code,
        }
    )
    manifest["checkpoint_sha256"] = _checkpoint_hash(args.output)
    (args.output / "nomos_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
