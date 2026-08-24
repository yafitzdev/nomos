"""Train compact Nomos retrieval on expanded, heterogeneous tool documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_document(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    parts = []
    for key in ("name", "description", "category_name", "method"):
        if value.get(key):
            parts.append(f"{key.replace('_', ' ').title()}: {value[key]}")
    profile = value.get("tool_profile")
    if isinstance(profile, dict):
        for key in ("function", "tags", "when_to_use", "limitation", "limitations"):
            item = profile.get(key)
            if item:
                rendered = ", ".join(map(str, item)) if isinstance(item, list) else str(item)
                parts.append(f"{key.replace('_', ' ').title()}: {rendered}")
    parameters = []
    for key in ("parameters", "required_parameters", "optional_parameters"):
        raw_parameters = value.get(key)
        if isinstance(raw_parameters, dict):
            raw_parameters = [
                {"name": name, **definition}
                if isinstance(definition, dict)
                else {"name": name}
                for name, definition in raw_parameters.items()
            ]
        if isinstance(raw_parameters, list):
            for parameter in raw_parameters[:20]:
                if isinstance(parameter, dict) and parameter.get("name"):
                    description = str(parameter.get("description") or "").strip()
                    parameters.append(
                        f"{parameter['name']}: {description}"
                        if description
                        else str(parameter["name"])
                    )
    if parameters:
        parts.append("Parameters: " + "; ".join(dict.fromkeys(parameters)))
    return ". ".join(parts) if parts else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _excluded_queries(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        rows = payload if isinstance(payload, list) else payload.get("rows") or []
        for row in rows:
            if isinstance(row, str):
                excluded.add(row.strip())
            elif isinstance(row, dict) and row.get("query"):
                excluded.add(str(row["query"]).strip())
    return excluded


def load_training_rows(
    path: Path,
    *,
    limit: int,
    seed: int,
    negative_count: int,
    excluded_queries: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Tool-REX input must be a JSON list")
    indices = list(range(len(payload)))
    random.Random(seed).shuffle(indices)
    excluded_queries = excluded_queries or set()
    rows: list[dict[str, str]] = []
    rejected = 0
    excluded = 0
    for index in indices:
        source = payload[index]
        if not isinstance(source, dict):
            rejected += 1
            continue
        anchor = str(source.get("query") or "").strip()
        positive = _clean_document(source.get("response"))
        negatives = source.get("rejected_response") or []
        if anchor in excluded_queries:
            excluded += 1
            continue
        if not anchor or not positive or not isinstance(negatives, list):
            rejected += 1
            continue
        cleaned_negatives = [
            document
            for value in negatives[:negative_count]
            if (document := _clean_document(value))
        ]
        if len(cleaned_negatives) < negative_count:
            rejected += 1
            continue
        row = {"anchor": anchor, "positive": positive}
        row.update(
            {f"negative_{position + 1}": value for position, value in enumerate(cleaned_negatives)}
        )
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("no valid Tool-REX training rows")
    return rows, {
        "source_rows": len(payload),
        "training_rows": len(rows),
        "rejected_rows": rejected,
        "excluded_rows": excluded,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-query-file", action="append", type=Path, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--negative-count", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--loss", choices=("triplet", "mnrl"), default="triplet")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
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
    excluded = _excluded_queries(args.exclude_query_file)
    rows, counts = load_training_rows(
        args.input,
        limit=args.limit,
        seed=args.seed,
        negative_count=args.negative_count,
        excluded_queries=excluded,
    )
    model = SentenceTransformer(str(args.base_model), local_files_only=True, device=args.device)
    if args.loss == "mnrl":
        loss = losses.MultipleNegativesRankingLoss(model, scale=args.scale)
        loss_name = "MultipleNegativesRankingLoss"
    else:
        if args.negative_count != 1:
            raise SystemExit("triplet loss requires --negative-count 1")
        loss = losses.TripletLoss(
            model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=args.margin,
        )
        loss_name = "TripletLoss(COSINE)"
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
        train_dataset=Dataset.from_list(rows),
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
        "negative_count": args.negative_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "scale": args.scale,
        "seed": args.seed,
        "device": args.device,
        "loss": loss_name,
        "margin": args.margin if args.loss == "triplet" else None,
        "source": "Lux1997/Tool-REX_train_retriever_50k",
        "source_license": "Apache-2.0",
    }
    (args.output / "nomos_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
