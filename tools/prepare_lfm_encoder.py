"""Convert an LFM2.5 masked-LM checkpoint into a SentenceTransformer body."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--max-seq-length", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    args = build_parser().parse_args(argv)
    mlm = AutoModelForMaskedLM.from_pretrained(
        str(args.source),
        local_files_only=True,
        trust_remote_code=True,
    )
    if not hasattr(mlm, "lfm2"):
        raise ValueError("source does not expose the expected LFM2 encoder body")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.source),
        local_files_only=True,
        trust_remote_code=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    mlm.lfm2.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    custom_modeling = args.source / "modeling_lfm2_bidirectional.py"
    if not custom_modeling.exists():
        raise FileNotFoundError(custom_modeling)
    shutil.copy2(custom_modeling, args.output / custom_modeling.name)

    transformer = Transformer(
        str(args.output),
        max_seq_length=args.max_seq_length,
        model_kwargs={"trust_remote_code": True},
    )
    pooling = Pooling(
        transformer.get_embedding_dimension(),
        pooling_mode=args.pooling,
    )
    model = SentenceTransformer(modules=[transformer, pooling, Normalize()])
    model.save_pretrained(str(args.output))
    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "method": "masked_lm_body_to_sentence_transformer.v1",
        "pooling": args.pooling,
        "max_seq_length": args.max_seq_length,
        "embedding_dimension": model.get_embedding_dimension(),
    }
    (args.output / "nomos_preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
