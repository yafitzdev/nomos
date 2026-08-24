"""Smoke-test an ONNX GenAI chat model from its local snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.local_llm import OnnxGenAIChat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-filename", default="onnx/model_q4f16.onnx")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = OnnxGenAIChat(args.model, model_filename=args.model_filename)
    result = model.complete(
        [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Return only the word OK."},
        ],
        max_new_tokens=16,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
