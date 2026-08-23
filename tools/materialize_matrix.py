"""Materialize a reproducible, unique matrix slice as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.matrix import coverage, materialize_cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = materialize_cells(args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in cells:
            handle.write(json.dumps({"matrix_version": "matrix.v1", **cell.as_dict()}, sort_keys=True))
            handle.write("\n")
    print(json.dumps({"count": len(cells), "seed": args.seed, "output": str(args.output), "coverage": coverage(cells)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
