"""Materialize unique legal cells from the capability-oriented matrix.v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.matrix_v2 import matrix_v2_coverage, materialize_matrix_v2_cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = materialize_matrix_v2_cells(args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                {
                    "matrix_version": "matrix.v2",
                    "matrix_cell_id": cell.cell_id,
                    **cell.as_dict(),
                },
                sort_keys=True,
            )
            + "\n"
            for cell in cells
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "count": len(cells),
                "coverage": matrix_v2_coverage(cells),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
