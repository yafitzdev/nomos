"""Materialize and verify the complete fixed-slot Nomos scaling matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.scaling_matrix_v1 import (
    MATRIX_PATH,
    digest,
    load_scaling_matrix,
    materialize_assignments,
    validate_assignments,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_scaling_matrix(args.matrix)
    assignments = materialize_assignments(spec)
    validation = validate_assignments(assignments, spec)
    rendered = "".join(json.dumps(value, sort_keys=True) + "\n" for value in assignments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    manifest = {
        "matrix_version": spec["matrix_version"],
        "dataset_version": spec["dataset_version"],
        "matrix_spec": str(args.matrix),
        "matrix_spec_sha256": digest(spec),
        "assignments_sha256": digest(assignments),
        "output": str(args.output),
        **validation,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
