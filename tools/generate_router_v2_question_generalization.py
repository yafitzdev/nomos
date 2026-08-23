"""Build a question-generalization view from the frozen router.v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.pilot_v2 import validate_pilot_state, write_pilot_jsonl
from fitz_tool.question_generalization_v2 import (
    build_question_generalization_training_view,
    generate_question_generalization_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--training-output",
        type=Path,
        default=None,
        help="Also write an interleaved original+generalized training view.",
    )
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=5000)
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict):
            errors.append({"line": line_number, "error": "row must be an object"})
            continue
        report = validate_pilot_state(value)
        if report.valid:
            rows.append(value)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    if errors:
        raise SystemExit(json.dumps({"invalid_rows": len(errors), "examples": errors[:5]}, indent=2))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frozen_rows = _read_rows(args.frozen_input)
    if len(frozen_rows) != args.expected_count:
        raise SystemExit(
            f"frozen benchmark has {len(frozen_rows)} rows; expected {args.expected_count}"
        )
    rows, manifest = generate_question_generalization_rows(frozen_rows)
    if len(rows) != len(frozen_rows):
        raise SystemExit("question-generalization transform changed the row count")
    manifest = {
        **manifest,
        "frozen_input": str(args.frozen_input),
        "frozen_input_sha256": hashlib.sha256(args.frozen_input.read_bytes()).hexdigest(),
        "frozen_benchmark_rows": len(frozen_rows),
    }
    write_pilot_jsonl(rows, args.output)
    training_manifest: dict[str, Any] | None = None
    if args.training_output is not None:
        if args.training_manifest is None:
            raise SystemExit("--training-manifest is required with --training-output")
        training_rows, training_manifest = build_question_generalization_training_view(
            frozen_rows, rows
        )
        write_pilot_jsonl(training_rows, args.training_output)
        training_manifest = {
            **training_manifest,
            "frozen_input": str(args.frozen_input),
            "frozen_input_sha256": manifest["frozen_input_sha256"],
            "derived_input": str(args.output),
        }
        args.training_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.training_manifest.write_text(
            json.dumps(training_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(args.manifest),
                **manifest,
                "training_output": str(args.training_output) if args.training_output else None,
                "training_manifest": str(args.training_manifest) if args.training_manifest else None,
                "training_view": training_manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
