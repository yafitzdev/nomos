"""Import JSONL emitted by the external Fitz-Sage V2 router exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.v2_import import import_exported_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-unvalidated-hard-negatives", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    decisions, skips = import_exported_rows(
        rows,
        include_unvalidated_hard_negatives=args.include_unvalidated_hard_negatives,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
    if skips:
        args.output.with_suffix(".errors.jsonl").write_text(
            "".join(json.dumps(skip, sort_keys=True) + "\n" for skip in skips),
            encoding="utf-8",
        )
    print(json.dumps({"input_rows": len(rows), "decision_states": len(decisions), "skips": len(skips), "output": str(args.output)}, indent=2, sort_keys=True))
    return 0 if not skips else 1


if __name__ == "__main__":
    raise SystemExit(main())
