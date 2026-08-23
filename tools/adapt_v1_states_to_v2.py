"""Convert decision-state.v1 JSONL into registry-aware decision-state.v2 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.adapters.fitz_sage_v2 import adapt_v1_decision_state
from fitz_tool.generic_contracts import validate_decision_state_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        args.input.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
            row = adapt_v1_decision_state(source)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        report = validate_decision_state_v2(row)
        if report.valid:
            rows.append(row)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    if errors:
        raise SystemExit(json.dumps({"invalid_rows": len(errors), "examples": errors[:5]}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
