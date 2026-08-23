"""Import external V2 trajectories and export validated decision-state rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.decision_states import validate_and_extract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(args.trajectories.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            trajectory = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(trajectory, dict):
            errors.append({"line": line_number, "error": "trajectory must be an object"})
            continue
        extracted, report = validate_and_extract(trajectory)
        if not report.valid:
            errors.append({"line": line_number, "validation": report.as_dict()})
            continue
        if args.accepted_only:
            extracted = [row for row in extracted if row["accepted"]]
        rows.extend(extracted)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "trajectories": str(args.trajectories),
                "decision_states": len(rows),
                "errors": len(errors),
                "accepted_only": args.accepted_only,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        args.output.with_suffix(".errors.jsonl").write_text(
            "".join(json.dumps(error, sort_keys=True) + "\n" for error in errors),
            encoding="utf-8",
        )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
