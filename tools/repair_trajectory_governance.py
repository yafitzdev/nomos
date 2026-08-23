"""Repair the pre-v2 governance freshness telemetry in trajectory JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _repair(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(dict(trajectory), ensure_ascii=False))
    previous_decision_tool = ""
    for event in output.get("events", []):
        if not isinstance(event, dict):
            continue
        if event.get("kind") != "decision":
            continue
        governance = event.get("governance")
        if not isinstance(governance, dict):
            governance = {"requirements": []}
            event["governance"] = governance
        governance["assessment_fresh"] = previous_decision_tool == "assess_evidence"
        previous_decision_tool = str(event.get("executed_tool") or "")
    provenance = output.get("provenance")
    if isinstance(provenance, dict):
        provenance["governance_telemetry_version"] = "runner-governance.v2-repair"
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            if line.strip():
                destination.write(
                    json.dumps(_repair(json.loads(line)), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
    print(json.dumps({"input": str(args.input), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
