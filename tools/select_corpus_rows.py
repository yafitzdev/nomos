"""Select a deterministic partition sample without changing source corpora."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20260824)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        raise SystemExit("count must be positive")
    selected: list[dict[str, Any]] = []
    eligible = 0
    rng = random.Random(args.seed)
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.partition != "all" and row.get("evaluation_partition") != args.partition:
                continue
            eligible += 1
            if len(selected) < args.count:
                selected.append(row)
            else:
                replacement = rng.randrange(eligible)
                if replacement < args.count:
                    selected[replacement] = row
    if len(selected) < args.count:
        raise SystemExit(f"only {len(selected)} eligible rows; cannot select {args.count}")
    selected.sort(key=lambda row: str(row.get("decision_state_id")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "partition": args.partition,
        "eligible": eligible,
        "selected": len(selected),
        "seed": args.seed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
