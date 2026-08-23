"""Generate a deterministic matrix-oracle fixture for contract testing.

This command intentionally does not generate the approved teacher corpus. Use
``tools.generate_generic_ninfer_v3`` for training data with real NInfer/Qwen
question surfaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.generic_pilot_v3 import (
    GENERIC_COHORT_COUNTS,
    GENERIC_PILOT_SEED,
    generate_generic_states,
    write_generic_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=sum(GENERIC_COHORT_COUNTS.values()))
    parser.add_argument("--seed", type=int, default=GENERIC_PILOT_SEED)
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="Acknowledge that this output is a deterministic test fixture, not training data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.fixture_only:
        raise SystemExit(
            "This command creates deterministic fixtures only; pass --fixture-only "
            "or use tools.generate_generic_ninfer_v3 for training data."
        )
    rows, manifest = generate_generic_states(count=args.expected_count, seed=args.seed)
    write_generic_jsonl(rows, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(args.manifest), **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
