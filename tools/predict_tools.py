"""Rank legal next tools with a trained Fitz-Tool router artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.contracts import validate_decision_state
from fitz_tool.router import load_router, rank_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    report = validate_decision_state(state)
    if not report.valid:
        raise SystemExit(json.dumps(report.as_dict(), indent=2))
    model, metadata = load_router(str(args.artifact))
    print(json.dumps(rank_tools(model, metadata, state, top_k=args.top_k), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
