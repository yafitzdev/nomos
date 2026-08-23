"""Rank an external agent's legal candidates with a router.v2 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.generic_contracts import validate_runner_request_v2
from fitz_tool.router_v2 import load_router_v2, rank_tools_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    report = validate_runner_request_v2(request)
    if not report.valid:
        raise SystemExit(json.dumps(report.as_dict(), indent=2))
    model, metadata = load_router_v2(str(args.artifact))
    print(json.dumps(rank_tools_v2(model, metadata, request, top_k=args.top_k), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
