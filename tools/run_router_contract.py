"""Serve the generic router-request.v2 contract over stdin/stdout JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.coprocessor import coprocessor_response
from fitz_tool.dense_selector import DenseToolRanker
from fitz_tool.generic_contracts import validate_runner_request_v2
from fitz_tool.router_v2 import load_router_v2, rank_tools_v2
from fitz_tool.tool_registry import ToolRegistry


RESPONSE_VERSION = "router-response.v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("dense", "model", "candidate_order"), default="dense"
    )
    parser.add_argument("--artifact", type=Path)
    return parser


def _candidate_order_ranked(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = ToolRegistry.from_dict(request["tool_registry"])
    return [
        {
            "tool_id": tool_id,
            "tool_family": registry.require(tool_id).tool_family,
            "semantic_fingerprint": registry.require(tool_id).semantic_fingerprint,
            "score": 0.0,
        }
        for tool_id in request["legal_candidate_ids"]
    ]


def route_request(
    request: Mapping[str, Any],
    *,
    mode: str,
    model: Any | None,
    metadata: Mapping[str, Any] | None,
    ranker: DenseToolRanker | None = None,
) -> dict[str, Any]:
    report = validate_runner_request_v2(request)
    if not report.valid:
        raise ValueError(json.dumps(report.as_dict(), sort_keys=True))
    operation = str(request.get("operation") or "")
    if operation == "verify_tool_call" or request.get("proposed_tool_call") is not None:
        ranked = []
        router_version = ranker.version if ranker else "deterministic-verifier"
    elif mode == "candidate_order":
        ranked = _candidate_order_ranked(request)
        router_version = "candidate-order-baseline"
    elif mode == "dense":
        if ranker is None:
            raise ValueError("dense mode requires an artifact")
        ranked = ranker.rank(request)
        router_version = ranker.version
    else:
        if model is None or metadata is None:
            raise ValueError("model mode requires an artifact")
        ranked = rank_tools_v2(model, metadata, request, top_k=len(request["legal_candidate_ids"]))
        router_version = str(metadata.get("router_version", "unknown"))
    return coprocessor_response(
        request,
        ranked,
        router_version=router_version,
        calibration=(ranker.calibration if ranker else None)
        or (metadata or {}).get("confidence_calibration"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode in {"dense", "model"} and args.artifact is None:
        raise SystemExit("--artifact is required in dense or model mode")
    model = metadata = None
    ranker = None
    if args.mode == "model":
        model, metadata = load_router_v2(str(args.artifact))
    elif args.mode == "dense":
        ranker = DenseToolRanker.from_path(args.artifact)
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    errors = 0
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        request: Mapping[str, Any] | None = None
        try:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("request must be an object")
            request = value
            response = route_request(
                request,
                mode=args.mode,
                model=model,
                metadata=metadata,
                ranker=ranker,
            )
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            errors += 1
            response = {
                "schema_version": RESPONSE_VERSION,
                "request_id": str(request.get("request_id", "")) if request else "",
                "error": f"line {line_number}: {exc}",
                "runner": {"name": "nomos-router-contract", "version": "router-contract.v2"},
            }
        print(json.dumps(response, ensure_ascii=False, sort_keys=True), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
