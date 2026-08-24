"""Replay OpenAI tool-decision requests with all tools versus Nomos top-k."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fitz_tool.adapters.openai_tools import (
    build_runner_request_from_openai,
    retain_openai_tools,
    selected_tool_from_response,
)
from fitz_tool.dense_selector import DenseToolRanker


TokenCounter = Callable[[Any], int]


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _load_token_counter(model_id: str | None) -> TokenCounter | None:
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the train extra to use --tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count(value: Any) -> int:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping) or not isinstance(value.get("request"), Mapping):
                raise ValueError(f"line {line_number} must contain an object-valued request")
            acceptable = value.get("acceptable_tools")
            expected = value.get("expected_tool")
            if acceptable is None and isinstance(expected, str):
                acceptable = [expected]
            if acceptable is not None and (
                not isinstance(acceptable, list)
                or any(not isinstance(item, str) for item in acceptable)
            ):
                raise ValueError(f"line {line_number} acceptable_tools must be a string list")
            cases.append(
                {
                    "case_id": str(value.get("case_id") or f"case-{line_number:05d}"),
                    "request": dict(value["request"]),
                    "acceptable_tools": list(acceptable or []),
                }
            )
    if not cases:
        raise ValueError("input contains no cases")
    return cases


def _clean_for_provider(body: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key not in {"x-nomos", "x_nomos"}}


def _call_provider(
    endpoint: str,
    body: Mapping[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(_clean_for_provider(body), ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"provider returned HTTP {exc.code}: {detail[:500]}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("provider response must be an object")
    usage = payload.get("usage")
    usage = dict(usage) if isinstance(usage, Mapping) else {}
    return {
        "latency_seconds": time.perf_counter() - started,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "selected_tool": selected_tool_from_response(payload),
        "response_id": payload.get("id"),
    }


def _condition_metrics(body: Mapping[str, Any], counter: TokenCounter | None) -> dict[str, Any]:
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    return {
        "visible_tools": len(tools),
        "request_bytes": _json_bytes(body),
        "tool_schema_bytes": _json_bytes(tools),
        "estimated_request_tokens": counter(body) if counter else None,
        "estimated_tool_schema_tokens": counter(tools) if counter else None,
    }


def _reduction(full: float, nomos: float) -> float | None:
    return 1.0 - nomos / full if full else None


def _sum(rows: Sequence[Mapping[str, Any]], condition: str, field: str) -> int:
    return sum(int(row["conditions"][condition].get(field) or 0) for row in rows)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> dict[str, Any]:
    full_bytes = _sum(rows, "full", "request_bytes")
    nomos_bytes = _sum(rows, "nomos", "request_bytes")
    full_schema_bytes = _sum(rows, "full", "tool_schema_bytes")
    nomos_schema_bytes = _sum(rows, "nomos", "tool_schema_bytes")
    full_estimated = _sum(rows, "full", "estimated_request_tokens")
    nomos_estimated = _sum(rows, "nomos", "estimated_request_tokens")
    full_estimated_schema = _sum(rows, "full", "estimated_tool_schema_tokens")
    nomos_estimated_schema = _sum(rows, "nomos", "estimated_tool_schema_tokens")
    full_actual = _sum(rows, "full", "prompt_tokens")
    nomos_actual = _sum(rows, "nomos", "prompt_tokens")
    full_completion = _sum(rows, "full", "completion_tokens")
    nomos_completion = _sum(rows, "nomos", "completion_tokens")
    labeled = [row for row in rows if row.get("acceptable_tools")]
    live_rows = [
        row
        for row in rows
        if "prompt_tokens" in row["conditions"]["full"]
        and "prompt_tokens" in row["conditions"]["nomos"]
    ]
    ranking_latencies = [float(row.get("ranking_seconds") or 0.0) for row in rows]
    summary: dict[str, Any] = {
        "cases": len(rows),
        "top_k": top_k,
        "mean_full_tools": _sum(rows, "full", "visible_tools") / len(rows),
        "mean_nomos_tools": _sum(rows, "nomos", "visible_tools") / len(rows),
        "request_byte_reduction": _reduction(full_bytes, nomos_bytes),
        "tool_schema_byte_reduction": _reduction(full_schema_bytes, nomos_schema_bytes),
        "nomos_top_k_recall": (
            sum(
                bool(set(row["ranked_tool_ids"][:top_k]) & set(row["acceptable_tools"]))
                for row in labeled
            )
            / len(labeled)
            if labeled
            else None
        ),
        "labeled_cases": len(labeled),
        "live_paired_cases": len(live_rows),
        "nomos_ranking_p50_seconds": statistics.median(ranking_latencies),
        "nomos_ranking_p95_seconds": _percentile(ranking_latencies, 0.95),
    }
    if full_estimated:
        summary.update(
            {
                "estimated_full_request_tokens": full_estimated,
                "estimated_nomos_request_tokens": nomos_estimated,
                "estimated_request_token_reduction": _reduction(
                    full_estimated, nomos_estimated
                ),
                "estimated_full_tool_schema_tokens": full_estimated_schema,
                "estimated_nomos_tool_schema_tokens": nomos_estimated_schema,
                "estimated_tool_schema_token_reduction": _reduction(
                    full_estimated_schema, nomos_estimated_schema
                ),
            }
        )
    if full_actual:
        full_cost = (
            full_actual * input_cost_per_million
            + full_completion * output_cost_per_million
        ) / 1_000_000
        nomos_cost = (
            nomos_actual * input_cost_per_million
            + nomos_completion * output_cost_per_million
        ) / 1_000_000
        summary.update(
            {
                "provider_full_prompt_tokens": full_actual,
                "provider_nomos_prompt_tokens": nomos_actual,
                "provider_prompt_token_reduction": _reduction(full_actual, nomos_actual),
                "provider_full_completion_tokens": full_completion,
                "provider_nomos_completion_tokens": nomos_completion,
                "estimated_full_cost": full_cost,
                "estimated_nomos_cost": nomos_cost,
                "estimated_cost_reduction": _reduction(full_cost, nomos_cost),
            }
        )
    live_labeled = [row for row in live_rows if row.get("acceptable_tools")]
    if live_labeled:
        summary["full_selected_tool_accuracy"] = sum(
            row["conditions"]["full"].get("selected_tool") in row["acceptable_tools"]
            for row in live_labeled
        ) / len(live_labeled)
        summary["nomos_selected_tool_accuracy"] = sum(
            row["conditions"]["nomos"].get("selected_tool") in row["acceptable_tools"]
            for row in live_labeled
        ) / len(live_labeled)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--nomos-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--tokenizer",
        help="Target agent tokenizer for offline request-token estimates.",
    )
    parser.add_argument(
        "--endpoint",
        help="Optional OpenAI-compatible chat-completions URL for a live paired replay.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    cases = _read_cases(args.input)
    counter = _load_token_counter(args.tokenizer)
    ranker = DenseToolRanker.from_path(args.nomos_model)
    api_key = os.environ.get(args.api_key_env) if args.endpoint else None
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for case in cases:
        request = build_runner_request_from_openai(
            case["request"], request_id=case["case_id"]
        )
        started = time.perf_counter()
        ranked = ranker.rank(request, top_k=len(request["legal_candidate_ids"]))
        ranking_seconds = time.perf_counter() - started
        ranked_ids = [str(item["tool_id"]) for item in ranked]
        full_body = dict(case["request"])
        nomos_body = retain_openai_tools(full_body, ranked_ids[: args.top_k])
        conditions = {
            "full": _condition_metrics(full_body, counter),
            "nomos": _condition_metrics(nomos_body, counter),
        }
        if args.endpoint:
            order = ["full", "nomos"]
            rng.shuffle(order)
            for condition in order:
                body = full_body if condition == "full" else nomos_body
                try:
                    conditions[condition].update(
                        _call_provider(
                            args.endpoint,
                            body,
                            api_key=api_key,
                            timeout=args.timeout,
                        )
                    )
                except RuntimeError as exc:
                    conditions[condition]["error"] = str(exc)
        rows.append(
            {
                "case_id": case["case_id"],
                "acceptable_tools": case["acceptable_tools"],
                "ranking_seconds": ranking_seconds,
                "ranked_tool_ids": ranked_ids,
                "conditions": conditions,
            }
        )
    report = {
        "schema_version": "nomos-openai-ab.v1",
        "mode": "live" if args.endpoint else "offline",
        "nomos_model": str(args.nomos_model),
        "tokenizer": args.tokenizer,
        "summary": summarize(
            rows,
            top_k=args.top_k,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
