"""Measure DeepSeek JSON-row token cost and batch capacity empirically."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.agentic_pilot import generate_agentic_states
from tools.generate_agentic_ninfer_v1 import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    _api_key,
    _assignment,
    _request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,4,8,10,12,16,20")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--output", type=Path, default=Path("runs/deepseek_agentic_batch_probe.json"))
    parser.add_argument("--no-api-key", action="store_true")
    return parser


def _run_one(
    assignments: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows, usage, error = _request(
        base_url=base_url,
        model=model,
        api_key=api_key,
        batch=assignments,
        teacher="deepseek",
        timeout=timeout,
        retries=retries,
        max_tokens=max_tokens,
    )
    return {
        "batch_size": len(assignments),
        "elapsed_seconds": time.perf_counter() - started,
        "parsed_rows": len(rows),
        "usage": usage,
        "error": error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or args.requests < 1 or args.concurrency < 1:
        raise SystemExit("batch sizes, requests, and concurrency must be positive")
    largest = max(batch_sizes) * args.requests
    skeletons, _manifest = generate_agentic_states(largest, seed=args.seed)
    key = _api_key(no_api_key=args.no_api_key, teacher="deepseek")
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        assignments = [_assignment(row) for row in skeletons]
        batches = [
            assignments[offset : offset + batch_size]
            for offset in range(0, batch_size * args.requests, batch_size)
        ]
        started = time.perf_counter()
        batch_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    batch,
                    base_url=args.base_url,
                    model=args.model,
                    api_key=key,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_tokens=args.max_tokens,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                batch_results.append(future.result())
        successes = [
            result
            for result in batch_results
            if result["error"] is None and result["parsed_rows"] == batch_size
        ]
        prompt_tokens = sum(result["usage"]["prompt_tokens"] for result in successes)
        completion_tokens = sum(result["usage"]["completion_tokens"] for result in successes)
        latencies = [float(result["elapsed_seconds"]) for result in successes]
        wall = time.perf_counter() - started
        record = {
            "batch_size": batch_size,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "successful_requests": len(successes),
            "failed_requests": len(batch_results) - len(successes),
            "wall_seconds": wall,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_per_row": prompt_tokens / (len(successes) * batch_size) if successes else 0.0,
            "completion_tokens_per_row": completion_tokens / (len(successes) * batch_size) if successes else 0.0,
            "total_tokens_per_row": (prompt_tokens + completion_tokens) / (len(successes) * batch_size) if successes else 0.0,
            "rows_per_second": len(successes) * batch_size / wall if wall else 0.0,
            "p50_latency_seconds": statistics.median(latencies) if latencies else 0.0,
            "p95_latency_seconds": sorted(latencies)[min(len(latencies) - 1, round((len(latencies) - 1) * 0.95))] if latencies else 0.0,
            "errors": [result["error"] for result in batch_results if result["error"]][:3],
        }
        results.append(record)
        print(json.dumps(record, sort_keys=True))
    successful_sizes = [record["batch_size"] for record in results if record["failed_requests"] == 0]
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "requested_batch_sizes": batch_sizes,
        "concurrency": args.concurrency,
        "requests_per_batch_size": args.requests,
        "recommended_batch_size": max(successful_sizes) if successful_sizes else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
