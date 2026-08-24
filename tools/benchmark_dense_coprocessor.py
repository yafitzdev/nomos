"""Measure deployment size, CPU ranking latency, memory, and verifier latency."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.coprocessor import coprocessor_response
from fitz_tool.dense_selector import DenseToolRanker


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _states(path: Path, partition: str, pools: set[int]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    routes: dict[int, dict[str, Any]] = {}
    verification: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evaluation_partition") != partition:
                continue
            if row.get("task_kind") == "verify" and verification is None:
                verification = row
            pool = len(row.get("legal_candidate_ids") or [])
            if (
                pool in pools
                and pool not in routes
                and row.get("task_kind") in {"route", "recover"}
                and (row.get("label") or {}).get("acceptable_tools")
            ):
                routes[pool] = row
            if set(routes) == pools and verification is not None:
                break
    missing = pools - set(routes)
    if missing:
        raise ValueError(f"input has no eligible states for candidate pools: {sorted(missing)}")
    if verification is None:
        raise ValueError("input has no verification state")
    return routes, verification


def benchmark(
    model_path: Path,
    input_path: Path,
    *,
    partition: str,
    pools: list[int],
    warm_repetitions: int,
    verifier_repetitions: int,
) -> dict[str, Any]:
    artifact_bytes = sum(
        path.stat().st_size for path in model_path.rglob("*") if path.is_file()
    )
    routes, verify_row = _states(input_path, partition, set(pools))
    rss_before = _rss_bytes()
    started = time.perf_counter()
    ranker = DenseToolRanker.from_path(
        model_path,
        device="cpu",
        query_strategy="multiview",
        candidate_strategy="multiview",
    )
    load_seconds = time.perf_counter() - started
    rss_loaded = _rss_bytes()

    pool_reports = {}
    for pool in pools:
        request = routes[pool]
        ranker._candidate_cache.clear()
        started = time.perf_counter()
        first = ranker.rank(request, top_k=3)
        cold_ms = (time.perf_counter() - started) * 1000.0
        warm_ms = []
        for _index in range(warm_repetitions):
            started = time.perf_counter()
            ranker.rank(request, top_k=3)
            warm_ms.append((time.perf_counter() - started) * 1000.0)
        acceptable = set(request["label"]["acceptable_tools"])
        pool_reports[str(pool)] = {
            "cold_registry_and_query_ms": cold_ms,
            "warm_query_p50_ms": statistics.median(warm_ms),
            "warm_query_p95_ms": _percentile(warm_ms, 0.95),
            "warm_requests_per_second": 1000.0 / statistics.mean(warm_ms),
            "top3_correct": bool(
                {str(item["tool_id"]) for item in first} & acceptable
            ),
        }

    verify_request = dict(verify_row)
    verify_request["schema_version"] = "runner-request.v2"
    verify_request["request_id"] = str(verify_row["decision_state_id"])
    verify_latencies = []
    for _index in range(verifier_repetitions):
        started = time.perf_counter()
        coprocessor_response(
            verify_request,
            [],
            router_version=ranker.version,
            calibration=ranker.calibration,
        )
        verify_latencies.append((time.perf_counter() - started) * 1000.0)
    rss_final = _rss_bytes()
    return {
        "model": str(model_path),
        "input": str(input_path),
        "partition": partition,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "artifact": {
            "bytes": artifact_bytes,
            "mebibytes": artifact_bytes / (1024**2),
        },
        "load": {
            "seconds": load_seconds,
            "rss_before_bytes": rss_before,
            "rss_loaded_bytes": rss_loaded,
            "rss_load_delta_mebibytes": (
                (rss_loaded - rss_before) / (1024**2)
                if rss_loaded is not None and rss_before is not None
                else None
            ),
            "rss_final_bytes": rss_final,
        },
        "ranking": pool_reports,
        "verification": {
            "repetitions": verifier_repetitions,
            "p50_ms": statistics.median(verify_latencies),
            "p95_ms": _percentile(verify_latencies, 0.95),
            "calls_per_second": 1000.0 / statistics.mean(verify_latencies),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--pool", action="append", type=int, default=[])
    parser.add_argument("--warm-repetitions", type=int, default=30)
    parser.add_argument("--verifier-repetitions", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = benchmark(
        args.model,
        args.input,
        partition=args.partition,
        pools=args.pool or [10, 30, 100],
        warm_repetitions=args.warm_repetitions,
        verifier_repetitions=args.verifier_repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
