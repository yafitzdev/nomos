"""Measure parallel throughput of an OpenAI-compatible synthetic-data teacher."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence


DEFAULT_BASE_URL = os.environ.get(
    "FITZ_TOOL_TEACHER_BASE_URL", "http://127.0.0.1:19003/v1"
)
DEFAULT_MODEL = os.environ.get("FITZ_TOOL_TEACHER_MODEL", "qwen3.8-27b-nvfp4")


@dataclass(frozen=True)
class RequestResult:
    latency_seconds: float
    completion_tokens: int
    prompt_tokens: int
    content: str = ""
    reasoning_content: str = ""
    finish_reason: str = ""
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep request concurrency against the synthetic-data teacher"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, action="append", default=[])
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--cases-per-request", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--show-sample", action="store_true")
    parser.add_argument("--no-api-key", action="store_true")
    parser.add_argument(
        "--stream-probe",
        action="store_true",
        help="Run one SSE request and separate first-token from decode-stream timing",
    )
    return parser


def _api_key(*, disabled: bool) -> str | None:
    if disabled:
        return None
    key = os.environ.get("FITZ_TOOL_TEACHER_API_KEY") or os.environ.get(
        "FITZ_AGENT_TEACHER_API_KEY"
    )
    if key:
        return key
    return getpass.getpass("Teacher API key: ")


def _prompt(case_number: int, case_count: int) -> str:
    case_instruction = (
        f"Create synthetic agentic-retrieval testcase {case_number}"
        if case_count == 1
        else (
            f"Create exactly {case_count} distinct synthetic agentic-retrieval testcases "
            f"numbered {case_number} through {case_number + case_count - 1}"
        )
    )
    output_instruction = (
        "Return one JSON object"
        if case_count == 1
        else f"Return one JSON array containing exactly {case_count} objects"
    )
    return f"""{case_instruction} from this source card.

Source card:
- Document: Payments API migration guide
- OAuth access tokens expire after 45 minutes.
- Refresh tokens are single-use and rotate after every successful refresh.
- Error AUTH-409 means a previously consumed refresh token was reused.
- The incident runbook says to revoke the session and require fresh authorization.

{output_instruction} with: question, capability, expected_facts, expected_tools,
expected_terminal_state, and one difficult paraphrase. Do not use Markdown and do not
invent facts. Make the question require at least two retrieval actions."""


def _request(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    case_number: int,
    cases_per_request: int,
    max_tokens: int,
    timeout: float,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You generate grounded training data as strict JSON.",
            },
            {"role": "user", "content": _prompt(case_number, cases_per_request)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "enable_thinking": False,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
            if not raw_body:
                raise ValueError(
                    f"empty HTTP {response.status} response "
                    f"content_type={response.headers.get('Content-Type')!r}"
                )
            body = json.loads(raw_body)
        usage = body.get("usage") or {}
        choice = body["choices"][0]
        return RequestResult(
            latency_seconds=time.perf_counter() - started,
            completion_tokens=int(usage.get("completion_tokens") or 0),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            content=str((choice.get("message") or {}).get("content") or ""),
            reasoning_content=str(
                (choice.get("message") or {}).get("reasoning_content") or ""
            ),
            finish_reason=str(choice.get("finish_reason") or ""),
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return RequestResult(
            latency_seconds=time.perf_counter() - started,
            completion_tokens=0,
            prompt_tokens=0,
            error=f"HTTPError {exc.code}: {detail}",
        )
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        return RequestResult(
            latency_seconds=time.perf_counter() - started,
            completion_tokens=0,
            prompt_tokens=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _stream_probe(args: argparse.Namespace, key: str | None) -> None:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You generate grounded training data as strict JSON.",
            },
            {"role": "user", "content": _prompt(0, args.cases_per_request)},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    first_delta: float | None = None
    last_delta: float | None = None
    completion_tokens = 0
    reasoning_characters = 0
    content_characters = 0
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        reported_headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() in {"server", "via", "x-request-id", "x-model"}
        }
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning = str(delta.get("reasoning_content") or "")
            content = str(delta.get("content") or "")
            if reasoning or content:
                now = time.perf_counter()
                first_delta = first_delta or now
                last_delta = now
                reasoning_characters += len(reasoning)
                content_characters += len(content)
    finished = time.perf_counter()
    stream_seconds = (
        last_delta - first_delta
        if first_delta is not None and last_delta is not None
        else 0.0
    )
    print(f"headers={json.dumps(reported_headers, sort_keys=True)}")
    print(f"time_to_first_delta_s={(first_delta or finished) - started:.3f}")
    print(f"stream_duration_s={stream_seconds:.3f}")
    print(f"total_wall_s={finished - started:.3f}")
    print(f"completion_tokens={completion_tokens}")
    print(f"reasoning_characters={reasoning_characters}")
    print(f"content_characters={content_characters}")
    if stream_seconds and completion_tokens:
        print(f"completion_tokens_per_stream_s={completion_tokens / stream_seconds:.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    levels = args.concurrency or [1, 2, 4, 8]
    if args.requests < 1 or any(level < 1 for level in levels):
        raise SystemExit("request count and concurrency must be positive")
    key = _api_key(disabled=args.no_api_key)
    if args.stream_probe:
        _stream_probe(args, key)
        return 0

    print(
        "concurrency,requests,success,json_valid,wall_s,completion_tokens,"
        "tokens_per_s,p50_s,p95_s"
    )
    for concurrency in levels:
        started = time.perf_counter()
        results: list[RequestResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _request,
                    base_url=args.base_url,
                    model=args.model,
                    api_key=key,
                    case_number=index * args.cases_per_request,
                    cases_per_request=args.cases_per_request,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                for index in range(args.requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        wall_seconds = time.perf_counter() - started
        successful = [result for result in results if result.error is None]
        latencies = [result.latency_seconds for result in successful]
        completion_tokens = sum(result.completion_tokens for result in successful)
        json_valid = 0
        valid_cases = 0
        for result in successful:
            try:
                parsed = json.loads(result.content)
                json_valid += 1
                if isinstance(parsed, list):
                    valid_cases += sum(isinstance(item, dict) for item in parsed)
                elif isinstance(parsed, dict):
                    valid_cases += 1
            except ValueError:
                pass
        throughput = completion_tokens / wall_seconds if wall_seconds else 0.0
        print(
            f"{concurrency},{args.requests},{len(successful)},{json_valid},{wall_seconds:.2f},"
            f"{completion_tokens},{throughput:.2f},"
            f"{statistics.median(latencies) if latencies else 0.0:.2f},"
            f"{_percentile(latencies, 0.95):.2f}"
        )
        print(
            f"cases requested={args.requests * args.cases_per_request} "
            f"parsed={valid_cases} cases_per_s={valid_cases / wall_seconds:.3f}"
        )
        if args.show_sample and successful:
            sample = successful[0]
            print(
                f"sample concurrency={concurrency} finish_reason={sample.finish_reason}:\n"
                f"content={sample.content}\nreasoning={sample.reasoning_content[:2000]}"
            )
        for failure in (result for result in results if result.error):
            print(f"error concurrency={concurrency}: {failure.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

