"""Regenerate repeated teacher wording without changing matrix assignments."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tools.generate_generic_ninfer_v3 import _request_and_validate_batch
from fitz_tool.generic_pilot_v3 import validate_generic_state


DEFAULT_INPUT = Path("data/generated/nomos_generic_ninfer_50000.jsonl")
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _read_duplicate_states(path: Path) -> tuple[list[dict[str, Any]], set[str], int]:
    seen: set[str] = set()
    replacements: list[dict[str, Any]] = []
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            rows += 1
            question = row.get("question")
            if not isinstance(question, str):
                raise ValueError(f"row {line_number} has no string question")
            key = _norm(question)
            if key in seen:
                replacements.append(row)
            else:
                seen.add(key)
    return replacements, seen, rows


def _chunks(items: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _generate_batches(
    batches: Sequence[list[dict[str, Any]]],
    *,
    base_url: str,
    model: str,
    api_key: str,
    concurrency: int,
    retries: int,
    invalid_retries: int,
    timeout: float,
    max_tokens: int,
    seed: int,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    def run(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows, error, _ = _request_and_validate_batch(
            states=batch,
            base_url=base_url,
            model=model,
            api_key=api_key,
            teacher="deepseek",
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
            invalid_retries=invalid_retries,
            seed=seed,
        )
        if error:
            raise RuntimeError(error)
        return batch, rows

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run, batch) for batch in batches]
        return [future.result() for future in futures]


def _build_replacements(
    states: list[dict[str, Any]],
    used_questions: set[str],
    *,
    base_url: str,
    model: str,
    api_key: str,
    batch_size: int,
    concurrency: int,
    retries: int,
    invalid_retries: int,
    timeout: float,
    max_tokens: int,
    seed: int,
    max_attempts: int,
) -> dict[str, dict[str, Any]]:
    pending = list(states)
    replacements: dict[str, dict[str, Any]] = {}
    for attempt in range(max_attempts):
        if not pending:
            return replacements
        generated = _generate_batches(
            list(_chunks(pending, batch_size)),
            base_url=base_url,
            model=model,
            api_key=api_key,
            concurrency=concurrency,
            retries=retries,
            invalid_retries=invalid_retries,
            timeout=timeout,
            max_tokens=max_tokens,
            seed=seed + attempt * 100003,
        )
        next_pending: list[dict[str, Any]] = []
        for batch, rows in generated:
            for state, row in zip(batch, rows):
                question_key = _norm(str(row["question"]))
                if question_key in used_questions:
                    next_pending.append(state)
                    continue
                used_questions.add(question_key)
                replacements[str(state["decision_state_id"])] = row
        pending = next_pending
        print(
            f"repair_attempt={attempt + 1} accepted={len(replacements)} "
            f"pending={len(pending)}",
            flush=True,
        )
        if pending:
            time.sleep(0.25)
    failed = ", ".join(str(row["decision_state_id"]) for row in pending[:8])
    raise RuntimeError(f"could not produce unique wording for {len(pending)} rows: {failed}")


def _write_repaired(
    source: Path,
    destination: Path,
    replacements: dict[str, dict[str, Any]],
) -> int:
    written = 0
    with source.open("r", encoding="utf-8") as source_handle, destination.open(
        "w", encoding="utf-8", newline=""
    ) as destination_handle:
        for line_number, line in enumerate(source_handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            decision_state_id = str(row["decision_state_id"])
            replacement = replacements.get(decision_state_id)
            if replacement is not None:
                row = replacement
            report = validate_generic_state(row)
            if not report.valid:
                raise ValueError(f"replacement invalid at line {line_number}: {report.errors}")
            destination_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--invalid-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.concurrency < 1:
        raise SystemExit("batch-size and concurrency must be positive")
    api_key = os.environ.get("FITZ_TOOL_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("set FITZ_TOOL_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY")
    replacements, used_questions, row_count = _read_duplicate_states(args.input)
    print(f"rows={row_count} duplicate_rows_to_repair={len(replacements)}", flush=True)
    if not replacements:
        return 0
    replacement_rows = _build_replacements(
        replacements,
        used_questions,
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        retries=args.retries,
        invalid_retries=args.invalid_retries,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    temporary = args.input.with_name(f".{args.input.name}.question-repair.tmp")
    written = _write_repaired(args.input, temporary, replacement_rows)
    if written != row_count:
        raise RuntimeError(f"wrote {written} rows; expected {row_count}")
    backup = args.input.with_name(f"{args.input.stem}.before-question-repair.jsonl")
    if backup.exists():
        raise RuntimeError(f"backup already exists: {backup}")
    args.input.rename(backup)
    temporary.replace(args.input)
    print(json.dumps({"rows": written, "repaired": len(replacement_rows), "backup": str(backup)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
