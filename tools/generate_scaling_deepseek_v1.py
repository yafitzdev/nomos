"""Generate exactly 25,000 accepted scaling rows with resumable DeepSeek batches."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.scaling_matrix_v1 import (
    DATASET_VERSION,
    MATRIX_VERSION,
    PROMPT_VERSION,
    digest,
    generation_prompt,
    load_scaling_matrix,
    materialize_row,
    normalized_question,
    replacement_assignment,
    semantic_signature,
    validate_assignments,
)


API_URL = "https://api.deepseek.com/v1/chat/completions"


@dataclass
class RequestFailure(Exception):
    reason: str
    category: str
    returned_items: int = 0
    finish_reason: str = ""

    def __str__(self) -> str:
        return self.reason


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _append_jsonl(path: Path, values: Sequence[Mapping[str, Any]], lock: threading.Lock) -> None:
    if not values:
        return
    rendered = "".join(json.dumps(value, sort_keys=True) + "\n" for value in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()


def _request_once(
    assignments: list[Mapping[str, Any]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Write high-quality synthetic agent requests as strict JSON. Ground-truth routing labels are supplied and must not be changed.",
            },
            {"role": "user", "content": generation_prompt(assignments)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except TimeoutError as exc:
        raise RequestFailure(f"timeout: {exc}", "timeout") from exc
    except urllib.error.HTTPError as exc:
        raise RequestFailure(f"HTTP {exc.code}", "http_error") from exc
    except urllib.error.URLError as exc:
        raise RequestFailure(f"network error: {exc.reason}", "network_error") from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestFailure("response body was not JSON", "malformed_json") from exc
    try:
        choice = body["choices"][0]
        finish_reason = str(choice.get("finish_reason") or "")
        content = json.loads(choice["message"]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RequestFailure("response content was not strict JSON", "malformed_json") from exc
    items = content.get("items") if isinstance(content, Mapping) else None
    returned = len(items) if isinstance(items, list) else 0
    if finish_reason != "stop":
        raise RequestFailure(
            f"unexpected finish reason: {finish_reason}",
            "finish_reason",
            returned_items=returned,
            finish_reason=finish_reason,
        )
    if not isinstance(items, list):
        raise RequestFailure("items was not an array", "wrong_item_count")
    expected_ids = [str(value["assignment_id"]) for value in assignments]
    actual_ids = [str(value.get("assignment_id")) for value in items if isinstance(value, Mapping)]
    if len(items) != len(assignments) or len(actual_ids) != len(items):
        raise RequestFailure(
            f"wrong item count: expected {len(assignments)}, received {len(items)}",
            "wrong_item_count",
            returned_items=returned,
            finish_reason=finish_reason,
        )
    if len(actual_ids) != len(set(actual_ids)):
        raise RequestFailure("duplicate assignment IDs in response", "duplicate_assignment_id", returned)
    if set(actual_ids) != set(expected_ids):
        missing = len(set(expected_ids) - set(actual_ids))
        unknown = len(set(actual_ids) - set(expected_ids))
        raise RequestFailure(
            f"assignment ID mismatch: missing={missing} unknown={unknown}",
            "assignment_id_mismatch",
            returned,
        )
    required_fields = {"assignment_id", "question", "current_step", "completed_steps"}
    if any(set(item) != required_fields for item in items):
        raise RequestFailure("response item fields did not match the strict schema", "wrong_fields", returned)
    by_id = {str(item["assignment_id"]): dict(item) for item in items}
    usage = body.get("usage") or {}
    metadata = {
        "finish_reason": finish_reason,
        "returned_items": returned,
        "usage": {
            key: int(usage.get(key) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    return [by_id[value] for value in expected_ids], metadata


def _process_batch(
    assignments: list[Mapping[str, Any]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    max_tokens: int,
    same_batch_attempts: int,
    depth: int = 0,
) -> tuple[list[tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Retry one batch twice, then split only the persistent failed batch."""

    events: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    batch_hash = digest([value["assignment_id"] for value in assignments])
    for attempt in range(same_batch_attempts):
        started = time.perf_counter()
        try:
            items, metadata = _request_once(
                assignments,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
            )
        except RequestFailure as exc:
            event = {
                "event": "request",
                "batch_hash": batch_hash,
                "assignment_ids": [value["assignment_id"] for value in assignments],
                "batch_size": len(assignments),
                "depth": depth,
                "attempt": attempt + 1,
                "status": "failed",
                "error_category": exc.category,
                "error": exc.reason[:500],
                "returned_items": exc.returned_items,
                "finish_reason": exc.finish_reason,
                "elapsed_seconds": time.perf_counter() - started,
            }
            events.append(event)
            lineage.append({key: event[key] for key in ("batch_hash", "batch_size", "depth", "attempt", "status", "error_category")})
            if attempt + 1 < same_batch_attempts:
                time.sleep(0.5 * (2**attempt))
            continue
        event = {
            "event": "request",
            "batch_hash": batch_hash,
            "assignment_ids": [value["assignment_id"] for value in assignments],
            "batch_size": len(assignments),
            "depth": depth,
            "attempt": attempt + 1,
            "status": "accepted_response",
            "error_category": None,
            "returned_items": metadata["returned_items"],
            "finish_reason": metadata["finish_reason"],
            "usage": metadata["usage"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        events.append(event)
        success_lineage = [
            *lineage,
            {key: event[key] for key in ("batch_hash", "batch_size", "depth", "attempt", "status")},
        ]
        return [
            (assignment, item, success_lineage)
            for assignment, item in zip(assignments, items)
        ], [], events
    if len(assignments) > 4:
        midpoint = len(assignments) // 2 if len(assignments) > 8 else 4
        halves = (assignments[:midpoint], assignments[midpoint:])
        split_event = {
            "event": "split",
            "batch_hash": batch_hash,
            "assignment_ids": [value["assignment_id"] for value in assignments],
            "batch_size": len(assignments),
            "depth": depth,
            "child_sizes": [len(value) for value in halves],
        }
        events.append(split_event)
        successes: list[tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
        failures: list[dict[str, Any]] = []
        for half in halves:
            child_successes, child_failures, child_events = _process_batch(
                half,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                same_batch_attempts=same_batch_attempts,
                depth=depth + 1,
            )
            successes.extend(child_successes)
            failures.extend(child_failures)
            events.extend(child_events)
        return successes, failures, events
    failures = [
        {
            "assignment_id": value["assignment_id"],
            "slot_id": value["slot_id"],
            "replacement_ordinal": value["replacement_ordinal"],
            "reason": "persistent_request_failure",
            "batch_hash": batch_hash,
            "depth": depth,
        }
        for value in assignments
    ]
    return [], failures, events


def _holdout_identities(path: Path | None) -> dict[str, set[str]]:
    identities = {
        "question": set(),
        "source_row_hash": set(),
        "registry_fingerprint": set(),
        "question_template_id": set(),
        "scenario_id": set(),
    }
    if path is None:
        return identities
    for row in _load_jsonl(path):
        identities["question"].add(normalized_question(str(row.get("question") or "")))
        identities["source_row_hash"].add(str((row.get("provenance") or {}).get("source_row_hash") or ""))
        identities["registry_fingerprint"].add(str((row.get("provenance") or {}).get("registry_fingerprint") or ""))
        identities["question_template_id"].add(str(row.get("question_template_id") or ""))
        identities["scenario_id"].add(str(row.get("scenario_id") or ""))
    return identities


def _overlap_reason(row: Mapping[str, Any], holdout: Mapping[str, set[str]]) -> str | None:
    values = {
        "question": normalized_question(str(row.get("question") or "")),
        "source_row_hash": str((row.get("provenance") or {}).get("source_row_hash") or ""),
        "registry_fingerprint": str((row.get("provenance") or {}).get("registry_fingerprint") or ""),
        "question_template_id": str(row.get("question_template_id") or ""),
        "scenario_id": str(row.get("scenario_id") or ""),
    }
    return next((f"frozen_holdout_{name}_overlap" for name, value in values.items() if value and value in holdout[name]), None)


def _repair_directive(detail: str) -> str:
    if "recovery is not observable" in detail:
        return "Explicitly use words such as earlier candidates failed or the previous options were rejected, and ask for a different route."
    if "completed_steps count" in detail:
        return "Correct the completed_steps array length exactly; do not describe completed steps only in prose."
    if "duplicate" in detail:
        return "Use substantially fresh wording and a new concrete example while preserving the assigned outcome."
    return "Correct the prior validation failure while preserving all assigned constraints."


def _stratified_canary(assignments: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 1 or count > len(assignments):
        raise ValueError("canary count must be between one and the full matrix size")
    families: dict[str, list[dict[str, Any]]] = {}
    for assignment in assignments:
        family = str(assignment["matrix_cell"]["scenario_family"])
        families.setdefault(family, []).append(assignment)
    selected = []
    offset = 0
    ordered_families = sorted(families)
    while len(selected) < count:
        for family in ordered_families:
            if len(selected) >= count:
                break
            if offset < len(families[family]):
                selected.append(families[family][offset])
        offset += 1
    return selected


def build_parser() -> argparse.ArgumentParser:
    spec = load_scaling_matrix()
    generation = spec["generation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=generation["model"])
    parser.add_argument("--batch-size", type=int, default=generation["batch_size"])
    parser.add_argument("--concurrency", type=int, default=generation["concurrency"])
    parser.add_argument("--max-tokens", type=int, default=generation["max_tokens"])
    parser.add_argument("--timeout", type=float, default=generation["timeout_seconds"])
    parser.add_argument("--same-batch-attempts", type=int, default=generation["same_batch_attempts"])
    parser.add_argument("--max-replacements", type=int, default=8)
    parser.add_argument("--canary-count", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("FITZ_TOOL_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("set FITZ_TOOL_DEEPSEEK_API_KEY")
    if min(args.batch_size, args.concurrency, args.max_tokens, args.same_batch_attempts) < 1:
        raise SystemExit("batch size, concurrency, max tokens, and attempts must be positive")
    if args.batch_size != 16 or args.concurrency != 512 or args.max_tokens != 8000:
        raise SystemExit("production scaling requires batch-size=16 concurrency=512 max-tokens=8000")
    spec = load_scaling_matrix()
    base_assignments = _load_jsonl(args.assignments)
    validate_assignments(base_assignments, spec)
    if args.canary_count:
        base_assignments = _stratified_canary(base_assignments, args.canary_count)
    target_slots = len(base_assignments)
    by_slot = {str(value["slot_id"]): value for value in base_assignments}
    holdout = _holdout_identities(args.holdout)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    existing = _load_jsonl(partial)
    if args.output.exists() and not existing:
        existing = _load_jsonl(args.output)
    accepted_by_slot: dict[str, dict[str, Any]] = {}
    normalized_questions: set[str] = set()
    semantic_signatures: set[str] = set()
    for row in existing:
        report = validate_decision_state_v2(row)
        slot_id = str(row.get("split_group_id") or "").removeprefix("scenario-")
        if not report.valid or slot_id not in by_slot:
            raise ValueError(f"invalid resumable row {row.get('decision_state_id')}: {report.as_dict()}")
        if slot_id in accepted_by_slot:
            raise ValueError(f"duplicate accepted slot in partial output: {slot_id}")
        accepted_by_slot[slot_id] = row
        normalized_questions.add(normalized_question(str(row["question"])))
        semantic_signatures.add(semantic_signature(str(row["question"]), str(row.get("teacher_paraphrase") or "")))
    prior_rejects = _load_jsonl(args.rejects)
    replacement_by_slot: Counter[str] = Counter()
    repair_by_slot: dict[str, str] = {}
    for rejection in prior_rejects:
        rejected_slot = str(rejection.get("slot_id"))
        replacement_by_slot[rejected_slot] = max(
            replacement_by_slot[rejected_slot],
            int(rejection.get("replacement_ordinal") or 0) + 1,
        )
        repair_by_slot[rejected_slot] = _repair_directive(str(rejection.get("detail") or ""))
    lock = threading.Lock()
    started = time.perf_counter()
    round_index = 0
    while len(accepted_by_slot) < target_slots:
        pending = []
        for slot_id, base in by_slot.items():
            if slot_id in accepted_by_slot:
                continue
            ordinal = int(replacement_by_slot[slot_id])
            if ordinal > args.max_replacements:
                raise RuntimeError(f"replacement limit exhausted for {slot_id}")
            next_assignment = base if ordinal == 0 else replacement_assignment(base, ordinal)
            if slot_id in repair_by_slot:
                next_assignment = dict(next_assignment)
                next_assignment["_repair_directive"] = repair_by_slot[slot_id]
            pending.append(next_assignment)
        unresolved_count = len(pending)
        if 0 < len(pending) < 4:
            companion_slots = [slot_id for slot_id in accepted_by_slot if slot_id not in {value["slot_id"] for value in pending}]
            for padding_index, slot_id in enumerate(companion_slots[: 4 - len(pending)]):
                padding = replacement_assignment(
                    by_slot[slot_id], 90 + round_index * 4 + padding_index
                )
                padding["_padding_only"] = True
                pending.append(padding)
        batches = [pending[index : index + args.batch_size] for index in range(0, len(pending), args.batch_size)]
        print(
            f"generation_round={round_index} pending_slots={unresolved_count} request_assignments={len(pending)} requests={len(batches)} accepted={len(accepted_by_slot)}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    _process_batch,
                    batch,
                    api_key=api_key,
                    model=args.model,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    same_batch_attempts=args.same_batch_attempts,
                ): batch
                for batch in batches
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                successes, failures, events = future.result()
                _append_jsonl(args.request_log, events, lock)
                rejections = list(failures)
                accepted_rows = []
                for assignment, generated, retry_history in successes:
                    if assignment.get("_padding_only"):
                        rejections.append(
                            {
                                "assignment_id": assignment["assignment_id"],
                                "slot_id": assignment["slot_id"],
                                "replacement_ordinal": assignment["replacement_ordinal"],
                                "matrix_cell_id": assignment["matrix_cell_id"],
                                "reason": "padding_only_not_training",
                                "teacher_output": generated,
                            }
                        )
                        continue
                    try:
                        row = materialize_row(
                            assignment,
                            generated,
                            model=args.model,
                            generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                            retry_history=retry_history,
                        )
                        question_key = normalized_question(str(row["question"]))
                        meaning_key = semantic_signature(str(row["question"]), str(row["teacher_paraphrase"]))
                        reason = _overlap_reason(row, holdout)
                        if question_key in normalized_questions:
                            reason = "duplicate_question"
                        elif meaning_key in semantic_signatures:
                            reason = "duplicate_semantic_signature"
                        if reason:
                            raise ValueError(reason)
                    except (ValueError, RuntimeError) as exc:
                        rejections.append(
                            {
                                "assignment_id": assignment["assignment_id"],
                                "slot_id": assignment["slot_id"],
                                "replacement_ordinal": assignment["replacement_ordinal"],
                                "matrix_cell_id": assignment["matrix_cell_id"],
                                "reason": "materialization_rejection",
                                "detail": str(exc)[:1000],
                                "teacher_output": generated,
                            }
                        )
                        continue
                    accepted_rows.append(row)
                    accepted_by_slot[str(assignment["slot_id"])] = row
                    normalized_questions.add(question_key)
                    semantic_signatures.add(meaning_key)
                for rejection in rejections:
                    if rejection.get("reason") == "padding_only_not_training":
                        continue
                    slot_id = str(rejection["slot_id"])
                    replacement_by_slot[slot_id] = max(
                        replacement_by_slot[slot_id],
                        int(rejection.get("replacement_ordinal") or 0) + 1,
                    )
                    repair_by_slot[slot_id] = _repair_directive(str(rejection.get("detail") or ""))
                _append_jsonl(partial, accepted_rows, lock)
                _append_jsonl(args.rejects, rejections, lock)
                if completed % 32 == 0:
                    print(
                        f"completed_requests={completed}/{len(batches)} accepted={len(accepted_by_slot)} rejected_total={len(prior_rejects) + sum(replacement_by_slot.values())}",
                        flush=True,
                    )
        round_index += 1
    rows = sorted(accepted_by_slot.values(), key=lambda value: int(value["matrix_cell"]["slot"]))
    if len(rows) != target_slots or len({row["matrix_cell_id"] for row in rows}) != target_slots:
        raise RuntimeError(f"final cohort is not exactly {target_slots} matrix-unique accepted rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    request_events = _load_jsonl(args.request_log)
    rejects = _load_jsonl(args.rejects)
    requests = [value for value in request_events if value.get("event") == "request"]
    usage: Counter[str] = Counter()
    for event in requests:
        usage.update(event.get("usage") or {})
    manifest = {
        "dataset_version": DATASET_VERSION,
        "matrix_version": MATRIX_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "thinking": {"type": "disabled"},
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout,
        "same_batch_attempts": args.same_batch_attempts,
        "mode": "canary" if args.canary_count else "production",
        "requested_slots": len(base_assignments),
        "api_requests_made": len(requests),
        "primary_requests": sum(value.get("depth") == 0 and value.get("attempt") == 1 for value in requests),
        "rows_returned": sum(int(value.get("returned_items") or 0) for value in requests),
        "accepted_rows": len(rows),
        "rejected_rows": len(rejects),
        "retry_requests": sum(int(value.get("attempt") or 0) > 1 for value in requests),
        "batch_splits": sum(value.get("event") == "split" for value in request_events),
        "timeouts": sum(value.get("error_category") == "timeout" for value in requests),
        "malformed_responses": sum(value.get("error_category") in {"malformed_json", "wrong_item_count", "wrong_fields", "assignment_id_mismatch", "duplicate_assignment_id"} for value in requests),
        "teacher_fallback_rows": 0,
        "replacement_rounds": round_index - 1,
        "unique_matrix_cells": len({row["matrix_cell_id"] for row in rows}),
        "unique_questions": len({normalized_question(row["question"]) for row in rows}),
        "unique_semantic_signatures": len({semantic_signature(row["question"], row["teacher_paraphrase"]) for row in rows}),
        "unique_registry_fingerprints": len({row["provenance"]["registry_fingerprint"] for row in rows}),
        "holdout_overlap": {name: 0 for name in holdout},
        "usage": dict(usage),
        "elapsed_seconds": time.perf_counter() - started,
        "resumed_rows": len(existing),
        "output": str(args.output),
        "rejects": str(args.rejects),
        "request_log": str(args.request_log),
        "cohort_sha256": digest(rows),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    api_key = ""
    os.environ.pop("FITZ_TOOL_DEEPSEEK_API_KEY", None)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
