"""Generate the language surface for generic Nomos agentic v1 rows."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from fitz_tool.agentic_pilot import (
    AGENTIC_DATASET_VERSION,
    AGENTIC_PILOT_VERSION,
    CAPABILITY_FOCUS,
    PROJECT_MARKER_RE,
    TASK_FOCUS,
    _digest,
    generate_agentic_states,
    validate_agentic_state,
)
from fitz_tool.router_v2 import FEATURE_VERSION
from fitz_tool.tool_registry import ToolRegistry


DEFAULT_BASE_URL = os.environ.get("FITZ_TOOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.environ.get("FITZ_TOOL_DEEPSEEK_MODEL", "deepseek-v4-flash")
PROMPT_VERSION = "agentic-deepseek-v1-question.v1"
MARKER_RE = re.compile(r"(?<![a-z0-9_])(fitz|sage|bm25)(?![a-z0-9_])", re.IGNORECASE)


def _api_key(*, no_api_key: bool, teacher: str) -> str | None:
    if no_api_key:
        return None
    names = (
        ("FITZ_TOOL_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")
        if teacher == "deepseek"
        else ("FITZ_TOOL_TEACHER_API_KEY", "FITZ_AGENT_TEACHER_API_KEY")
    )
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return getpass.getpass(f"{teacher} teacher API key: ")


def _assignment(row: Mapping[str, Any]) -> dict[str, Any]:
    cell = row["matrix_cell"]
    target = str(cell["target_capability"])
    task_kind = str(row["task_kind"])
    assignment: dict[str, Any] = {
        "assignment_id": row["decision_state_id"],
        "task_kind": task_kind,
        "target_focus": CAPABILITY_FOCUS.get(target, "additional tool candidates"),
        "task_focus": TASK_FOCUS.get(str(cell.get("task_domain")), "the current task"),
        "candidate_pool_size": cell["candidate_pool_size"],
        "top_k": cell["top_k"],
        "unseen_axis": cell["unseen_axis"],
        "question_style": (
            "ask for the next ranked operation"
            if task_kind == "route"
            else "describe that prior options were insufficient and ask for new alternatives"
            if task_kind == "recover"
            else "ask for a safe validation of a proposed operation"
        ),
    }
    if task_kind == "recover":
        assignment.update(
            {
                "recovery_round": cell["recovery_round"],
                "expansion_trigger": cell["expansion_trigger"],
                "prior_candidate_count": cell["prior_candidate_count"],
            }
        )
    if task_kind == "verify":
        assignment["validation_case"] = cell["validation_case"]
        proposed = row.get("proposed_tool_call") or {}
        assignment["proposed_call_shape"] = {
            "argument_names": sorted((proposed.get("arguments") or {}).keys()),
            "input_modality": proposed.get("input_modality"),
        }
    return assignment


def _prompt(batch: list[dict[str, Any]], teacher: str) -> str:
    shape = (
        'Return a JSON object with one key, "items", whose value is the array.'
        if teacher == "deepseek"
        else "Return a JSON array"
    )
    return f"""Generate one concise synthetic user request for each assignment below.

Nomos is a project-agnostic CPU tool-routing and call-validation coprocessor.
The requests must use generic language. Never mention a concrete tool name, tool
ID, registry ID, vendor, project, benchmark, or capability identifier. Do not
invent facts or execution results. Do not answer the request; write what a user
would send to an agent.

For route assignments, ask which operation should be selected from the available
candidate pool. For recover assignments, make clear that the earlier candidates
were insufficient and that more candidates should be requested, without naming a
hidden or arbitrary tool. For verify assignments, ask the agent to validate a
proposed operation against the current state and constraints without revealing
whether it is valid.

Return exactly one object per assignment with:
{{"assignment_id":"...","question":"...","difficult_paraphrase":"..."}}
Use distinct wording across assignments. Each string must be 20-320 characters.
{shape} with exactly {len(batch)} objects and no Markdown.

Assignments:
{json.dumps(batch, ensure_ascii=False, sort_keys=True)}"""


def _decode(content: str, teacher: str) -> list[dict[str, Any]]:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(value)
    if teacher == "deepseek" and isinstance(parsed, Mapping):
        parsed = parsed.get("items")
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("teacher response must be an array of objects")
    return parsed


def _request(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    batch: list[dict[str, Any]],
    teacher: str,
    timeout: float,
    retries: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int], str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Generate generic synthetic data as strict JSON only."},
            {"role": "user", "content": _prompt(batch, teacher)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False,
    }
    if teacher == "deepseek":
        payload["thinking"] = {"type": "disabled"}
        payload["response_format"] = {"type": "json_object"}
    else:
        payload["enable_thinking"] = False
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error = "unknown teacher error"
    response_content = ""
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            usage = body.get("usage") or {}
            response_content = str((body["choices"][0].get("message") or {}).get("content") or "")
            parsed = _decode(response_content, teacher)
            if len(parsed) != len(batch):
                raise ValueError(f"teacher returned {len(parsed)} rows for {len(batch)} assignments")
            return parsed, {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }, None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTPError {exc.code}: {detail}"
            if exc.code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, json.JSONDecodeError):
                last_error = f"JSONDecodeError: {exc}; response_chars={len(response_content)}"
            else:
                last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, last_error


def _request_with_fallback(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    batch: list[dict[str, Any]],
    teacher: str,
    timeout: float,
    retries: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int], str | None]:
    outputs, usage, error = _request(
        base_url=base_url,
        model=model,
        api_key=api_key,
        batch=batch,
        teacher=teacher,
        timeout=timeout,
        retries=retries,
        max_tokens=max_tokens,
    )
    if error is None:
        return outputs, usage, None
    if len(batch) == 1:
        return [], usage, error
    midpoint = max(1, len(batch) // 2)
    left, left_usage, left_error = _request_with_fallback(
        base_url=base_url,
        model=model,
        api_key=api_key,
        batch=batch[:midpoint],
        teacher=teacher,
        timeout=timeout,
        retries=retries,
        max_tokens=max_tokens,
    )
    right, right_usage, right_error = _request_with_fallback(
        base_url=base_url,
        model=model,
        api_key=api_key,
        batch=batch[midpoint:],
        teacher=teacher,
        timeout=timeout,
        retries=retries,
        max_tokens=max_tokens,
    )
    combined = {
        key: usage.get(key, 0) + left_usage.get(key, 0) + right_usage.get(key, 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if left_error or right_error:
        return [], combined, f"batch split failed; original={error}; left={left_error}; right={right_error}"
    return left + right, combined, None


def _valid_text(value: Any, row: Mapping[str, Any]) -> bool:
    if not isinstance(value, str) or not 20 <= len(value.strip()) <= 320:
        return False
    if MARKER_RE.search(value) or PROJECT_MARKER_RE.search(value):
        return False
    normalized = value.casefold()
    registry = ToolRegistry.from_dict(row["tool_registry"])
    for token in (row["decision_state_id"], registry.registry_id):
        if str(token).casefold() in normalized:
            return False
    for tool in registry.tools:
        if tool.tool_id.casefold() in normalized:
            return False
    return True


def _teacher_row(row: Mapping[str, Any], generated: Mapping[str, Any], model: str, teacher: str) -> dict[str, Any] | None:
    if generated.get("assignment_id") != row["decision_state_id"]:
        return None
    if not _valid_text(generated.get("question"), row) or not _valid_text(generated.get("difficult_paraphrase"), row):
        return None
    output = dict(row)
    question = str(generated["question"]).strip()
    output["question"] = question
    output["teacher_paraphrase"] = str(generated["difficult_paraphrase"]).strip()
    output["agent_state"] = dict(output["agent_state"])
    output["agent_state"]["question_length_band"] = "short" if len(question) < 100 else "long"
    output["query_state"] = dict(output["query_state"])
    output["query_state"]["query_terms"] = [token for token in question.lower().split() if len(token) > 4][:10]
    provenance = dict(output["provenance"])
    provenance.update(
        {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "artifact": "DeepSeek-api" if teacher == "deepseek" else "NInfer-local-endpoint",
            "teacher": teacher,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "feature_version": FEATURE_VERSION,
        }
    )
    output["provenance"] = provenance
    output["source_kind"] = f"{teacher}_agentic_teacher"
    output["instance_signature"] = _digest(
        {
            "type_signature": output["type_signature"],
            "question": output["question"],
            "teacher_paraphrase": output["teacher_paraphrase"],
            "proposed_tool_call": output.get("proposed_tool_call"),
            "seed": provenance.get("seed"),
        }
    )
    return output


def _generate_unique_batch(
    *,
    rows: list[dict[str, Any]],
    base_url: str,
    model: str,
    teacher: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    invalid_retries: int,
    max_tokens: int,
    seen_questions: set[str],
    seen_paraphrases: set[str],
    seen_lock: Lock,
) -> tuple[list[dict[str, Any]], dict[str, int], int, str | None]:
    pending = list(rows)
    accepted: dict[str, dict[str, Any]] = {}
    usage = Counter()
    invalid_attempts = 0
    for attempt in range(invalid_retries + 1):
        assignments = [_assignment(row) for row in pending]
        generated, request_usage, error = _request_with_fallback(
            base_url=base_url,
            model=model,
            api_key=api_key,
            batch=assignments,
            teacher=teacher,
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
        )
        usage.update(request_usage)
        if error:
            return [], dict(usage), invalid_attempts, error
        next_pending: list[dict[str, Any]] = []
        for row, item in zip(pending, generated):
            output = _teacher_row(row, item, model, teacher)
            if output is None:
                invalid_attempts += 1
                next_pending.append(row)
                continue
            question_key = " ".join(output["question"].casefold().split())
            paraphrase_key = " ".join(output["teacher_paraphrase"].casefold().split())
            with seen_lock:
                duplicate = question_key in seen_questions or paraphrase_key in seen_paraphrases
                if not duplicate:
                    seen_questions.add(question_key)
                    seen_paraphrases.add(paraphrase_key)
            if duplicate:
                invalid_attempts += 1
                next_pending.append(row)
            else:
                accepted[str(row["decision_state_id"])] = output
        pending = next_pending
        if not pending:
            return [accepted[str(row["decision_state_id"])] for row in rows], dict(usage), invalid_attempts, None
    failed = ", ".join(str(row["decision_state_id"]) for row in pending[:5])
    return [], dict(usage), invalid_attempts, f"teacher validation failed after {invalid_retries + 1} attempts: {failed}"


def _read_existing(path: Path) -> tuple[set[str], set[str], set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    questions: set[str] = set()
    paraphrases: set[str] = set()
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return ids, questions, paraphrases, rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not validate_agentic_state(row).valid:
            raise RuntimeError(f"invalid existing row at {path}:{line_number}")
        row_id = str(row["decision_state_id"])
        question = " ".join(str(row["question"]).casefold().split())
        paraphrase = " ".join(str(row.get("teacher_paraphrase") or "").casefold().split())
        if row_id in ids or question in questions or paraphrase in paraphrases:
            raise RuntimeError(f"duplicate existing row at {path}:{line_number}")
        ids.add(row_id)
        questions.add(question)
        paraphrases.add(paraphrase)
        rows.append(row)
    return ids, questions, paraphrases, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=Path("data/generated/nomos_agentic_pilot_1000.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/nomos_agentic_pilot_1000_manifest.json"))
    parser.add_argument("--teacher", choices=("deepseek", "ninfer"), default="deepseek")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--invalid-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--no-api-key", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.count, args.batch_size, args.concurrency) < 1:
        raise SystemExit("count, batch-size, and concurrency must be positive")
    base_url = args.base_url or (DEFAULT_BASE_URL if args.teacher == "deepseek" else os.environ.get("FITZ_TOOL_NINFER_BASE_URL", "http://127.0.0.1:19003/v1"))
    model = args.model or (DEFAULT_MODEL if args.teacher == "deepseek" else os.environ.get("FITZ_TOOL_NINFER_MODEL", "Qwen/Qwen3.8-27B"))
    api_key = _api_key(no_api_key=args.no_api_key, teacher=args.teacher)
    skeletons, skeleton_manifest = generate_agentic_states(args.count, seed=args.seed)
    existing_ids, seen_questions, seen_paraphrases, existing_rows = _read_existing(args.output) if args.resume else (set(), set(), set(), [])
    todo = [row for row in skeletons if row["decision_state_id"] not in existing_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_usage = Counter()
    invalid_attempts = 0
    errors: list[str] = []
    results: dict[int, list[dict[str, Any]]] = {}
    batch_index = 0
    pending: list[tuple[int, Any]] = []
    seen_lock = Lock()

    def consume(index: int, future: Any) -> None:
        nonlocal invalid_attempts
        rows, usage, invalid, error = future.result()
        total_usage.update(usage)
        invalid_attempts += invalid
        if error:
            errors.append(error)
            return
        results[index] = rows

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for offset in range(0, len(todo), args.batch_size):
            batch = todo[offset : offset + args.batch_size]
            future = executor.submit(
                _generate_unique_batch,
                rows=batch,
                base_url=base_url,
                model=model,
                teacher=args.teacher,
                api_key=api_key,
                timeout=args.timeout,
                retries=args.retries,
                invalid_retries=args.invalid_retries,
                max_tokens=args.max_tokens,
                seen_questions=seen_questions,
                seen_paraphrases=seen_paraphrases,
                seen_lock=seen_lock,
            )
            pending.append((batch_index, future))
            batch_index += 1
            if len(pending) >= args.concurrency * 2:
                index, first = pending.pop(0)
                consume(index, first)
        for index, future in pending:
            consume(index, future)

    if errors:
        raise RuntimeError("; ".join(errors[:3]))
    generated_rows: list[dict[str, Any]] = []
    for index in range(batch_index):
        generated_rows.extend(results.get(index, []))
    generated_rows.sort(key=lambda row: str(row["decision_state_id"]))
    if len(existing_rows) + len(generated_rows) != args.count:
        raise RuntimeError(f"generated {len(existing_rows) + len(generated_rows)} rows; expected {args.count}")
    all_rows = existing_rows + generated_rows
    invalid_rows = [row["decision_state_id"] for row in all_rows if not validate_agentic_state(row).valid]
    if invalid_rows:
        raise RuntimeError(f"final validation failed for {invalid_rows[:5]}")
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for row in generated_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    elapsed = time.perf_counter() - started
    manifest = {
        **skeleton_manifest,
        "teacher": args.teacher,
        "model": model,
        "base_url": base_url,
        "prompt_version": PROMPT_VERSION,
        "count": args.count,
        "existing_count": len(existing_rows),
        "generated_count": len(generated_rows),
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "retries": args.retries,
        "invalid_attempts": invalid_attempts,
        "usage": dict(total_usage),
        "teacher_rows_per_completion_token": (
            len(generated_rows) / total_usage["completion_tokens"] if total_usage["completion_tokens"] else 0.0
        ),
        "elapsed_seconds": elapsed,
        "output": str(args.output),
        "dataset_version": AGENTIC_DATASET_VERSION,
        "pilot_version": AGENTIC_PILOT_VERSION,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
