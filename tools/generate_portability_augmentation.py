"""Generate teacher-worded training rows for weak external-registry cases.

The original 50k corpus remains frozen. This tool creates a separate training
corpus with fresh matrix cells, unseen registry metadata styles, and extra
coverage for capabilities that underperformed on the external-registry suite.
Natural-language wording is always supplied by the configured teacher.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Sequence

from fitz_tool.external_registry_fixtures import (
    TARGET_CAPABILITIES,
    build_portability_augmentation_registry,
)
from fitz_tool.generic_pilot_v3 import (
    GENERIC_DATASET_VERSION,
    GENERIC_MATRIX_PATH,
    GENERIC_PILOT_SEED,
    TRAIN_TEMPLATE_IDS,
    _build_state,
    _instance_signature,
    _sample_cell,
    _source_cards,
    _type_signature,
    generic_matrix_cell_id,
    load_generic_matrix_spec,
    validate_generic_state,
)
from fitz_tool.router_v2 import FEATURE_VERSION
from tools.generate_generic_ninfer_v3 import (
    DEFAULT_BASE_URL,
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MODEL,
    _api_key,
    _request_and_validate_batch,
)


DEFAULT_BASE_INPUT = Path("data/generated/nomos_generic_ninfer_50000.jsonl")
DEFAULT_OUTPUT = Path("data/generated/nomos_generic_portability_augmentation_20000.jsonl")
DEFAULT_MANIFEST = Path("runs/nomos_generic_portability_augmentation_20000_manifest.json")
DEFAULT_COUNT = 20_000
DEFAULT_START_INDEX = 50_000
PORTABILITY_COHORT = "portability_augmentation"
PROMPT_VERSION = "generic-ninfer-v3-portability-augmentation.v1"
QUESTION_MARKER_RE = re.compile(r"(?<![a-z0-9_])(fitz|sage|bm25)(?![a-z0-9_])", re.IGNORECASE)

# The first five groups are deliberately overrepresented because they were the
# weakest capabilities on the frozen four-registry portability test.
FOCUSED_TARGET_COUNTS = {
    "compare_evidence": 5_000,
    "search_content": 4_000,
    "search_metadata": 3_000,
    "exact_pattern_search": 3_000,
    "list_sources": 2_000,
}


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _load_ledger(path: Path) -> dict[str, set[str]]:
    ledger = {
        "decision_state_id": set(),
        "matrix_cell_id": set(),
        "type_signature": set(),
        "instance_signature": set(),
        "question": set(),
        "teacher_paraphrase": set(),
    }
    duplicate_fields: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ledger:
                value = row.get(field)
                if value is None:
                    continue
                key = _normalized_text(value) if field in {"question", "teacher_paraphrase"} else str(value)
                if field != "teacher_paraphrase" and key in ledger[field]:
                    duplicate_fields[field] += 1
                ledger[field].add(key)
    if duplicate_fields:
        raise RuntimeError(f"base corpus contains duplicate keys: {dict(duplicate_fields)}")
    if not ledger["decision_state_id"]:
        raise ValueError(f"base corpus is empty: {path}")
    return ledger


def _target_sequence(count: int, seed: int) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    targets: list[str] = []
    remaining = count
    for target, target_count in FOCUSED_TARGET_COUNTS.items():
        take = min(target_count, remaining)
        targets.extend([target] * take)
        remaining -= take
        if remaining == 0:
            break
    other_targets = [target for target in TARGET_CAPABILITIES if target not in FOCUSED_TARGET_COUNTS]
    for index in range(remaining):
        targets.append(other_targets[index % len(other_targets)])
    random.Random(seed + 17).shuffle(targets)
    return targets


def _skeleton_batches(
    *,
    count: int,
    seed: int,
    start_index: int,
    batch_size: int,
    base_ledger: Mapping[str, set[str]],
) -> Iterable[list[dict[str, Any]]]:
    spec = load_generic_matrix_spec(GENERIC_MATRIX_PATH)
    cards = _source_cards()
    rng = random.Random(seed)
    targets = _target_sequence(count, seed)
    used_cells = set(base_ledger["matrix_cell_id"])
    used_types = set(base_ledger["type_signature"])
    used_instances = set(base_ledger["instance_signature"])
    batch: list[dict[str, Any]] = []

    for offset, target in enumerate(targets):
        index = start_index + offset
        cell = _sample_cell(rng, spec, target, used_cells)
        cell["agent_contract_profile"] = "registry_portability_augmentation"
        cell_id = generic_matrix_cell_id(cell)
        if cell_id in used_cells:
            raise RuntimeError(f"duplicate matrix cell in augmentation at offset {offset}")
        used_cells.add(cell_id)
        style = ("atlas", "sable", "orbit")[offset % 3]
        registry = build_portability_augmentation_registry(style, offset // 3)
        template_id = TRAIN_TEMPLATE_IDS[offset % len(TRAIN_TEMPLATE_IDS)]
        row_seed = seed + index * 1009 + offset
        state = _build_state(
            index,
            "train",
            target,
            registry,
            cell,
            cards,
            template_id,
            row_seed,
            f"{PORTABILITY_COHORT}|{style}|registry-{offset // 3 % 997:03d}|template-{offset % len(TRAIN_TEMPLATE_IDS)}",
        )
        state["evaluation_cohort"] = PORTABILITY_COHORT
        state["evaluation_partition"] = "train"
        state["type_signature"] = _type_signature(state)
        state["instance_signature"] = _instance_signature(state)
        if state["type_signature"] in used_types:
            raise RuntimeError(f"duplicate type signature in augmentation at offset {offset}")
        if state["instance_signature"] in used_instances:
            raise RuntimeError(f"duplicate instance signature in augmentation at offset {offset}")
        used_types.add(state["type_signature"])
        used_instances.add(state["instance_signature"])
        batch.append(state)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _read_existing_output(path: Path) -> tuple[set[str], set[str], set[str], set[str], set[str], set[str]]:
    ids: set[str] = set()
    questions: set[str] = set()
    paraphrases: set[str] = set()
    instances: set[str] = set()
    matrix_cells: set[str] = set()
    type_signatures: set[str] = set()
    if not path.exists():
        return ids, questions, paraphrases, instances, matrix_cells, type_signatures
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank row in existing augmentation at line {line_number}")
            row = json.loads(line)
            decision_state_id = str(row.get("decision_state_id") or "")
            if not decision_state_id:
                raise RuntimeError(f"missing decision_state_id in existing augmentation at line {line_number}")
            if decision_state_id in ids:
                raise RuntimeError(f"duplicate decision_state_id in existing augmentation: {decision_state_id}")
            ids.add(decision_state_id)
            questions.add(_normalized_text(row.get("question")))
            paraphrases.add(_normalized_text(row.get("teacher_paraphrase")))
            instances.add(str(row.get("instance_signature") or ""))
            matrix_cells.add(str(row.get("matrix_cell_id") or ""))
            type_signatures.add(str(row.get("type_signature") or ""))
    return ids, questions, paraphrases, instances, matrix_cells, type_signatures


def _request_unique_batch(
    *,
    states: list[dict[str, Any]],
    seen_questions: set[str],
    seen_paraphrases: set[str],
    seen_lock: Lock,
    base_url: str,
    model: str,
    teacher: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    max_tokens: int,
    invalid_retries: int,
    duplicate_retries: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str | None, int]:
    pending = list(states)
    accepted: dict[str, dict[str, Any]] = {}
    invalid_attempts = 0
    for attempt in range(duplicate_retries + 1):
        rows, error, invalid_count = _request_and_validate_batch(
            states=pending,
            base_url=base_url,
            model=model,
            teacher=teacher,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
            invalid_retries=invalid_retries,
            seed=seed + attempt,
        )
        invalid_attempts += invalid_count
        if error:
            return [], error, invalid_attempts
        next_pending: list[dict[str, Any]] = []
        for state, row in zip(pending, rows):
            question_key = _normalized_text(row.get("question"))
            paraphrase_key = _normalized_text(row.get("teacher_paraphrase"))
            with seen_lock:
                duplicate = (
                    not question_key
                    or question_key in seen_questions
                    or not paraphrase_key
                    or paraphrase_key in seen_paraphrases
                )
                if not duplicate:
                    seen_questions.add(question_key)
                    seen_paraphrases.add(paraphrase_key)
            if duplicate:
                next_pending.append(state)
            else:
                accepted[str(state["decision_state_id"])] = row
        pending = next_pending
        if not pending:
            return [accepted[str(state["decision_state_id"])] for state in states], None, invalid_attempts
        if attempt < duplicate_retries:
            time.sleep(0.2)
    failed = ", ".join(str(state["decision_state_id"]) for state in pending[:5])
    return [], f"teacher repeated duplicate wording after {duplicate_retries + 1} attempts: {failed}", invalid_attempts


def _validate_new_rows(
    rows: Iterable[Mapping[str, Any]],
    base_ledger: Mapping[str, set[str]],
) -> dict[str, Any]:
    seen = {field: set(values) for field, values in base_ledger.items()}
    target_counts: Counter[str] = Counter()
    rows_count = 0
    for row in rows:
        report = validate_generic_state(row)
        if not report.valid:
            raise RuntimeError(json.dumps(report.as_dict(), sort_keys=True))
        rows_count += 1
        target = str((row.get("sampling_context") or {}).get("target_capability"))
        target_counts[target] += 1
        for field in ("decision_state_id", "matrix_cell_id", "type_signature", "instance_signature"):
            value = str(row.get(field) or "")
            if value in seen[field]:
                raise RuntimeError(f"duplicate {field}: {value}")
            seen[field].add(value)
        for field in ("question", "teacher_paraphrase"):
            value = _normalized_text(row.get(field))
            if not value or value in seen[field]:
                raise RuntimeError(f"duplicate or empty {field}: {value!r}")
            if QUESTION_MARKER_RE.search(value):
                raise RuntimeError(f"project-specific marker in {field}: {value!r}")
            seen[field].add(value)
    return {"count": rows_count, "target_capability_counts": dict(sorted(target_counts.items()))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--seed", type=int, default=GENERIC_PILOT_SEED + 70000)
    parser.add_argument("--teacher", choices=("ninfer", "deepseek"), default="ninfer")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--invalid-retries", type=int, default=3)
    parser.add_argument("--duplicate-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--no-api-key", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.count, args.batch_size, args.concurrency) < 1:
        raise SystemExit("count, batch-size, and concurrency must be positive")
    if args.start_index < 0 or args.invalid_retries < 0 or args.duplicate_retries < 0:
        raise SystemExit("start-index must be non-negative; retry counts cannot be negative")
    if not args.base_input.exists():
        raise SystemExit(f"base input does not exist: {args.base_input}")
    base_ledger = _load_ledger(args.base_input)
    (
        existing_ids,
        existing_questions,
        existing_paraphrases,
        existing_instances,
        existing_matrix_cells,
        existing_type_signatures,
    ) = (
        _read_existing_output(args.output)
        if args.resume
        else (set(), set(), set(), set(), set(), set())
    )
    if len(existing_ids) > args.count:
        raise SystemExit(f"existing augmentation has {len(existing_ids)} rows but count is {args.count}")
    seen_questions = set(base_ledger["question"]) | existing_questions
    seen_paraphrases = set(base_ledger["teacher_paraphrase"]) | existing_paraphrases
    base_ledger = {field: set(values) for field, values in base_ledger.items()}
    base_ledger["decision_state_id"].update(existing_ids)
    base_ledger["instance_signature"].update(existing_instances)
    base_ledger["matrix_cell_id"].update(existing_matrix_cells)
    base_ledger["type_signature"].update(existing_type_signatures)
    base_url = args.base_url or (DEFAULT_DEEPSEEK_BASE_URL if args.teacher == "deepseek" else DEFAULT_BASE_URL)
    model = args.model or (DEFAULT_DEEPSEEK_MODEL if args.teacher == "deepseek" else DEFAULT_MODEL)
    api_key = _api_key(no_api_key=args.no_api_key, teacher=args.teacher)
    mode = "a" if args.resume and args.output.exists() else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_count = len(existing_ids)
    invalid_count = 0
    started = time.perf_counter()
    target_counts: Counter[str] = Counter()
    seen_lock = Lock()

    with args.output.open(mode, encoding="utf-8") as handle:
        pending: list[tuple[list[dict[str, Any]], Any]] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for skeletons in _skeleton_batches(
                count=args.count,
                seed=args.seed,
                start_index=args.start_index,
                batch_size=args.batch_size,
                base_ledger=base_ledger,
            ):
                todo = [state for state in skeletons if state["decision_state_id"] not in existing_ids]
                if not todo:
                    continue
                future = executor.submit(
                    _request_unique_batch,
                    states=todo,
                    seen_questions=seen_questions,
                    seen_paraphrases=seen_paraphrases,
                    seen_lock=seen_lock,
                    base_url=base_url,
                    model=model,
                    teacher=args.teacher,
                    api_key=api_key,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_tokens=args.max_tokens,
                    invalid_retries=args.invalid_retries,
                    duplicate_retries=args.duplicate_retries,
                    seed=args.seed,
                )
                pending.append((todo, future))
                if len(pending) >= args.concurrency * 2:
                    states, request_future = pending.pop(0)
                    rows, error, invalid_attempts = request_future.result()
                    invalid_count += invalid_attempts
                    if error:
                        raise RuntimeError(error)
                    for row in rows:
                        target_counts[str(row["sampling_context"]["target_capability"])] += 1
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        generated_count += 1
                    handle.flush()
                    if generated_count % (args.batch_size * 8) == 0:
                        elapsed = time.perf_counter() - started
                        print(f"generated={generated_count}/{args.count} elapsed_seconds={elapsed:.1f}", flush=True)
            for states, request_future in pending:
                rows, error, invalid_attempts = request_future.result()
                invalid_count += invalid_attempts
                if error:
                    raise RuntimeError(error)
                for row in rows:
                    target_counts[str(row["sampling_context"]["target_capability"])] += 1
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    generated_count += 1
                handle.flush()

    if generated_count != args.count:
        raise RuntimeError(f"generated {generated_count} rows; expected {args.count}")
    report = _validate_new_rows(
        (json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()),
        {field: set(values) for field, values in _load_ledger(args.base_input).items()},
    )
    manifest = {
        "dataset_version": GENERIC_DATASET_VERSION,
        "augmentation_version": "generic-portability-augmentation.v1",
        "teacher": args.teacher,
        "model": model,
        "base_url": base_url,
        "prompt_version": PROMPT_VERSION,
        "feature_version": FEATURE_VERSION,
        "base_input": str(args.base_input),
        "base_count": len(_load_ledger(args.base_input)["decision_state_id"]),
        "count": generated_count,
        "start_index": args.start_index,
        "target_capability_counts": dict(sorted(target_counts.items())),
        "validation": report,
        "invalid_attempts": invalid_count,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
