"""Generate generic Nomos questions through an approved external teacher.

The matrix cells, registries, legal candidates, and deterministic labels are
created locally. NInfer or DeepSeek supplies the natural-language question
surfaces. The result is accepted only when the generated wording is generic and
does not leak tool identities or project-specific vocabulary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.generic_pilot_v3 import (
    CAPABILITY_FOCUS,
    GENERIC_COHORT_COUNTS,
    GENERIC_DATASET_VERSION,
    GENERIC_GENERATED_AT,
    GENERIC_MATRIX_PATH,
    GENERIC_PILOT_SEED,
    HOLDOUT_TEMPLATE_IDS,
    TARGET_CAPABILITIES,
    TASK_FOCUS,
    TRAIN_TEMPLATE_IDS,
    _build_state,
    _registry_for_row,
    _sample_cell,
    _source_cards,
    _instance_signature,
    _type_signature,
    load_generic_matrix_spec,
)
from fitz_tool.router_v2 import FEATURE_VERSION
from fitz_tool.tool_registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "generic-ninfer-v3-question.v2"
DEFAULT_BASE_URL = os.environ.get(
    "FITZ_TOOL_NINFER_BASE_URL",
    os.environ.get("FITZ_TOOL_TEACHER_BASE_URL", "http://127.0.0.1:19003/v1"),
)
DEFAULT_MODEL = os.environ.get(
    "FITZ_TOOL_NINFER_MODEL",
    os.environ.get("FITZ_TOOL_TEACHER_MODEL", "Qwen/Qwen3.8-27B"),
)
DEFAULT_DEEPSEEK_BASE_URL = os.environ.get(
    "FITZ_TOOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
)
DEFAULT_DEEPSEEK_MODEL = os.environ.get(
    "FITZ_TOOL_DEEPSEEK_MODEL", "deepseek-v4-flash"
)
MARKER_RE = re.compile(r"(?<![a-z0-9_])(fitz|sage|bm25)(?![a-z0-9_])", re.IGNORECASE)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _api_key(*, no_api_key: bool, teacher: str) -> str | None:
    if no_api_key:
        return None
    if teacher == "deepseek":
        key = (
            os.environ.get("FITZ_TOOL_DEEPSEEK_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("FITZ_TOOL_TEACHER_API_KEY")
            or os.environ.get("FITZ_AGENT_TEACHER_API_KEY")
        )
    else:
        key = os.environ.get("FITZ_TOOL_TEACHER_API_KEY") or os.environ.get(
            "FITZ_AGENT_TEACHER_API_KEY"
        )
    return key or getpass.getpass(f"{teacher} teacher API key: ")


def _prompt_version(teacher: str) -> str:
    return f"generic-{teacher}-v3-question.v2"


def _question_style(template_id: str) -> str:
    return {
        "generic_direct": "direct and concrete",
        "generic_paraphrase": "a natural paraphrase without using the obvious action verb",
        "generic_indirect": "indirect and goal-oriented",
        "generic_compositional": "compositional, combining the task, state, and a constraint",
        "generic_constraint": "explicit about safety, modality, or side-effect constraints",
        "generic_stateful": "stateful, referring to what has already been checked and what remains",
        "generic_contrastive": "contrastive, distinguishing the precise next move from a tempting alternative",
        "generic_evidence": "evidence-oriented, asking how to verify the result",
        "generic_priority": "priority-oriented under a limited step budget",
        "generic_failure": "recovery-oriented after an incomplete or ambiguous attempt",
        "generic_metadata": "resource- or catalog-oriented without naming a tool",
        "generic_action": "a concise next-action request",
        "generic_holdout_implicit": "strongly implicit and indirect",
        "generic_holdout_consequence": "framed around what must happen before a decision",
        "generic_holdout_tradeoff": "framed around precision, safety, and available context",
        "generic_holdout_sparse": "short and sparse-context",
        "generic_holdout_ambiguous": "ambiguous but resolvable from the state",
        "generic_holdout_plan": "a planning question that does not use the word plan",
    }[template_id]


def _assignment(state: Mapping[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry.from_dict(state["tool_registry"])
    legal = registry.resolve(state["legal_candidate_ids"])
    target = str((state.get("sampling_context") or {}).get("target_capability"))
    cell = state["matrix_cell"]
    return {
        "assignment_id": state["decision_state_id"],
        "target_focus": CAPABILITY_FOCUS[target],
        "task_focus": TASK_FOCUS[str(cell["task_domain"])],
        "question_style": _question_style(str(state["question_template_id"])),
        "state_context": {
            key: cell[key]
            for key in (
                "task_domain",
                "information_operation",
                "source_modality",
                "evidence_topology",
                "retrieval_obstacle",
                "agent_phase",
                "source_inventory_state",
                "query_specificity",
                "match_strategy",
                "evidence_inspection_state",
                "requirement_progress",
                "assessment_freshness",
                "candidate_set_difficulty",
                "remaining_steps",
                "unresolved_requirement_count",
                "observed_evidence_count",
                "distractor_count",
                "prior_search_count",
            )
        },
        "candidate_constraints": {
            "candidate_count": len(legal),
            "input_modalities": sorted({modality for tool in legal for modality in tool.input_modalities}),
            "output_modalities": sorted({modality for tool in legal for modality in tool.output_modalities}),
            "side_effect_classes": sorted({tool.side_effect_class for tool in legal}),
        },
    }


def _prompt(batch: list[dict[str, Any]], teacher: str = "ninfer") -> str:
    response_shape = (
        'Return a JSON object with one key, "items", whose value is the array of assignment objects.'
        if teacher == "deepseek"
        else "Return a JSON array"
    )
    return f"""Generate one synthetic user request for each assigned state below.

The task is to train a project-agnostic tool-routing coprocessor. The user request
must ask what kind of tool operation should happen next, but it must not name a
concrete tool, tool ID, registry, project, vendor, or capability identifier.
Use the supplied state and candidate semantics. Do not invent facts, tool names,
source names, or execution results. Do not answer the request; write the request
that a user would send to an agent.
Use independently worded requests across assignments; do not reuse stock wording
or copy a request from another assignment.

For each assignment return exactly:
{{"assignment_id": "...", "question": "...", "difficult_paraphrase": "..."}}

Respect each assignment's question_style field. The difficult paraphrase must ask for the same next operation with
materially different wording. Keep each string concise, ideally 40 to 160 characters; both must be 20 to 320 characters.

{response_shape} with exactly {len(batch)} objects and no Markdown.

Assignments:
{json.dumps(batch, ensure_ascii=False, sort_keys=True)}"""


def _decode(content: str, teacher: str = "ninfer") -> Any:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(value)
    if teacher == "deepseek" and isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        return parsed["items"]
    return parsed


def _request(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    batch: list[dict[str, Any]],
    teacher: str = "ninfer",
    timeout: float,
    retries: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], str | None]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Generate synthetic user requests as strict JSON only."},
            {"role": "user", "content": _prompt(batch, teacher)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False,
    }
    if teacher == "ninfer":
        payload["enable_thinking"] = False
    elif teacher == "deepseek":
        payload["thinking"] = {"type": "disabled"}
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error = "unknown NInfer error"
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
            response_content = str((body["choices"][0].get("message") or {}).get("content") or "")
            parsed = _decode(response_content, teacher)
            if not isinstance(parsed, list) or len(parsed) != len(batch):
                raise ValueError("NInfer returned the wrong number of objects")
            if not all(isinstance(item, dict) for item in parsed):
                raise ValueError("NInfer returned a non-object item")
            return parsed, None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTPError {exc.code}: {detail}"
                if exc.code == 429 and attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
            elif isinstance(exc, json.JSONDecodeError):
                last_error = (
                    f"JSONDecodeError: {exc}; response_chars={len(response_content)}; "
                    f"response_preview={response_content[:240]!r}"
                )
            else:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.75 * (attempt + 1))
    return [], last_error


def _request_with_fallback(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    batch: list[dict[str, Any]],
    teacher: str = "ninfer",
    timeout: float,
    retries: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Request a batch, splitting it when the teacher cannot serialize it."""

    outputs, error = _request(
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
        return outputs, None
    if len(batch) == 1:
        return [], error

    midpoint = max(1, len(batch) // 2)
    left, left_error = _request_with_fallback(
        base_url=base_url,
        model=model,
        api_key=api_key,
        batch=batch[:midpoint],
        teacher=teacher,
        timeout=timeout,
        retries=retries,
        max_tokens=max_tokens,
    )
    right, right_error = _request_with_fallback(
        base_url=base_url,
        model=model,
        api_key=api_key,
        batch=batch[midpoint:],
        teacher=teacher,
        timeout=timeout,
        retries=retries,
        max_tokens=max_tokens,
    )
    if left_error or right_error:
        return [], (
            f"batch recovery failed; original={error}; "
            f"left={left_error}; right={right_error}"
        )
    return left + right, None


def _valid_text(value: Any, assignment: Mapping[str, Any], registry: ToolRegistry) -> bool:
    if not isinstance(value, str) or not 20 <= len(value.strip()) <= 320:
        return False
    normalized = value.casefold()
    if MARKER_RE.search(value):
        return False
    for token in (assignment["assignment_id"], registry.registry_id):
        if str(token).casefold() in normalized:
            return False
    for tool in registry.tools:
        if tool.tool_id.casefold() in normalized:
            return False
    return True


def _teacher_row(
    state: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    model: str,
    seed: int,
    teacher: str = "ninfer",
) -> dict[str, Any] | None:
    registry = ToolRegistry.from_dict(state["tool_registry"])
    assignment = _assignment(state)
    if teacher not in {"ninfer", "deepseek"}:
        raise ValueError(f"unsupported teacher: {teacher}")
    question = generated.get("question")
    paraphrase = generated.get("difficult_paraphrase")
    if generated.get("assignment_id") != assignment["assignment_id"]:
        return None
    if not _valid_text(question, assignment, registry) or not _valid_text(paraphrase, assignment, registry):
        return None
    row = dict(state)
    row["question"] = str(question).strip()
    row["teacher_paraphrase"] = str(paraphrase).strip()
    row["agent_state"] = dict(row["agent_state"])
    row["agent_state"]["question_length_band"] = "short" if len(row["question"]) < 100 else "long"
    row["query_state"] = dict(row["query_state"])
    row["query_state"]["query_terms"] = [token for token in row["question"].lower().split() if len(token) > 4][:10]
    provenance = dict(row["provenance"])
    if teacher == "ninfer":
        source_kind = "ninfer_generic_teacher"
        artifact = "NInfer-local-endpoint"
    elif teacher == "deepseek":
        source_kind = "deepseek_generic_teacher"
        artifact = "DeepSeek-api"
    provenance.update(
        {
            "prompt_version": _prompt_version(teacher),
            "model": model,
            "artifact": artifact,
            "teacher": teacher,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "feature_version": FEATURE_VERSION,
        }
    )
    row["provenance"] = provenance
    row["source_kind"] = source_kind
    provenance["artifact"] = artifact
    provenance["teacher"] = teacher
    provenance["prompt_version"] = _prompt_version(teacher)
    row["type_signature"] = _type_signature(row)
    row["instance_signature"] = _instance_signature(row)
    return row


def _request_and_validate_batch(
    *,
    states: list[dict[str, Any]],
    base_url: str,
    model: str,
    teacher: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    max_tokens: int,
    invalid_retries: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str | None, int]:
    """Generate one batch and retry only rows rejected by generic validation."""

    pending = list(states)
    accepted: dict[str, dict[str, Any]] = {}
    invalid_attempts = 0
    last_error: str | None = None
    for attempt in range(invalid_retries + 1):
        assignments = [_assignment(state) for state in pending]
        outputs, error = _request_with_fallback(
            base_url=base_url,
            model=model,
            api_key=api_key,
            batch=assignments,
            teacher=teacher,
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
        )
        if error:
            last_error = error
            break
        next_pending: list[dict[str, Any]] = []
        for state, generated in zip(pending, outputs):
            row = _teacher_row(state, generated, model=model, seed=seed, teacher=teacher)
            if row is None:
                invalid_attempts += 1
                next_pending.append(state)
            else:
                accepted[str(state["decision_state_id"])] = row
        pending = next_pending
        if not pending:
            break
        if attempt < invalid_retries:
            time.sleep(0.25)

    if last_error:
        return [], last_error, invalid_attempts
    if pending:
        failed_ids = ", ".join(str(state["decision_state_id"]) for state in pending[:5])
        return (
            [],
            f"NInfer wording failed generic validation after {invalid_retries + 1} attempts "
            f"for {failed_ids}",
            invalid_attempts,
        )
    return [accepted[str(state["decision_state_id"])] for state in states], None, invalid_attempts


def _skeleton_batches(count: int, seed: int, batch_size: int):
    if count < 1 or count > sum(GENERIC_COHORT_COUNTS.values()):
        raise ValueError(f"count must be between 1 and {sum(GENERIC_COHORT_COUNTS.values())}")
    spec = load_generic_matrix_spec(GENERIC_MATRIX_PATH)
    cards = _source_cards()
    rng = random.Random(seed)
    used_cells: set[str] = set()
    index = 0
    batch: list[dict[str, Any]] = []
    for cohort, cohort_count in GENERIC_COHORT_COUNTS.items():
        for offset in range(min(cohort_count, count - index)):
            target = TARGET_CAPABILITIES[offset % len(TARGET_CAPABILITIES)]
            cell = _sample_cell(rng, spec, target, used_cells)
            cell["agent_contract_profile"] = {
                "train": "registry_alpha",
                "validation": "registry_beta",
                "familiar_registries": "registry_gamma",
                "unseen_tool_ids": "registry_alpha",
                "id_renames": "registry_beta",
                "schema_variants": "registry_gamma",
                "modality_variants": "registry_gamma",
                "heldout_family": "registry_delta",
                "heldout_sources": "registry_alpha",
                "heldout_questions": "registry_alpha",
                "alternate_registry": "registry_epsilon",
            }[cohort]
            cell_id = _digest(cell)
            used_cells.add(cell_id)
            registry = _registry_for_row(cohort, offset)
            template_pool = HOLDOUT_TEMPLATE_IDS if cohort == "heldout_questions" else TRAIN_TEMPLATE_IDS
            template_id = template_pool[offset % len(template_pool)]
            state = _build_state(
                index,
                cohort,
                target,
                registry,
                cell,
                cards,
                template_id,
                seed + index * 1009 + offset,
                f"{cohort}|registry-{offset % 32}|source-{offset % 16}|template-{offset % len(template_pool)}",
            )
            batch.append(state)
            index += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if index >= count:
                break
        if index >= count:
            break
    if batch:
        yield batch


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing: set[str] = set()
    invalid_count = 0
    duplicate_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                invalid_count += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            decision_state_id = row.get("decision_state_id")
            if not decision_state_id:
                invalid_count += 1
                continue
            key = str(decision_state_id)
            if key in existing:
                duplicate_count += 1
            else:
                existing.add(key)

    if invalid_count or duplicate_count:
        backup = path.with_name(
            f"{path.stem}.resume-backup-{time.time_ns()}{path.suffix}"
        )
        temporary = path.with_name(f".{path.name}.resume-repair-{time.time_ns()}.tmp")
        try:
            retained: set[str] = set()
            with path.open("r", encoding="utf-8") as source, temporary.open(
                "w", encoding="utf-8"
            ) as target:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    decision_state_id = row.get("decision_state_id")
                    if not decision_state_id:
                        continue
                    key = str(decision_state_id)
                    if key in retained:
                        continue
                    retained.add(key)
                    target.write(line if line.endswith("\n") else f"{line}\n")
            shutil.copy2(path, backup)
            temporary.replace(path)
            print(
                f"resume_repaired path={path} invalid_rows={invalid_count} "
                f"duplicate_rows={duplicate_count} backup={backup}",
                flush=True,
            )
        finally:
            if temporary.exists():
                temporary.unlink()
    return existing


def _existing_teachers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    teachers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            teacher = (row.get("provenance") or {}).get("teacher")
            if teacher:
                teachers.add(str(teacher))
    return teachers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=sum(GENERIC_COHORT_COUNTS.values()))
    parser.add_argument("--seed", type=int, default=GENERIC_PILOT_SEED)
    parser.add_argument("--teacher", choices=("ninfer", "deepseek"), default="ninfer")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--invalid-retries",
        type=int,
        default=3,
        help="Retry rows whose wording fails generic validation.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--no-api-key", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.concurrency < 1 or args.invalid_retries < 0:
        raise SystemExit("batch-size and concurrency must be positive; invalid-retries cannot be negative")
    base_url = args.base_url or (
        DEFAULT_DEEPSEEK_BASE_URL if args.teacher == "deepseek" else DEFAULT_BASE_URL
    )
    model = args.model or (DEFAULT_DEEPSEEK_MODEL if args.teacher == "deepseek" else DEFAULT_MODEL)
    existing = _existing_ids(args.output) if args.resume else set()
    existing_teachers = _existing_teachers(args.output) if args.resume else set()
    api_key = _api_key(no_api_key=args.no_api_key, teacher=args.teacher)
    generated_count = len(existing)
    invalid_count = 0
    started = time.perf_counter()
    mode = "a" if args.resume and args.output.exists() else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, encoding="utf-8") as handle:
        pending: list[tuple[list[dict[str, Any]], Any]] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for skeletons in _skeleton_batches(args.count, args.seed, args.batch_size):
                todo = [state for state in skeletons if state["decision_state_id"] not in existing]
                if not todo:
                    continue
                future = executor.submit(
                    _request_and_validate_batch,
                    states=todo,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    teacher=args.teacher,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_tokens=args.max_tokens,
                    invalid_retries=args.invalid_retries,
                    seed=args.seed,
                )
                pending.append((todo, future))
                if len(pending) >= args.concurrency * 2:
                    done = pending.pop(0)
                    skeleton_rows, request_future = done
                    rows, error, invalid_attempts = request_future.result()
                    invalid_count += invalid_attempts
                    if error:
                        raise RuntimeError(error)
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        generated_count += 1
                    handle.flush()
                    if generated_count % (args.batch_size * 8) == 0:
                        elapsed = time.perf_counter() - started
                        print(f"generated={generated_count}/{args.count} elapsed_seconds={elapsed:.1f}", flush=True)
            for skeleton_rows, request_future in pending:
                rows, error, invalid_attempts = request_future.result()
                invalid_count += invalid_attempts
                if error:
                    raise RuntimeError(error)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    generated_count += 1
                handle.flush()
    if generated_count != args.count:
        raise RuntimeError(f"generated {generated_count} rows; expected {args.count}")
    manifest = {
        "dataset_version": GENERIC_DATASET_VERSION,
        "pilot_version": "generic-pilot.v3.50k-teacher",
        "teacher": "mixed" if existing_teachers and existing_teachers != {args.teacher} else args.teacher,
        "teachers": sorted(existing_teachers | {args.teacher}),
        "model": model,
        "base_url": base_url,
        "prompt_version": _prompt_version(args.teacher),
        "feature_version": FEATURE_VERSION,
        "seed": args.seed,
        "count": generated_count,
        "invalid_rows": invalid_count,
        "invalid_retries": args.invalid_retries,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "generated_at": GENERIC_GENERATED_AT,
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
