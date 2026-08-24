"""Generate broad, stateful Nomos training rows from ToolRet triplets via DeepSeek."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.router_v2 import FEATURE_VERSION
from fitz_tool.tool_registry import SIDE_EFFECT_CLASSES, ToolRegistry


DATASET_VERSION = "nomos-toolret-agentic.v1"
MATRIX_VERSION = "matrix.toolret-agentic.v1"
PROMPT_VERSION = "toolret-agentic-deepseek.v1"
VALIDATOR_VERSION = "toolret-agentic-validator.v1"
VIEWER_BASE = "https://datasets-server.huggingface.co/rows"
VIEWER_DATASET = "mangopy/ToolRet-Training-20w"
VIEWER_CONFIG = "ToolRet-Training-20w"
VIEWER_TOTAL = 208_826
PROJECT_MARKER = re.compile(r"(?<![a-z0-9_])(fitz|sage|nomos|pyrrho)(?![a-z0-9_])", re.I)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_document(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("ToolRet documentation must decode to an object")
    return parsed


def _token(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.:-]+", "_", value.casefold()).strip("_.:-")
    if not normalized or not normalized[0].isalpha():
        normalized = f"{fallback}_{normalized}"
    return normalized[:128]


def _json_type(value: Any) -> str:
    name = str(value or "string").casefold()
    if "bool" in name:
        return "boolean"
    if "int" in name:
        return "integer"
    if "float" in name or "number" in name:
        return "number"
    if "list" in name or "array" in name:
        return "array"
    if "dict" in name or "object" in name:
        return "object"
    return "string"


def _tool(value: str, ordinal: int) -> dict[str, Any]:
    document = _parse_document(value)
    name = str(document.get("name") or f"operation_{ordinal}")
    description = str(document.get("description") or f"Operation documented as {name}.")
    if len(description.strip()) < 12:
        description = f"Operation for {description.strip() or name}."
    raw_parameters = document.get("parameters") or {}
    if isinstance(raw_parameters, list):
        raw_parameters = {
            str(item.get("name") or f"argument_{index}"): item
            for index, item in enumerate(raw_parameters)
            if isinstance(item, Mapping)
        }
    properties: dict[str, Any] = {}
    required = []
    if isinstance(raw_parameters, Mapping):
        for parameter_name, definition in raw_parameters.items():
            definition = definition if isinstance(definition, Mapping) else {}
            type_text = str(definition.get("type") or "string")
            properties[str(parameter_name)] = {
                "type": _json_type(type_text),
                "description": str(definition.get("description") or "Argument value."),
            }
            if "optional" not in type_text.casefold() and "default" not in definition:
                required.append(str(parameter_name))
    lowered = json.dumps(document, ensure_ascii=False).casefold()
    modality = (
        "image" if "image" in lowered else "audio" if "audio" in lowered else "video" if "video" in lowered else "json"
    )
    fingerprint = _digest(document)
    return {
        "tool_id": f"op_{fingerprint[:20]}",
        "tool_family": "toolret_general",
        "description": description,
        "capabilities": [_token(name, f"operation_{ordinal}")],
        "input_modalities": [modality],
        "output_modalities": ["json"],
        "evidence_roles": ["action"],
        "side_effect_class": "none",
        "argument_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "constraints": ["none"],
        "prerequisites": ["none"],
    }


def _viewer_page(offset: int, length: int, timeout: float) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": VIEWER_DATASET,
            "config": VIEWER_CONFIG,
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    last_error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(
            f"{VIEWER_BASE}?{query}", headers={"User-Agent": "nomos-research/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            return [dict(item["row"]) for item in payload["rows"]]
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Dataset Viewer failed: {last_error}")


def _source_rows(count: int, timeout: float, cache_path: Path) -> list[dict[str, Any]]:
    if cache_path.exists():
        cached = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
        if len(cached) >= count:
            return cached[:count]
    from datasets import load_dataset

    dataset = load_dataset(
        VIEWER_DATASET,
        VIEWER_CONFIG,
        split="train",
        streaming=True,
    )
    rows = [dict(row) for row in dataset.take(count)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _skeleton(source: Mapping[str, Any], index: int, seed: int) -> dict[str, Any]:
    positives = [str(value) for value in source.get("positive") or []]
    negatives = [str(value) for value in source.get("negative") or []]
    if not positives or not negatives:
        raise ValueError(f"ToolRet row lacks positive/negative documents: {source.get('id')}")
    documents = [positives[0], *negatives[:15]]
    tools = []
    seen_ids = set()
    for ordinal, document in enumerate(documents):
        tool = _tool(document, ordinal)
        if tool["tool_id"] not in seen_ids:
            tools.append(tool)
            seen_ids.add(tool["tool_id"])
    if len(tools) < 4:
        raise ValueError(f"ToolRet row has too few unique candidates: {source.get('id')}")
    registry = ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": f"toolret_train_{index:07d}",
            "tools": tools,
        }
    )
    target_id = tools[0]["tool_id"]
    rng = random.Random(seed + index * 7919)
    legal_ids = [tool.tool_id for tool in registry.tools]
    rng.shuffle(legal_ids)
    recovery = index % 5 == 0
    previous_ids = []
    if recovery:
        previous_ids = [value for value in legal_ids if value != target_id][-3:]
        legal_ids = [value for value in legal_ids if value not in previous_ids]
    transition = ("none", "related_complete", "confusable_complete", "stale_intent", "failed_candidates")[index % 5]
    cell = {
        "matrix_version": MATRIX_VERSION,
        "source_row_hash": _digest(source.get("id")),
        "task_kind": "recover" if recovery else "route",
        "history_transition": transition,
        "candidate_pool_size": len(legal_ids),
        "language_surface": "deepseek",
        "ordinal": index,
    }
    return {
        "index": index,
        "source": dict(source),
        "registry": registry,
        "target_id": target_id,
        "legal_ids": legal_ids,
        "previous_ids": previous_ids,
        "cell": cell,
        "matrix_cell_id": _digest(cell),
    }


def _assignment(skeleton: Mapping[str, Any]) -> dict[str, Any]:
    source = skeleton["source"]
    registry: ToolRegistry = skeleton["registry"]
    positive = registry.require(str(skeleton["target_id"]))
    negatives = [
        tool.description
        for tool in registry.resolve(skeleton["legal_ids"])
        if tool.tool_id != skeleton["target_id"]
    ][:3]
    return {
        "assignment_id": source["id"],
        "original_request": source["query"],
        "routing_goal": source.get("prompt"),
        "needed_operation": positive.description,
        "confusable_operations": negatives,
        "task_kind": skeleton["cell"]["task_kind"],
        "history_transition": skeleton["cell"]["history_transition"],
    }


def _prompt(assignments: list[dict[str, Any]]) -> str:
    return f"""Create stateful, project-agnostic tool-routing requests from these assignments.

For each assignment, preserve the user's concrete intent and constraints, but use fresh wording.
Never mention a tool name, tool ID, registry, benchmark, hidden label, or capability identifier.
The current_step must uniquely favor needed_operation over confusable_operations.
For recover rows, say earlier candidates failed and ask for a different option.
completed_steps must contain 0-3 short observable past actions and must not state the answer.

Return one JSON object with an items array of exactly {len(assignments)} objects:
{{"assignment_id":"...","question":"...","current_step":"...","completed_steps":["..."]}}
Question and current_step must each be 20-400 characters. Strict JSON only, no Markdown.

Assignments:
{json.dumps(assignments, ensure_ascii=False, sort_keys=True)}"""


def _request_batch(
    assignments: list[dict[str, Any]], *, api_key: str, model: str, timeout: float, retries: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Generate rigorous synthetic routing data as JSON."},
            {"role": "user", "content": _prompt(assignments)},
        ],
        "max_tokens": 5000,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    last_error = "unknown error"
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            content = json.loads(body["choices"][0]["message"]["content"])
            items = content.get("items")
            if not isinstance(items, list) or len(items) != len(assignments):
                raise ValueError("wrong item count")
            by_id = {str(item.get("assignment_id")): item for item in items if isinstance(item, dict)}
            ordered = [by_id[str(item["assignment_id"])] for item in assignments]
            usage = body.get("usage") or {}
            return ordered, {key: int(usage.get(key) or 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(last_error)


def _materialize(
    skeleton: Mapping[str, Any], generated: Mapping[str, Any], *, index: int, seed: int, model: str
) -> dict[str, Any]:
    registry: ToolRegistry = skeleton["registry"]
    question = str(generated.get("question") or "").strip()
    current_step = str(generated.get("current_step") or "").strip()
    completed = generated.get("completed_steps") or []
    fallback_used = False
    if not 8 <= len(question) <= 400:
        question = str(skeleton["source"].get("query") or "Route the current request.")[:400]
        fallback_used = True
    if not 8 <= len(current_step) <= 400:
        current_step = str(
            skeleton["source"].get("prompt")
            or registry.require(str(skeleton["target_id"])).description
        )[:400]
        fallback_used = True
    if not isinstance(completed, list):
        completed = []
        fallback_used = True
    completed = [value for value in completed if isinstance(value, str) and value.strip()]
    forbidden = [registry.registry_id, *registry.by_id]
    combined = f"{question}\n{current_step}".casefold()
    if PROJECT_MARKER.search(combined) or any(value.casefold() in combined for value in forbidden):
        raise ValueError(f"forbidden marker in teacher text at {index}")
    legal_ids = list(skeleton["legal_ids"])
    target_id = str(skeleton["target_id"])
    previous_ids = list(skeleton["previous_ids"])
    cell = dict(skeleton["cell"])
    cell_id = str(skeleton["matrix_cell_id"])
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    row = {
        "_source_query": str(skeleton["source"].get("query") or ""),
        "schema_version": "decision-state.v2",
        "dataset_version": DATASET_VERSION,
        "decision_state_id": f"toolret-agentic-{seed}-{index:07d}",
        "trajectory_id": f"toolret-agentic-trajectory-{seed}-{index:07d}",
        "scenario_id": f"toolret-agentic-scenario-{seed}-{index:07d}",
        "step": len(completed),
        "question": question,
        "task_kind": cell["task_kind"],
        "agent_state": {"state_name": "active", "phase": "execution"},
        "history": [{"completed_step": value, "status": "complete"} for value in completed[:3]],
        "plan": {"remaining_step": current_step},
        "observed_evidence": [],
        "governance": {
            "allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
            "call_allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
        },
        "resource_state": {"remaining_steps": 8},
        "source_state": {"source_ids": [], "available_modalities": ["json", "image", "audio", "video"], "inventory_state": "known", "inspection_state": "partial", "schema_known": True},
        "query_state": {"query_terms": current_step.casefold().split()[:20], "schema_known": True},
        "previous_candidate_ids": previous_ids,
        "expansion_context": {
            "expansion_allowed": bool(previous_ids),
            "expansion_round": 1 if previous_ids else 0,
            "trigger": "wrong_tool" if previous_ids else "none",
            "prior_candidate_ids": previous_ids,
            "excluded_candidate_ids": previous_ids,
            "unresolved_requirement": current_step,
        },
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": legal_ids,
        "label": {
            "acceptable_tools": [target_id],
            "ranked_tools": [target_id, *[value for value in legal_ids if value != target_id]],
            "hard_negative_tools": [value for value in legal_ids if value != target_id],
            "label_source": VALIDATOR_VERSION,
        },
        "accepted": True,
        "evaluation_partition": "train",
        "split_group_id": f"toolret-agentic-scenario-{seed}-{index:07d}",
        "question_template_id": "deepseek-stateful-rewrite",
        "matrix_cell": cell,
        "matrix_cell_id": cell_id,
        "teacher_paraphrase": current_step,
        "provenance": {
            "corpus": DATASET_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "teacher": "deepseek",
            "artifact": "DeepSeek-api",
            "generated_at": now,
            "seed": seed + index * 7919,
            "validator_version": VALIDATOR_VERSION,
            "feature_version": FEATURE_VERSION,
            "registry_fingerprint": registry.fingerprint,
            "trajectory_hash": _digest({"source": skeleton["source"].get("id"), "generated": generated}),
            "matrix_cell_id": cell_id,
            "source_row_hash": _digest(skeleton["source"]),
            "teacher_fallback_used": fallback_used,
        },
    }
    report = validate_decision_state_v2(row)
    if not report.valid:
        raise ValueError(report.as_dict())
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--exclude-input", type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("data/raw/toolret-training-sample-4096.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("FITZ_TOOL_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("set FITZ_TOOL_DEEPSEEK_API_KEY")
    if min(args.count, args.batch_size, args.concurrency) < 1:
        raise SystemExit("count, batch size, and concurrency must be positive")
    started = time.perf_counter()
    excluded_hashes: set[str] = set()
    if args.exclude_input:
        with args.exclude_input.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    excluded_hashes.add(str((row.get("provenance") or {}).get("source_row_hash")))
    source_pool = _source_rows(
        args.count + len(excluded_hashes), args.timeout, args.source_cache
    )
    sources = [source for source in source_pool if _digest(source) not in excluded_hashes][
        : args.count
    ]
    if len(sources) != args.count:
        raise RuntimeError(f"found only {len(sources)} new ToolRet rows; expected {args.count}")
    skeletons = [
        _skeleton(source, args.start_index + offset, args.seed)
        for offset, source in enumerate(sources)
    ]
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    generated_rows = []
    completed_ids: set[str] = set()
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                generated_rows.append(row)
                completed_ids.add(str(row["decision_state_id"]))
    pending_skeletons = [
        value
        for value in skeletons
        if f"toolret-agentic-{args.seed}-{int(value['index']):07d}" not in completed_ids
    ]
    batches = [
        pending_skeletons[index : index + args.batch_size]
        for index in range(0, len(pending_skeletons), args.batch_size)
    ]
    usage: Counter[str] = Counter()
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as partial_handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    _request_batch,
                    [_assignment(value) for value in batch],
                    api_key=api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                ): batch
                for batch in batches
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                batch = futures[future]
                generated, request_usage = future.result()
                usage.update(request_usage)
                new_rows = [
                    _materialize(
                        value,
                        item,
                        index=int(value["index"]),
                        seed=args.seed,
                        model=args.model,
                    )
                    for value, item in zip(batch, generated)
                ]
                generated_rows.extend(new_rows)
                for row in new_rows:
                    partial_handle.write(json.dumps(row, sort_keys=True) + "\n")
                partial_handle.flush()
                if completed % 32 == 0:
                    print(f"completed_requests={completed}/{len(batches)} rows={len(generated_rows)}", flush=True)
    generated_rows.sort(key=lambda row: row["decision_state_id"])
    seen_questions: set[str] = set()
    repaired_duplicates = 0
    dropped_duplicates = 0
    unique_rows = []
    for row in generated_rows:
        normalized = row["question"].casefold().strip()
        if normalized in seen_questions:
            row["question"] = (
                f"{row['question']} Specific request: {row['_source_query']}"
            )[:400]
            row["query_state"]["query_terms"] = row["question"].casefold().split()[:20]
            row["provenance"]["trajectory_hash"] = _digest(
                {
                    "prior_hash": row["provenance"]["trajectory_hash"],
                    "repaired_question": row["question"],
                }
            )
            repaired_duplicates += 1
            normalized = row["question"].casefold().strip()
        if normalized in seen_questions:
            dropped_duplicates += 1
            continue
        seen_questions.add(normalized)
        row.pop("_source_query", None)
        unique_rows.append(row)
    generated_rows = unique_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in generated_rows), encoding="utf-8")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "matrix_version": MATRIX_VERSION,
        "count": len(generated_rows),
        "start_index": args.start_index,
        "excluded_source_rows": len(excluded_hashes),
        "unique_matrix_cells": len({row["matrix_cell_id"] for row in generated_rows}),
        "unique_registry_fingerprints": len({row["provenance"]["registry_fingerprint"] for row in generated_rows}),
        "source": VIEWER_DATASET,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "usage": dict(usage),
        "teacher_fallback_rows": sum(
            bool(row["provenance"]["teacher_fallback_used"]) for row in generated_rows
        ),
        "repaired_duplicate_questions": repaired_duplicates,
        "dropped_duplicate_questions": dropped_duplicates,
        "resumed_rows": len(completed_ids),
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
