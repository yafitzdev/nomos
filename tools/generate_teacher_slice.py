"""Generate matrix-bound scenario candidates through an OpenAI-compatible teacher."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.contracts import validate_scenario, validate_source_card
from fitz_tool.matrix import MatrixCell, _target_pool, materialize_cells
from fitz_tool.uniqueness import annotate_signatures


DEFAULT_NINFER_BASE_URL = os.environ.get(
    "FITZ_TOOL_NINFER_BASE_URL",
    os.environ.get("FITZ_TOOL_TEACHER_BASE_URL", "http://127.0.0.1:19003/v1"),
)
DEFAULT_NINFER_MODEL = os.environ.get(
    "FITZ_TOOL_NINFER_MODEL",
    os.environ.get("FITZ_TOOL_TEACHER_MODEL", "qwen3.8-27b-nvfp4"),
)
DEFAULT_DEEPSEEK_BASE_URL = os.environ.get(
    "FITZ_TOOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
)
DEFAULT_DEEPSEEK_MODEL = os.environ.get("FITZ_TOOL_DEEPSEEK_MODEL", "deepseek-chat")
PROMPT_VERSION = "teacher-prompt.v1"
# A single positive source card cannot faithfully represent an absent fact.
# Negative/absence cases require a multi-source corpus or an explicit negative
# source manifest, so keep them out of this first text-only pilot.
TEXT_SINGLE_CARD_TOPOLOGIES = {"one_passage", "multiple_passages"}
TEXT_SINGLE_CARD_STATES = {
    "initial",
    "no_hits",
    "noisy_hits",
    "partial_evidence",
    "expansion_needed",
    "insufficient",
    "fresh_sufficient",
}
TEXT_SINGLE_CARD_TERMINALS = {
    "ongoing",
    "selection",
    "abstention",
    "clarification",
    "step_limit_termination",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--teacher", choices=("ninfer", "deepseek"), required=True)
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint; defaults by teacher (NInfer or DeepSeek).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Teacher model ID; defaults by teacher (NInfer or DeepSeek).",
    )
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-slice",
        action="append",
        type=Path,
        default=[],
        help="Previously accepted scenario JSONL; reserve its matrix/type/instance signatures.",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-api-key", action="store_true")
    return parser


def _api_key(*, disabled: bool, teacher: str) -> str | None:
    if disabled:
        return None
    if teacher == "deepseek":
        key = os.environ.get("FITZ_TOOL_DEEPSEEK_API_KEY") or os.environ.get(
            "FITZ_AGENT_DEEPSEEK_API_KEY"
        )
    else:
        key = os.environ.get("FITZ_TOOL_TEACHER_API_KEY") or os.environ.get(
            "FITZ_AGENT_TEACHER_API_KEY"
        )
    if key:
        return key
    return getpass.getpass(f"{teacher} teacher API key: ")


def _excluded_signatures(paths: list[Path]) -> tuple[set[str], set[str], set[str]]:
    cell_ids: set[str] = set()
    type_signatures: set[str] = set()
    instance_signatures: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid excluded slice {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"invalid excluded slice {path}:{line_number}: row is not an object")
            cell = row.get("matrix_cell")
            if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str):
                cell_ids.add(cell["cell_id"])
            for field, target in (
                ("type_signature", type_signatures),
                ("instance_signature", instance_signatures),
            ):
                value = row.get(field)
                if isinstance(value, str):
                    target.add(value)
    return cell_ids, type_signatures, instance_signatures


def _prompt(cells: list[MatrixCell], source_card: dict[str, Any]) -> str:
    facts = "\n".join(
        f"- {fact['fact_id']}: {fact['statement']}"
        for fact in source_card["facts"]
    )
    cell_payload = []
    for cell in cells:
        values = cell.values
        cell_payload.append(
            {
                **cell.as_dict(),
                "legal_tool_pool": _target_pool(
                    values["agent_state"],
                    values["information_operation"],
                    values["source_modality"],
                    values["terminal_condition"],
                ),
            }
        )
    return f"""Create exactly {len(cells)} distinct technical integration-research testcase objects.

The objects are matrix-bound. Do not combine cells, omit cells, or create a second
object for a cell. Every question must be answerable only from the source-card facts.
Do not invent facts, tools, document identifiers, or terminal outcomes.
Keep questions concise and retrieval-friendly: use distinctive source terms
such as identifiers, product names, error codes, and nouns from the selected
facts. Do not add unsupported qualifiers such as "latest", "current", or
"authoritative" unless the assigned facts support them.

Source card: {source_card['source_id']} ({source_card['title']})
Facts:
{facts}

Assigned matrix cells:
{json.dumps(cell_payload, ensure_ascii=False, sort_keys=True)}

Return one JSON array with exactly {len(cells)} objects in the same order as the
assigned cells. Each object must contain:
- question: a concrete user question
- difficult_paraphrase: a materially different wording of the same question
- expected_fact_ids: fact IDs from the source card. Return an empty list only when
  the assigned topology is absent or the terminal condition is abstention or clarification.
- expected_tools: one or more tools from that cell's legal_tool_pool, including
  that cell's next_tool_target
- expected_terminal_state: exactly that cell's terminal_condition

The legal_tool_pool is authoritative for each cell. Do not copy a tool from
another cell. In particular, never use PDF, table, or code tools unless that
cell's legal_tool_pool contains them.

Use only these tool names:
set_retrieval_plan, search_bm25, grep_search, search_metadata, list_sources,
list_tabular_sources, inspect_table_schema, search_table_rows, list_pdf_sources,
inspect_pdf_structure, search_pdf_pages, read_file, inspect_code, inspect_evidence,
expand_context, compare_evidence, update_requirement_progress, assess_evidence,
finalize_document_selection.

Return JSON only. No Markdown and no explanation."""


def _decode_json_content(content: str) -> Any:
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()
    return json.loads(content)


def _request(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    cells: list[MatrixCell],
    source_card: dict[str, Any],
    timeout: float,
    retries: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], str | None]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You generate grounded synthetic training data as strict JSON.",
            },
            {"role": "user", "content": _prompt(cells, source_card)},
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

    last_error = "unknown teacher error"
    for _attempt in range(retries + 1):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            content = str((body["choices"][0].get("message") or {}).get("content") or "")
            parsed = _decode_json_content(content)
            if not isinstance(parsed, list) or len(parsed) != len(cells):
                raise ValueError(f"teacher returned {type(parsed).__name__} with wrong item count")
            if not all(isinstance(item, dict) for item in parsed):
                raise ValueError("teacher array contains a non-object item")
            return parsed, None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTPError {exc.code}: {detail}"
            else:
                last_error = f"{type(exc).__name__}: {exc}"
    return [], last_error


def _scenario(
    *,
    ordinal: int,
    teacher: str,
    model: str,
    seed: int,
    cell: MatrixCell,
    generated: dict[str, Any],
    source_card: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    source_id = source_card["source_id"]
    expected_fact_ids = generated.get("expected_fact_ids")
    expected_facts = [
        {"source_id": source_id, "fact_id": fact_id}
        for fact_id in expected_fact_ids
    ] if isinstance(expected_fact_ids, list) else []
    state_name = cell.values["agent_state"]
    history_by_state = {
        "initial": [],
        "no_hits": [{"tool": "search_bm25", "status": "ok", "result_count": 0}],
        "noisy_hits": [{"tool": "search_bm25", "status": "ok", "result_count": 5}],
        "partial_evidence": [{"tool": "search_bm25", "status": "ok", "result_count": 3}],
        "expansion_needed": [{"tool": "inspect_evidence", "status": "ok", "result_count": 1}],
        "insufficient": [{"tool": "search_bm25", "status": "ok", "result_count": 0}],
        "contradiction": [{"tool": "compare_evidence", "status": "ok", "result_count": 2}],
        "disputed": [{"tool": "compare_evidence", "status": "ok", "result_count": 2}],
        "fresh_sufficient": [{"tool": "assess_evidence", "status": "ok", "result_count": len(expected_facts)}],
    }
    state_setup = {
        "state_name": state_name,
        "history": history_by_state.get(state_name, []),
        "observed_evidence": [
            {
                "evidence_id": f"setup-E{index}",
                "source_id": source_id,
                "fact_ids": [fact["fact_id"]],
            }
            for index, fact in enumerate(expected_facts[:2], start=1)
        ],
        "requirements": [
            {
                "requirement_id": "R1",
                "status": "covered" if expected_facts else "missing",
            }
        ],
        "governance": {
            "assessment_fresh": state_name == "fresh_sufficient",
            "path": cell.values["governance_path"],
        },
    }
    scenario = {
        "schema_version": "scenario.v1",
        "scenario_id": f"{teacher}-{seed}-{ordinal:06d}",
        "matrix_version": "matrix.v1",
        "matrix_cell": cell.as_dict(),
        "source_card_ids": [source_id],
        "state_setup": state_setup,
        "question": generated.get("question", ""),
        "difficult_paraphrase": generated.get("difficult_paraphrase", ""),
        "expected_facts": expected_facts,
        "expected_tools": generated.get("expected_tools", []),
        "expected_terminal_state": generated.get("expected_terminal_state", ""),
        "provenance": {
            "teacher": teacher,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "seed": seed,
            "source_card_hashes": [
                source_card["content_sha256"],
                *([source_card["normalized_content_sha256"]]
                  if source_card.get("normalized_content_sha256")
                  else []),
            ],
            "generated_at": generated_at,
        },
    }
    return annotate_signatures(scenario)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1 or args.batch_size < 1 or args.concurrency < 1:
        raise SystemExit("count, batch-size, and concurrency must be positive")
    if args.teacher == "ninfer":
        base_url = args.base_url or DEFAULT_NINFER_BASE_URL
        model = args.model or DEFAULT_NINFER_MODEL
    else:
        base_url = args.base_url or DEFAULT_DEEPSEEK_BASE_URL
        model = args.model or DEFAULT_DEEPSEEK_MODEL
    source_card = json.loads(args.source_card.read_text(encoding="utf-8"))
    source_report = validate_source_card(source_card)
    if not source_report.valid:
        raise SystemExit(json.dumps(source_report.as_dict(), indent=2))
    excluded_cell_ids, excluded_types, excluded_instances = _excluded_signatures(args.exclude_slice)
    allowed_modalities = (
        None
        if source_card["modality"] == "mixed"
        else {source_card["modality"]}
    )
    cells = materialize_cells(
        args.count,
        seed=args.seed,
        excluded_cell_ids=excluded_cell_ids,
        allowed_source_modalities=allowed_modalities,
        allowed_evidence_topologies=(
            TEXT_SINGLE_CARD_TOPOLOGIES if source_card["modality"] == "text" else None
        ),
        allowed_agent_states=(TEXT_SINGLE_CARD_STATES if source_card["modality"] == "text" else None),
        allowed_terminal_conditions=(
            TEXT_SINGLE_CARD_TERMINALS if source_card["modality"] == "text" else None
        ),
    )
    batches = [cells[start : start + args.batch_size] for start in range(0, len(cells), args.batch_size)]
    key = _api_key(disabled=args.no_api_key, teacher=args.teacher)
    results: dict[int, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _request,
                base_url=base_url,
                model=model,
                api_key=key,
                cells=batch,
                source_card=source_card,
                timeout=args.timeout,
                retries=args.retries,
                max_tokens=args.max_tokens,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            generated, error = future.result()
            if error:
                errors.append({"batch_index": batch_index, "error": error})
            else:
                results[batch_index] = generated

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    scenarios: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_types = set(excluded_types)
    seen_instances = set(excluded_instances)
    for batch_index, batch in enumerate(batches):
        generated = results.get(batch_index)
        if generated is None:
            continue
        start = batch_index * args.batch_size
        for offset, item in enumerate(generated):
            scenario = _scenario(
                ordinal=start + offset,
                teacher=args.teacher,
                model=model,
                seed=args.seed,
                cell=batch[offset],
                generated=item,
                source_card=source_card,
                generated_at=generated_at,
            )
            report = validate_scenario(scenario)
            fact_ids = {
                fact.get("fact_id")
                for fact in source_card.get("facts", [])
                if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
            }
            if (
                source_card.get("modality") != "mixed"
                and scenario["matrix_cell"].get("source_modality") != source_card.get("modality")
            ):
                report.add("matrix_cell.source_modality", "does not match the source card modality")
            if scenario["type_signature"] in seen_types:
                report.add("type_signature", "duplicate type signature in excluded or current slices")
            if scenario["instance_signature"] in seen_instances:
                report.add("instance_signature", "duplicate instance signature in excluded or current slices")
            for fact in scenario.get("expected_facts", []):
                if fact.get("fact_id") not in fact_ids:
                    report.add("expected_facts", f"unknown source-card fact: {fact.get('fact_id')}")
            if report.valid:
                scenarios.append(scenario)
                seen_types.add(scenario["type_signature"])
                seen_instances.add(scenario["instance_signature"])
            else:
                rejected.append({"ordinal": start + offset, "issues": report.as_dict()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    if errors or rejected:
        error_path = args.output.with_suffix(".errors.jsonl")
        error_path.write_text(
            "".join(
                json.dumps(error, sort_keys=True) + "\n"
                for error in [*errors, *rejected]
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "requested": args.count,
                "teacher_returned": sum(len(batch) for batch in results.values()),
                "accepted": len(scenarios),
                "rejected": len(rejected),
                "failed_batches": len(errors),
                "excluded_cell_ids": len(excluded_cell_ids),
                "reserved_type_signatures": len(excluded_types),
                "reserved_instance_signatures": len(excluded_instances),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if len(scenarios) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
