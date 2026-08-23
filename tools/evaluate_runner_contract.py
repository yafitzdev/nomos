"""Evaluate router choices inside a multi-step external-registry execution loop.

The router is invoked through ``tools.run_router_contract`` as a separate
process. Tool execution is deterministic and synthetic: a correct capability
returns one new evidence item and advances the task, while a wrong capability
terminates that task. This is a contract and state-transition test, not a
replacement for real API execution.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.external_registry_fixtures import (
    EXTERNAL_REGISTRY_STYLES,
    EXTERNAL_STYLE_TOOL_IDS,
    TARGET_CAPABILITIES,
    build_external_registry,
)
from fitz_tool.generic_contracts import validate_runner_request_v2
from fitz_tool.tool_registry import ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/generated/nomos_generic_portability_100000.jsonl")
DEFAULT_MODELS = (
    "nomos_50k=artifacts/nomos_generic_ninfer_full.pt",
    "nomos_100k=artifacts/nomos_generic_portability_100000.pt",
)

RETRIEVAL_CAPABILITIES = {
    "plan_retrieval",
    "list_sources",
    "search_content",
    "exact_pattern_search",
    "search_metadata",
    "inspect_structured_schema",
    "search_structured_records",
    "inspect_document_structure",
    "search_document_pages",
    "read_content",
    "inspect_code_structure",
}

STAGE_PROMPTS = {
    "plan_retrieval": "choose the operation that plans the next retrieval step",
    "list_sources": "choose the operation that lists the currently available sources",
    "search_content": "choose the operation that searches the relevant source content",
    "exact_pattern_search": "choose the operation that finds the exact requested identifier or phrase",
    "search_metadata": "choose the operation that filters source metadata and catalog fields",
    "inspect_structured_schema": "choose the operation that inspects structured fields and types",
    "search_structured_records": "choose the operation that retrieves matching structured records",
    "inspect_document_structure": "choose the operation that inspects document sections and structure",
    "search_document_pages": "choose the operation that finds the relevant document pages",
    "read_content": "choose the operation that reads the selected source content",
    "inspect_code_structure": "choose the operation that inspects code symbols and definitions",
    "inspect_evidence": "choose the operation that inspects the support and provenance of evidence",
    "compare_evidence": "choose the operation that compares competing evidence",
    "assess_evidence": "choose the operation that assesses whether the evidence is sufficient",
    "finalize_selection": "choose the operation that finalizes the supported selection",
    "expand_context": "choose the operation that expands incomplete surrounding context",
    "update_requirements": "choose the operation that updates requirement coverage",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        metavar="NAME=ARTIFACT",
        help="Router artifact specification; repeat for multiple models.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=sorted(EXTERNAL_REGISTRY_STYLES),
        default=list(EXTERNAL_REGISTRY_STYLES),
    )
    return parser


def _read_source_rows(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evaluation_cohort") == "heldout_questions":
                rows.append(row)
    if not rows:
        raise ValueError("input does not contain heldout_questions rows")
    random.Random(seed).shuffle(rows)
    return rows[: min(limit, len(rows))]


def _stage_capabilities(initial: str) -> list[str]:
    if initial not in TARGET_CAPABILITIES:
        raise ValueError(f"unsupported target capability: {initial}")
    if initial == "finalize_selection":
        return [initial]
    if initial in {"compare_evidence", "assess_evidence"}:
        return [initial, "finalize_selection"]
    if initial in RETRIEVAL_CAPABILITIES:
        return [initial, "inspect_evidence", "assess_evidence", "finalize_selection"]
    return [initial, "assess_evidence", "finalize_selection"]


def _target_id(style: str, capability: str) -> str:
    return EXTERNAL_STYLE_TOOL_IDS[style][TARGET_CAPABILITIES.index(capability)]


def _candidate_ids(
    registry: ToolRegistry,
    target_id: str,
    *,
    seed: int,
) -> list[str]:
    distractors = [tool.tool_id for tool in registry.tools if tool.tool_id != target_id]
    random.Random(seed).shuffle(distractors)
    candidates = [target_id, *distractors[:6]]
    random.Random(seed + 7919).shuffle(candidates)
    return candidates


def _state_name(capability: str, stage: int, original: Mapping[str, Any]) -> str:
    if stage == 0:
        return str((original.get("agent_state") or {}).get("state_name", "partial_evidence"))
    if capability == "finalize_selection":
        return "fresh_sufficient"
    return "partial_evidence"


def _phase(capability: str) -> str:
    if capability in {"assess_evidence", "finalize_selection"}:
        return "assessment"
    if capability in {"inspect_evidence", "compare_evidence"}:
        return "inspection"
    return "retrieval"


def _stage_request(
    original: Mapping[str, Any],
    registry: ToolRegistry,
    legal_ids: list[str],
    *,
    task_id: str,
    capability: str,
    stage: int,
    prior_capabilities: list[str],
) -> dict[str, Any]:
    original_question = str(original["question"])
    if stage == 0:
        question = original_question
    else:
        question = (
            f"Original objective: {original_question} After the preceding result, "
            f"{STAGE_PROMPTS[capability]}."
        )
    history = copy.deepcopy(original.get("history") or [])
    for previous in prior_capabilities:
        history.append({"action_family": "tool_use", "result": "strong", "capability": previous})
    evidence = copy.deepcopy(original.get("observed_evidence") or [])
    if stage:
        source_ids = list(original.get("source_card_ids") or ["external_source"])
        evidence.append(
            {
                "claim_count": 1,
                "evidence_id": f"execution-{task_id}-{stage}",
                "inspection_status": "inspected",
                "modality": "text",
                "source_id": str(source_ids[0]),
            }
        )
    governance = copy.deepcopy(original.get("governance") or {})
    governance["assessment_fresh"] = "assess_evidence" in prior_capabilities
    source_state = copy.deepcopy(original.get("source_state") or {})
    source_state["inspection_state"] = "full_context" if stage else source_state.get(
        "inspection_state", "partial"
    )
    query_state = copy.deepcopy(original.get("query_state") or {})
    query_state["operation"] = capability
    query_state["specificity"] = "schema_bound" if "structured" in capability else "semantic"
    resource_state = copy.deepcopy(original.get("resource_state") or {})
    resource_state["observed_evidence_count"] = len(evidence)
    resource_state["remaining_steps"] = max(0, len(_stage_capabilities(str((original.get("sampling_context") or {}).get("target_capability")))) - stage)
    request = {
        "schema_version": "runner-request.v2",
        "request_id": f"{task_id}-step-{stage}",
        "question": question,
        "agent_state": {
            "phase": _phase(capability),
            "question_length_band": (original.get("agent_state") or {}).get("question_length_band", "long"),
            "state_name": _state_name(capability, stage, original),
        },
        "history": history,
        "plan": {
            "active": capability != "finalize_selection",
            "operation": capability,
            "objective": original_question,
            "requirements": list(governance.get("requirements") or []),
        },
        "observed_evidence": evidence,
        "governance": governance,
        "resource_state": resource_state,
        "source_state": source_state,
        "query_state": query_state,
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": legal_ids,
    }
    report = validate_runner_request_v2(request)
    if not report.valid:
        raise ValueError(f"invalid generated request {request['request_id']}: {report.as_dict()}")
    return request


def _build_tasks(rows: list[dict[str, Any]], styles: Sequence[str], seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for style_index, style in enumerate(styles):
        registry = build_external_registry(style)
        for row_index, original in enumerate(rows):
            task_id = f"{style}-{row_index:04d}"
            initial = str((original.get("sampling_context") or {}).get("target_capability"))
            capabilities = _stage_capabilities(initial)
            steps: list[dict[str, Any]] = []
            prior: list[str] = []
            for stage, capability in enumerate(capabilities):
                target_id = _target_id(style, capability)
                legal_ids = _candidate_ids(
                    registry,
                    target_id,
                    seed=seed + style_index * 1000003 + row_index * 7919 + stage * 104729,
                )
                request = _stage_request(
                    original,
                    registry,
                    legal_ids,
                    task_id=task_id,
                    capability=capability,
                    stage=stage,
                    prior_capabilities=prior,
                )
                steps.append(
                    {
                        "request": request,
                        "capability": capability,
                        "target_id": target_id,
                        "legal_ids": legal_ids,
                    }
                )
                prior.append(capability)
            tasks.append(
                {
                    "task_id": task_id,
                    "style": style,
                    "initial_capability": initial,
                    "steps": steps,
                }
            )
    return tasks


def _parse_models(values: Sequence[str]) -> list[tuple[str, Path]]:
    raw_values = list(values) or list(DEFAULT_MODELS)
    models: list[tuple[str, Path]] = []
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"model must use NAME=ARTIFACT syntax: {raw}")
        name, artifact = raw.split("=", 1)
        if not name or not artifact:
            raise ValueError(f"model must use NAME=ARTIFACT syntax: {raw}")
        models.append((name, Path(artifact)))
    return models


def _run_router(
    tasks: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    artifact: Path | None,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    requests = [step["request"] for task in tasks for step in task["steps"]]
    payload = "".join(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n" for request in requests)
    command = [sys.executable, "-m", "tools.run_router_contract", "--mode", mode]
    if artifact is not None:
        command.extend(("--artifact", str(artifact)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"router contract process failed ({mode}): {completed.stderr[-4000:]}"
        )
    responses: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        response = json.loads(line)
        request_id = str(response.get("request_id", ""))
        if response.get("schema_version") != "router-response.v2":
            raise RuntimeError(f"invalid router response schema at line {line_number}")
        if response.get("error"):
            raise RuntimeError(f"router response error at line {line_number}: {response['error']}")
        if request_id in responses:
            raise RuntimeError(f"duplicate router response: {request_id}")
        ranked = response.get("ranked_tools")
        if not isinstance(ranked, list) or not ranked or response.get("selected_tool") != ranked[0].get("tool_id"):
            raise RuntimeError(f"invalid router response ranking at line {line_number}")
        responses[request_id] = response
    expected = {str(request["request_id"]) for request in requests}
    if set(responses) != expected:
        missing = sorted(expected - set(responses))[:5]
        extra = sorted(set(responses) - expected)[:5]
        raise RuntimeError(f"router response IDs do not match; missing={missing}, extra={extra}")
    return responses


def _metric_for_tasks(
    tasks: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    planned_steps = executed_steps = correct_steps = executed_correct_steps = 0
    invalid = executed_invalid = failed_tasks = successful_tasks = 0
    failure_stages: Counter[str] = Counter()
    by_style: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_capability: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    traces: list[dict[str, Any]] = []
    for task in tasks:
        task_success = True
        task_executed = 0
        events: list[dict[str, Any]] = []
        for stage, step in enumerate(task["steps"]):
            planned_steps += 1
            request = step["request"]
            response = responses[str(request["request_id"])]
            legal = set(step["legal_ids"])
            selected = str(response["selected_tool"])
            ranked_ids = [str(item["tool_id"]) for item in response["ranked_tools"]]
            is_invalid = selected not in legal or not set(ranked_ids).issubset(legal)
            invalid += int(is_invalid)
            correct = selected == step["target_id"]
            correct_steps += int(correct)
            executed = task_success
            executed_steps += int(executed)
            task_executed += int(executed)
            executed_correct_steps += int(executed and correct)
            executed_invalid += int(executed and is_invalid)
            event = {
                "step": stage,
                "request_id": request["request_id"],
                "legal_tools": step["legal_ids"],
                "selected_tool": selected,
                "expected_tool": step["target_id"],
                "capability": step["capability"],
                "rank": next(
                    (index for index, tool_id in enumerate(ranked_ids, start=1) if tool_id == step["target_id"]),
                    None,
                ),
                "execution": {
                    "executed": executed,
                    "status": "ok" if executed and correct else "failed" if executed else "not_executed",
                    "result": "evidence_added" if executed and correct else "unexpected_capability" if executed else "task_already_failed",
                },
            }
            events.append(event)
            if not correct and task_success:
                task_success = False
                failed_tasks += 1
                failure_stages[str(stage)] += 1
        if task_success:
            successful_tasks += 1
        traces.append(
            {
                "task_id": task["task_id"],
                "style": task["style"],
                "initial_capability": task["initial_capability"],
                "planned_steps": len(task["steps"]),
                "executed_steps": task_executed,
                "success": task_success,
                "events": events,
            }
        )
        by_style[str(task["style"])].append(traces[-1])
        by_capability[str(task["initial_capability"])].append(traces[-1])

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        item_count = len(items)
        item_planned = sum(int(item["planned_steps"]) for item in items)
        item_executed = sum(int(item["executed_steps"]) for item in items)
        item_successes = sum(bool(item["success"]) for item in items)
        item_correct = sum(
            event["selected_tool"] == event["expected_tool"]
            for item in items
            for event in item["events"]
        )
        item_executed_correct = sum(
            event["execution"]["executed"] and event["selected_tool"] == event["expected_tool"]
            for item in items
            for event in item["events"]
        )
        item_invalid = sum(
            event["selected_tool"] not in set(event["legal_tools"])
            for item in items
            for event in item["events"]
        )
        item_executed_invalid = sum(
            event["execution"]["executed"]
            and event["selected_tool"] not in set(event["legal_tools"])
            for item in items
            for event in item["events"]
        )
        return {
            "tasks": item_count,
            "planned_steps": item_planned,
            "executed_steps": item_executed,
            "task_success_rate": item_successes / item_count if item_count else 0.0,
            "step_accuracy": item_correct / item_planned if item_planned else 0.0,
            "executed_step_accuracy": item_executed_correct / item_executed if item_executed else 0.0,
            "mean_executed_steps": item_executed / item_count if item_count else 0.0,
            "invalid_selection_rate": item_invalid / item_planned if item_planned else 0.0,
            "executed_invalid_selection_rate": item_executed_invalid / item_executed if item_executed else 0.0,
        }

    return {
        **summarize(traces),
        "failed_tasks": failed_tasks,
        "successful_tasks": successful_tasks,
        "executed_step_accuracy": executed_correct_steps / executed_steps if executed_steps else 0.0,
        "executed_invalid_selection_rate": executed_invalid / executed_steps if executed_steps else 0.0,
        "failure_stage_counts": dict(sorted(failure_stages.items())),
        "by_style": {key: summarize(value) for key, value in sorted(by_style.items())},
        "by_initial_capability": {
            key: summarize(value) for key, value in sorted(by_capability.items())
        },
        "traces": traces,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    models = _parse_models(args.models)
    rows = _read_source_rows(args.input, args.limit, args.seed)
    tasks = _build_tasks(rows, args.styles, args.seed)
    report: dict[str, Any] = {
        "schema_version": "runner-evaluation.v1",
        "source_cohort": "heldout_questions",
        "source_rows": len(rows),
        "tasks": len(tasks),
        "styles": list(args.styles),
        "contract": {
            "request": "runner-request.v2",
            "response": "router-response.v2",
            "execution": "deterministic capability simulator",
        },
        "models": {},
    }
    trace_records: dict[str, list[dict[str, Any]]] = {}
    baseline_responses = _run_router(tasks, mode="candidate_order", artifact=None, timeout=args.timeout)
    baseline_metrics = _metric_for_tasks(tasks, baseline_responses)
    trace_records["candidate_order"] = baseline_metrics.pop("traces")
    report["models"]["candidate_order"] = baseline_metrics
    for name, artifact in models:
        responses = _run_router(tasks, mode="model", artifact=artifact, timeout=args.timeout)
        metrics = _metric_for_tasks(tasks, responses)
        trace_records[name] = metrics.pop("traces")
        report["models"][name] = {"artifact": str(artifact), **metrics}
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_output.open("w", encoding="utf-8", newline="\n") as handle:
            for model_name, traces in trace_records.items():
                for trace in traces:
                    handle.write(
                        json.dumps(
                            {"model": model_name, **trace}, ensure_ascii=False, sort_keys=True
                        )
                        + "\n"
                    )
        report["trace_output"] = str(args.trace_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
