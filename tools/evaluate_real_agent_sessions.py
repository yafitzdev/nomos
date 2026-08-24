"""Run weak/strong agents with and without Nomos over executable local tools."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from fitz_tool.call_validation import validate_tool_call
from fitz_tool.coprocessor import validation_repair_guidance
from fitz_tool.dense_selector import DenseToolRanker
from fitz_tool.external_registry_fixtures import (
    EXTERNAL_REGISTRY_STYLES,
    build_external_registry,
)
from fitz_tool.final_holdout_fixtures import (
    FINAL_HOLDOUT_VERSION,
    FINAL_REGISTRY_STYLES,
    canonical_capability as final_canonical_capability,
    build_final_registry,
)
from fitz_tool.local_llm import OnnxGenAIChat
from fitz_tool.promotion_holdout_fixtures import (
    PROMOTION_HOLDOUT_VERSION,
    PROMOTION_REGISTRY_STYLES,
    canonical_capability as promotion_canonical_capability,
    build_promotion_registry,
)
from fitz_tool.tool_registry import SIDE_EFFECT_CLASSES, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "agent_sessions"

WORKFLOWS = (
    {
        "name": "exact_identifier",
        "question": "Find the exact stable webhook event identifier and finish with a supported selection.",
        "stages": ("exact_pattern_search", "inspect_evidence", "finalize_selection"),
    },
    {
        "name": "structured_records",
        "question": "Determine which event is retrying using the structured data, then finalize.",
        "stages": (
            "inspect_structured_schema",
            "search_structured_records",
            "inspect_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "document_navigation",
        "question": "Locate the webhook retry schedule in the documentation and finalize the result.",
        "stages": (
            "inspect_document_structure",
            "search_document_pages",
            "inspect_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "code_inspection",
        "question": "Use the client implementation to determine its retry delays and finalize.",
        "stages": (
            "inspect_code_structure",
            "read_content",
            "inspect_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "evidence_governance",
        "question": "Check the stable API version and decide whether the evidence is sufficient.",
        "stages": (
            "search_content",
            "inspect_evidence",
            "update_requirements",
            "assess_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "source_discovery",
        "question": "Find which local source defines required event fields and finalize.",
        "stages": ("list_sources", "search_metadata", "read_content", "finalize_selection"),
    },
    {
        "name": "context_recovery",
        "question": "Recover enough context around the retry policy to make a supported selection.",
        "stages": ("search_content", "expand_context", "inspect_evidence", "finalize_selection"),
    },
    {
        "name": "comparison",
        "question": "Compare the documented retry schedule with the client implementation and finalize.",
        "stages": (
            "search_content",
            "inspect_code_structure",
            "compare_evidence",
            "assess_evidence",
            "finalize_selection",
        ),
    },
)

# This suite was added only after model and retrieval architecture selection.
# Its wording and sequences are intentionally absent from training generators.
FINAL_WORKFLOWS = (
    {
        "name": "event_token_resolution",
        "question": "Resolve the canonical callback label verbatim and return only a supported result.",
        "stages": ("exact_pattern_search", "inspect_evidence", "finalize_selection"),
    },
    {
        "name": "retry_record_triage",
        "question": "Use the machine-readable event inventory to identify the entry still being retried.",
        "stages": (
            "inspect_structured_schema",
            "search_structured_records",
            "inspect_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "manual_location",
        "question": "Navigate the integration manual to recover the retry timing and support the answer.",
        "stages": (
            "inspect_document_structure",
            "search_document_pages",
            "read_content",
            "finalize_selection",
        ),
    },
    {
        "name": "implementation_policy",
        "question": "Determine the retry policy encoded by the client and reconcile it with available evidence.",
        "stages": (
            "inspect_code_structure",
            "read_content",
            "compare_evidence",
            "assess_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "source_attribution",
        "question": "Identify which local resource defines mandatory event fields and produce a supported choice.",
        "stages": (
            "list_sources",
            "search_metadata",
            "read_content",
            "inspect_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "evidence_repair",
        "question": "Repair an under-specified retry-policy observation until it is decision-ready.",
        "stages": (
            "search_content",
            "expand_context",
            "update_requirements",
            "assess_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "cross_source_reconciliation",
        "question": "Plan and reconcile the manual and implementation before committing a retry schedule.",
        "stages": (
            "plan_retrieval",
            "search_content",
            "inspect_code_structure",
            "compare_evidence",
            "inspect_evidence",
            "finalize_selection",
        ),
    },
    {
        "name": "governed_version_decision",
        "question": "Determine the stable API version, close its evidence obligations, and commit the result.",
        "stages": (
            "search_content",
            "inspect_evidence",
            "update_requirements",
            "assess_evidence",
            "finalize_selection",
        ),
    },
)

STAGE_REQUESTS = {
    "plan_retrieval": "Plan the information-gathering sequence.",
    "list_sources": "List the available local sources.",
    "search_content": "Search source content for the requested fact.",
    "exact_pattern_search": "Find the exact literal identifier.",
    "search_metadata": "Use source metadata to identify the right resource.",
    "inspect_structured_schema": "Inspect fields and types in the structured source.",
    "search_structured_records": "Retrieve the matching structured record.",
    "inspect_document_structure": "Inspect document sections and landmarks.",
    "search_document_pages": "Locate the relevant document section or range.",
    "read_content": "Read the selected source content.",
    "inspect_code_structure": "Inspect code symbols and definitions.",
    "inspect_evidence": "Inspect support and provenance for the candidate evidence.",
    "expand_context": "Expand the incomplete result with surrounding context.",
    "compare_evidence": "Compare the observations for agreement or conflict.",
    "update_requirements": "Update requirement coverage from inspected evidence.",
    "assess_evidence": "Assess whether the evidence is sufficient.",
    "finalize_selection": "Finalize the supported result.",
}

FINAL_STAGE_REQUESTS = {
    "plan_retrieval": "Work out the order of the remaining information operations.",
    "list_sources": "Expose the resource handles that can be used at this point.",
    "search_content": "Locate wording relevant to the unresolved fact.",
    "exact_pattern_search": "Match the requested literal value exactly.",
    "search_metadata": "Narrow the resource catalog by its descriptive attributes.",
    "inspect_structured_schema": "Establish the fields and types in the machine-readable input.",
    "search_structured_records": "Select the machine-readable entries that meet the condition.",
    "inspect_document_structure": "Map the manual's landmarks before opening a section.",
    "search_document_pages": "Find the manual range associated with the issue.",
    "read_content": "Open the chosen resource in full.",
    "inspect_code_structure": "Trace the implementation's definitions and symbols.",
    "inspect_evidence": "Check the observation's support and origin.",
    "expand_context": "Widen the incomplete observation with neighboring material.",
    "compare_evidence": "Reconcile the observations and surface any disagreement.",
    "update_requirements": "Mark decision conditions covered or still outstanding.",
    "assess_evidence": "Judge whether the accumulated support permits a conclusion.",
    "finalize_selection": "Commit the answer that is supported by the completed work.",
}

PROMOTION_STAGE_REQUESTS = {
    "plan_retrieval": "Determine the order of evidence operations still needed.",
    "list_sources": "Obtain the workspace's currently addressable resource references.",
    "search_content": "Discover material discussing the unresolved technical point.",
    "exact_pattern_search": "Locate the requested symbol as a verbatim occurrence.",
    "search_metadata": "Filter resources through their catalog attributes.",
    "inspect_structured_schema": "Learn the available structured fields and their types.",
    "search_structured_records": "Evaluate the condition and return qualifying structured entries.",
    "inspect_document_structure": "Build a navigation map of the document.",
    "search_document_pages": "Retrieve the document locations covering the subject.",
    "read_content": "Load the complete body of the selected resource.",
    "inspect_code_structure": "Enumerate the implementation definitions involved.",
    "inspect_evidence": "Audit the observation's traceability and support.",
    "expand_context": "Extend the clipped observation with adjacent source text.",
    "compare_evidence": "Reconcile the observations for consistency or conflict.",
    "update_requirements": "Synchronize covered and outstanding decision conditions.",
    "assess_evidence": "Determine whether support is adequate for a conclusion.",
    "finalize_selection": "Record the justified answer and close the decision.",
}

PROMOTION_WORKFLOWS = (
    {
        "name": "sdk_symbol_resolution",
        "question": "Find the SDK's canonical retry constant verbatim and support the resulting answer.",
        "stages": ("exact_pattern_search", "inspect_evidence", "finalize_selection"),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
    {
        "name": "payload_failure_analysis",
        "question": "Identify the failing payload entry from machine-readable integration telemetry.",
        "stages": (
            "inspect_structured_schema",
            "search_structured_records",
            "inspect_evidence",
            "finalize_selection",
        ),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
    {
        "name": "pdf_limit_resolution",
        "question": "Recover the request ceiling documented in the API manual and justify it.",
        "stages": (
            "inspect_document_structure",
            "search_document_pages",
            "read_content",
            "finalize_selection",
        ),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
    {
        "name": "source_code_default",
        "question": "Establish the retry default implemented by the client and verify its support.",
        "stages": (
            "inspect_code_structure",
            "read_content",
            "inspect_evidence",
            "assess_evidence",
            "finalize_selection",
        ),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
    {
        "name": "catalog_provenance",
        "question": "Determine which workspace artifact defines the required callback fields.",
        "stages": (
            "list_sources",
            "search_metadata",
            "read_content",
            "inspect_evidence",
            "finalize_selection",
        ),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
    {
        "name": "clipped_policy_recovery",
        "question": "Turn a clipped retry-policy observation into sufficient, traceable support.",
        "stages": (
            "search_content",
            "expand_context",
            "inspect_evidence",
            "assess_evidence",
            "finalize_selection",
        ),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
    {
        "name": "implementation_discrepancy",
        "question": "Reconcile the manual and client behavior before deciding the effective retry policy.",
        "stages": (
            "plan_retrieval",
            "search_content",
            "inspect_code_structure",
            "compare_evidence",
            "update_requirements",
            "assess_evidence",
            "finalize_selection",
        ),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
    {
        "name": "release_readiness",
        "question": "Resolve the stable-version requirement and close the evidence decision safely.",
        "stages": (
            "search_content",
            "inspect_evidence",
            "update_requirements",
            "assess_evidence",
            "finalize_selection",
        ),
        "stage_requests": PROMOTION_STAGE_REQUESTS,
    },
)


CONDITION_POLICIES = {
    # One-shot controls. Validation is used only as the scoring/execution gate;
    # the agent never sees repair feedback and cannot recover from a bad choice.
    "full_raw": {
        "use_selector": False,
        "top_k": None,
        "one_shot": True,
        "repair_feedback": False,
        "candidate_recovery": False,
    },
    "nomos_raw": {
        "use_selector": True,
        "top_k": 3,
        "one_shot": True,
        "repair_feedback": False,
        "candidate_recovery": False,
    },
    # Historical assisted controls are retained so prior reports reproduce.
    "full": {
        "use_selector": False,
        "top_k": None,
        "one_shot": False,
        "repair_feedback": True,
        "candidate_recovery": True,
    },
    "nomos": {
        "use_selector": True,
        "top_k": 3,
        "one_shot": False,
        "repair_feedback": True,
        "candidate_recovery": True,
    },
}


class ChatBackend(Protocol):
    def complete(
        self, messages: Sequence[Mapping[str, str]], *, max_new_tokens: int = 128
    ) -> dict[str, Any]: ...


class DeepSeekChat:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def complete(
        self, messages: Sequence[Mapping[str, str]], *, max_new_tokens: int = 128
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "stream": False,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read())
        usage = body.get("usage") or {}
        return {
            "text": str(body["choices"][0]["message"]["content"]),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }


class DenseSelector:
    def __init__(
        self,
        path: Path,
        *,
        strategy: str = "multiview",
        candidate_strategy: str = "multiview",
    ) -> None:
        self.ranker = DenseToolRanker.from_path(
            path,
            query_strategy=strategy,
            candidate_strategy=candidate_strategy,
        )

    def rank(
        self, request: Mapping[str, Any], registry: ToolRegistry, legal_ids: list[str]
    ) -> list[str]:
        del registry, legal_ids
        return [item["tool_id"] for item in self.ranker.rank(request)]


def _request(
    workflow: Mapping[str, Any],
    registry: ToolRegistry,
    legal_ids: list[str],
    *,
    stage: str,
    previous_ids: list[str],
    completed: list[str],
) -> dict[str, Any]:
    supplied_stage_requests = workflow.get("stage_requests")
    if isinstance(supplied_stage_requests, Mapping):
        stage_requests = supplied_stage_requests
    elif workflow.get("name") in {item["name"] for item in FINAL_WORKFLOWS}:
        stage_requests = FINAL_STAGE_REQUESTS
    else:
        stage_requests = STAGE_REQUESTS
    question = f"Objective: {workflow['question']} Current step: {stage_requests[stage]}"
    modalities = sorted(
        {value for tool in registry.tools for value in tool.input_modalities}
    )
    return {
        "schema_version": "runner-request.v2",
        "request_id": f"{workflow['name']}:{len(completed)}",
        "question": question,
        "task_kind": "recover" if previous_ids else "route",
        "agent_state": {"state_name": "active", "phase": "execution"},
        "history": [{"completed_step": stage_requests[value]} for value in completed],
        "plan": {"remaining_step": stage_requests[stage]},
        "observed_evidence": [
            {"result_id": f"result_{index}", "inspection_status": "inspected"}
            for index, _value in enumerate(completed)
        ],
        "governance": {
            "allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
            "call_allowed_side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
        },
        "resource_state": {"remaining_steps": 12 - len(completed)},
        "source_state": {
            "source_ids": [path.name for path in FIXTURES.iterdir() if path.is_file()],
            "available_modalities": modalities,
            "inventory_state": "known",
            "inspection_state": "full_context",
            "schema_known": True,
        },
        "query_state": {"query_terms": question.lower().split(), "schema_known": True},
        "previous_candidate_ids": list(previous_ids),
        "expansion_context": {
            "expansion_allowed": bool(previous_ids),
            "prior_candidate_ids": list(previous_ids),
            "excluded_candidate_ids": list(previous_ids),
            "trigger": "wrong_tool" if previous_ids else "none",
            "unresolved_requirement": stage_requests[stage],
        },
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": legal_ids,
    }


def _prompt(request: Mapping[str, Any], registry: ToolRegistry, visible_ids: list[str], feedback: str) -> str:
    tools = []
    for tool_id in visible_ids:
        tool = registry.require(tool_id)
        tools.append(
            {
                "tool_id": tool.tool_id,
                "description": tool.description,
                "arguments_schema": tool.argument_schema,
            }
        )
    return (
        f"{request['question']}\n"
        f"Completed steps: {json.dumps(request['history'])}\n"
        f"Previous failure: {feedback or 'none'}\n"
        "Choose exactly one visible tool. Return strict JSON only as "
        '{"tool_id":"...","arguments":{...}}. Arguments must satisfy its schema.\n'
        f"Visible tools: {json.dumps(tools, ensure_ascii=False, sort_keys=True)}"
    )


def _parse_call(text: str) -> dict[str, Any] | None:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _repair_feedback(guidance: Mapping[str, Any] | None, reasons: str) -> str:
    if not guidance or guidance.get("strategy") != "repair_same_tool_call":
        return reasons
    shape = json.dumps(guidance["call_shape"], ensure_ascii=False, sort_keys=True)
    allowed = json.dumps(guidance["allowed_argument_names"], ensure_ascii=False)
    return (
        f"{reasons}. Keep this tool if it is relevant, but repair the call. "
        f"Use exactly this JSON shape: {shape}. Replace placeholders with real values. "
        f"Allowed argument names: {allowed}; do not add other keys."
    )


def _execute(capability: str, expected: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if capability != expected:
        return {"ok": False, "error": f"{capability} did not satisfy {expected}"}
    files = sorted(path for path in FIXTURES.iterdir() if path.is_file())
    if capability == "list_sources":
        value: Any = [path.name for path in files]
    elif capability in {"search_content", "exact_pattern_search", "search_document_pages", "read_content", "expand_context"}:
        value = {
            path.name: path.read_text(encoding="utf-8")[:2000]
            for path in files
            if path.suffix in {".md", ".py"}
        }
    elif capability == "search_metadata":
        value = [{"name": path.name, "suffix": path.suffix, "bytes": path.stat().st_size} for path in files]
    elif capability == "inspect_structured_schema":
        with (FIXTURES / "events.csv").open(encoding="utf-8", newline="") as handle:
            value = next(csv.reader(handle))
    elif capability == "search_structured_records":
        with (FIXTURES / "events.csv").open(encoding="utf-8", newline="") as handle:
            value = [row for row in csv.DictReader(handle) if row["status"] == "retrying"]
    elif capability == "inspect_document_structure":
        value = [
            line.strip("# ")
            for line in (FIXTURES / "integration_guide.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("#")
        ]
    elif capability == "inspect_code_structure":
        tree = ast.parse((FIXTURES / "client.py").read_text(encoding="utf-8"))
        value = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    elif capability == "inspect_evidence":
        value = {"inspected": True, "provenance": "local_fixture"}
    elif capability == "compare_evidence":
        value = {"agreement": True, "retry_delays": [10, 30, 90]}
    elif capability == "update_requirements":
        value = {"coverage": "complete"}
    elif capability == "assess_evidence":
        value = {"sufficient": True}
    elif capability == "finalize_selection":
        value = {"selected": True}
    else:
        value = {"executed": capability, "arguments": dict(arguments)}
    return {"ok": True, "value": value}


def _canonical_capability(tool_id: str, suite: str, registry: ToolRegistry) -> str:
    if suite == "final":
        return final_canonical_capability(tool_id)
    if suite == "promotion":
        return promotion_canonical_capability(tool_id) or "irrelevant"
    return registry.require(tool_id).capabilities[0]


def _run_session(
    backend: ChatBackend,
    selector: DenseSelector | None,
    workflow: Mapping[str, Any],
    registry: ToolRegistry,
    *,
    condition: str,
    max_attempts: int,
    suite: str = "development",
) -> dict[str, Any]:
    policy = CONDITION_POLICIES[condition]
    attempt_limit = 1 if policy["one_shot"] else max_attempts
    completed: list[str] = []
    events = []
    totals = Counter()
    for stage in workflow["stages"]:
        rejected: list[str] = []
        feedback = ""
        stage_passed = False
        for attempt in range(attempt_limit):
            legal_ids = [tool.tool_id for tool in registry.tools if tool.tool_id not in rejected]
            request = _request(
                workflow,
                registry,
                legal_ids,
                stage=stage,
                previous_ids=rejected,
                completed=completed,
            )
            ranked = selector.rank(request, registry, legal_ids) if selector else legal_ids
            top_k = policy["top_k"]
            visible = ranked[:top_k] if isinstance(top_k, int) else legal_ids
            oracle_hit = any(
                _canonical_capability(tool_id, suite, registry) == stage
                for tool_id in visible
            )
            prompt = _prompt(request, registry, visible, feedback)
            result = backend.complete(
                [
                    {"role": "system", "content": "You are a precise tool-calling agent. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=128,
            )
            totals["prompt_tokens"] += int(result["prompt_tokens"])
            totals["completion_tokens"] += int(result["completion_tokens"])
            totals["visible_tools"] += len(visible)
            totals["available_tools"] += len(legal_ids)
            call = _parse_call(str(result["text"]))
            if call is None:
                error = "malformed JSON tool call"
                feedback = error if policy["repair_feedback"] else ""
                events.append(
                    {
                        "stage": stage,
                        "attempt": attempt + 1,
                        "valid": False,
                        "selection_correct": False,
                        "oracle_visible_hit": oracle_hit,
                        "error": error,
                        "visible_tool_ids": visible,
                    }
                )
                continue
            validation_state = {**request, "legal_candidate_ids": visible}
            validation = validate_tool_call(registry, validation_state, call)
            registry_tool_ids = {tool.tool_id for tool in registry.tools}
            selected_capability = (
                _canonical_capability(validation.tool_id, suite, registry)
                if validation.tool_id in registry_tool_ids
                else None
            )
            selection_correct = selected_capability == stage
            if not validation.valid:
                repair = (
                    validation_repair_guidance(registry, validation)
                    if policy["repair_feedback"]
                    else None
                )
                reasons = ", ".join(validation.failure_reasons)
                feedback = _repair_feedback(repair, reasons) if policy["repair_feedback"] else ""
                # A malformed call is a request to repair the arguments, not a
                # signal that the selected tool itself is irrelevant.  Keep a
                # repairable tool visible so the agent can retry it.  Only a
                # permanent tool-level rejection advances the candidate page.
                if (
                    policy["candidate_recovery"]
                    and not validation.repairable
                    and validation.tool_id in legal_ids
                ):
                    rejected.append(validation.tool_id)
                events.append(
                    {
                        "stage": stage,
                        "attempt": attempt + 1,
                        "tool_id": validation.tool_id,
                        "capability": selected_capability,
                        "valid": False,
                        "selection_correct": selection_correct,
                        "oracle_visible_hit": oracle_hit,
                        "error": feedback,
                        "repairable": validation.repairable,
                        "repair": repair,
                        "visible_tool_ids": visible,
                    }
                )
                continue
            tool = registry.require(validation.tool_id)
            capability = _canonical_capability(tool.tool_id, suite, registry)
            execution = _execute(capability, stage, call.get("arguments") or {})
            events.append(
                {
                    "stage": stage,
                    "attempt": attempt + 1,
                    "tool_id": tool.tool_id,
                    "capability": capability,
                    "valid": True,
                    "selection_correct": selection_correct,
                    "oracle_visible_hit": oracle_hit,
                    "execution": execution,
                    "visible_tool_count": len(visible),
                    "visible_tool_ids": visible,
                }
            )
            if execution["ok"]:
                completed.append(stage)
                stage_passed = True
                break
            if policy["candidate_recovery"] and tool.tool_id not in rejected:
                rejected.append(tool.tool_id)
            feedback = str(execution["error"]) if policy["repair_feedback"] else ""
        if not stage_passed:
            break
    return {
        "workflow": workflow["name"],
        "condition": condition,
        "registry": registry.registry_id,
        "success": len(completed) == len(workflow["stages"]),
        "completed_stages": len(completed),
        "required_stages": len(workflow["stages"]),
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "visible_tools": totals["visible_tools"],
        "available_tools": totals["available_tools"],
        "events": events,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("onnx", "deepseek"), required=True)
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--nomos-model", type=Path, required=True)
    parser.add_argument(
        "--selector-strategy", choices=("single", "multiview"), default="multiview"
    )
    parser.add_argument(
        "--candidate-strategy", choices=("single", "multiview"), default="multiview"
    )
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument(
        "--suite",
        choices=("development", "final", "promotion"),
        default="development",
        help="Use the development suite or the frozen post-selection holdout.",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        choices=tuple(
            str(workflow["name"])
            for workflow in (*WORKFLOWS, *FINAL_WORKFLOWS, *PROMOTION_WORKFLOWS)
        ),
        help="Optionally restrict execution to named workflows.",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--condition",
        choices=("full_raw", "nomos_raw", "full", "nomos", "both", "ablation"),
        default="both",
        help=(
            "raw conditions are one-shot with no repair or recovery; ablation runs "
            "full_raw, nomos_raw, and the complete Nomos coprocessor"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend == "onnx":
        if args.onnx_model is None:
            raise SystemExit("--onnx-model is required for the onnx backend")
        backend: ChatBackend = OnnxGenAIChat(args.onnx_model)
    else:
        api_key = os.environ.get("FITZ_TOOL_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("set FITZ_TOOL_DEEPSEEK_API_KEY for the DeepSeek backend")
        backend = DeepSeekChat(api_key, args.deepseek_model)
    selector = DenseSelector(
        args.nomos_model,
        strategy=args.selector_strategy,
        candidate_strategy=args.candidate_strategy,
    )
    if args.suite == "final":
        available_workflows = FINAL_WORKFLOWS
        registry_styles = FINAL_REGISTRY_STYLES
        registry_builder = build_final_registry
        suite_version = FINAL_HOLDOUT_VERSION
    elif args.suite == "promotion":
        available_workflows = PROMOTION_WORKFLOWS
        registry_styles = PROMOTION_REGISTRY_STYLES
        registry_builder = build_promotion_registry
        suite_version = PROMOTION_HOLDOUT_VERSION
    else:
        available_workflows = WORKFLOWS
        registry_styles = EXTERNAL_REGISTRY_STYLES
        registry_builder = build_external_registry
        suite_version = "development.v1"
    if args.condition == "both":
        conditions = ("full", "nomos")
    elif args.condition == "ablation":
        conditions = ("full_raw", "nomos_raw", "nomos")
    else:
        conditions = (args.condition,)
    workflows = (
        tuple(
            workflow
            for workflow in available_workflows
            if workflow["name"] in set(args.workflow)
        )
        if args.workflow
        else available_workflows
    )
    if not workflows:
        raise SystemExit("the selected workflow does not belong to the requested suite")
    traces = []
    for index in range(args.sessions):
        workflow = workflows[index % len(workflows)]
        registry = registry_builder(registry_styles[index % len(registry_styles)])
        for condition in conditions:
            traces.append(
                _run_session(
                    backend,
                    selector if CONDITION_POLICIES[condition]["use_selector"] else None,
                    workflow,
                    registry,
                    condition=condition,
                    max_attempts=args.max_attempts,
                    suite=args.suite,
                )
            )
    summaries = {}
    for condition in conditions:
        rows = [row for row in traces if row["condition"] == condition]
        events = [event for row in rows for event in row["events"]]
        executions = [event for event in events if "execution" in event]
        summaries[condition] = {
            "sessions": len(rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_completed_stage_rate": sum(
                row["completed_stages"] / row["required_stages"] for row in rows
            )
            / len(rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "tool_call_attempts": len(events),
            "prompt_tokens_per_attempt": (
                sum(row["prompt_tokens"] for row in rows) / len(events)
                if events
                else 0.0
            ),
            "successful_execution_rate": (
                sum(bool(event["execution"]["ok"]) for event in executions) / len(events)
                if events
                else 0.0
            ),
            "tool_selection_accuracy": (
                sum(bool(event["selection_correct"]) for event in events) / len(events)
                if events
                else 0.0
            ),
            "schema_valid_call_rate": (
                sum(bool(event["valid"]) for event in events) / len(events)
                if events
                else 0.0
            ),
            "wrong_tool_executions": sum(
                not bool(event["execution"]["ok"]) for event in executions
            ),
            "invalid_calls": sum(not bool(event["valid"]) for event in events),
            "visible_oracle_hit_rate": (
                sum(bool(event["oracle_visible_hit"]) for event in events) / len(events)
                if events
                else 0.0
            ),
            "tool_description_reduction": 1.0
            - sum(row["visible_tools"] for row in rows)
            / sum(row["available_tools"] for row in rows),
        }
    report = {
        "backend": args.backend,
        "suite": args.suite,
        "suite_version": suite_version,
        "sessions_per_condition": args.sessions,
        "max_attempts": args.max_attempts,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    with args.trace_output.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
