"""Adapt the external Fitz-Sage V2 CLI to the runner.v1 JSONL contract.

This adapter deliberately launches V2 as a subprocess. Fitz-Tool never imports
V2 Python modules, and the adapter only consumes the public JSON result emitted
by ``fitz_agent.cli``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.uniqueness import normalize_text, stable_hash


RUNNER_NAME = "fitz-sage-v2-cli-adapter"
RUNNER_VERSION = "v2-cli-adapter.v1"
CONTRACT_VERSION = "runner.v1"
SEARCH_TOOLS = {
    "search_bm25",
    "grep_search",
    "search_metadata",
    "search_table_rows",
    "search_pdf_pages",
}
PROGRESS_TOOLS = {
    "inspect_evidence",
    "expand_context",
    "compare_evidence",
    "update_requirement_progress",
    "assess_evidence",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-card", type=Path, action="append", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:19003/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--backend", choices=("llama-cpp",), default="llama-cpp")
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--governance", choices=("off", "shadow", "enforce"), default="shadow")
    parser.add_argument("--scenario-timeout", type=float, default=300.0)
    parser.add_argument("--no-prewarm", action="store_true")
    return parser


def _read_cards(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for path in paths:
        card = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(card, dict) or not isinstance(card.get("source_id"), str):
            raise SystemExit(f"invalid source card: {path}")
        cards[str(card["source_id"])] = card
    return cards


def _run_v2(
    scenario: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-c",
        "from fitz_agent.cli import main; raise SystemExit(main())",
        "--source",
        str(args.source_root.resolve()),
        "--collection",
        f"fitz_tool_runner_{stable_hash({'scenario_id': scenario.get('scenario_id')})[:12]}",
        "--db",
        ":memory:",
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--base-url",
        args.base_url,
        "--max-steps",
        str(args.max_steps),
        "--governance",
        args.governance,
        "--question",
        str(scenario["question"]),
        "--json",
    ]
    if args.no_prewarm:
        command.append("--no-prewarm")
    completed = subprocess.run(
        command,
        cwd=args.v2_root.resolve(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=args.scenario_timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"V2 exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"V2 did not return JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("V2 result must be an object")
    return result


def _source_fact_index(cards: Mapping[str, Mapping[str, Any]]) -> tuple[dict[tuple[str, str], str], set[str]]:
    statements: dict[tuple[str, str], str] = {}
    hashes: set[str] = set()
    for source_id, card in cards.items():
        for key in ("content_sha256", "normalized_content_sha256"):
            value = card.get(key)
            if isinstance(value, str):
                hashes.add(value)
        for fact in card.get("facts", []):
            if isinstance(fact, Mapping) and isinstance(fact.get("fact_id"), str):
                statements[(source_id, str(fact["fact_id"]))] = str(fact.get("statement") or "")
    return statements, hashes


def _evidence_text(result: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for item in result.get("selected_evidence", []):
        if isinstance(item, Mapping):
            pieces.extend(str(item.get(key) or "") for key in ("excerpt", "content"))
    return normalize_text(" ".join(pieces))


def _acceptance(
    scenario: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    fact_statements: Mapping[tuple[str, str], str],
    source_hashes: set[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if result.get("status") != "selected":
        reasons.append(f"V2 terminal status is {result.get('status')!r}, not selected")
    selection = result.get("metadata", {}).get("selection", {})
    if selection.get("validated") is not True:
        reasons.append("V2 selection was not deterministically validated")
    if selection.get("selection_rewrites"):
        reasons.append("V2 applied selection rewrites")

    selected = [item for item in result.get("selected_evidence", []) if isinstance(item, Mapping)]
    selected_text = _evidence_text(result)
    missing: list[str] = []
    provenance_bad = False
    for fact in scenario.get("expected_facts", []):
        if not isinstance(fact, Mapping):
            continue
        source_id = str(fact.get("source_id") or "")
        fact_id = str(fact.get("fact_id") or "")
        statement = fact_statements.get((source_id, fact_id), "")
        if statement and normalize_text(statement) not in selected_text:
            missing.append(f"{source_id}:{fact_id}")
    for item in selected:
        provenance = item.get("provenance")
        content_hash = provenance.get("content_sha256") if isinstance(provenance, Mapping) else None
        if content_hash and str(content_hash) not in source_hashes:
            provenance_bad = True
    if missing:
        reasons.append("missing expected facts: " + ", ".join(missing))
    if provenance_bad:
        reasons.append("selected evidence provenance is outside the source-card hashes")
    if not scenario.get("expected_facts") and selected:
        reasons.append("scenario expects no facts but V2 selected evidence")
    return not reasons, reasons


def _progress(result: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    progress = result.get("metadata", {}).get("requirement_progress", [])
    if not isinstance(progress, list) or not progress:
        return []
    latest = progress[-1]
    if not isinstance(latest, Mapping):
        return []
    values = latest.get("progress")
    return [dict(item) for item in values if isinstance(item, Mapping)] if isinstance(values, list) else []


def _governance(result: Mapping[str, Any], trace_index: int) -> dict[str, Any]:
    trajectory = result.get("metadata", {}).get("governance_trajectory", [])
    latest = trajectory[-1] if isinstance(trajectory, list) and trajectory else {}
    fresh = bool(latest) and trace_index > int(latest.get("step_number", -1))
    return {
        "assessment_fresh": fresh,
        "requirements": _progress(result, trace_index),
        "verdict": latest.get("verdict") if isinstance(latest, Mapping) else None,
    }


def _state_name(
    action: str,
    action_result: Mapping[str, Any] | None,
    *,
    first: bool,
    prior_evidence: Sequence[str],
    result: Mapping[str, Any],
) -> str:
    if first:
        return "initial"
    if action == "compare_evidence":
        return "contradiction"
    if action == "assess_evidence":
        verdict = str((result.get("metadata", {}).get("governance") or {}).get("verdict") or "")
        if verdict == "SUFFICIENT":
            return "fresh_sufficient"
        if verdict == "DISPUTED":
            return "disputed"
        return "insufficient"
    if action == "finalize_document_selection":
        return "fresh_sufficient" if result.get("status") == "selected" else "insufficient"
    evidence_ids = list((action_result or {}).get("evidence_ids") or [])
    if action in SEARCH_TOOLS:
        return "no_hits" if not evidence_ids else "noisy_hits"
    if action in PROGRESS_TOOLS:
        return "partial_evidence" if prior_evidence or evidence_ids else "expansion_needed"
    return "partial_evidence" if prior_evidence else "initial"


def _trajectory(
    scenario: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fact_statements, source_hashes = _source_fact_index(cards)
    accepted, rejection_reasons = _acceptance(
        scenario,
        result,
        fact_statements=fact_statements,
        source_hashes=source_hashes,
    )
    trace = [item for item in result.get("tool_trace", []) if isinstance(item, Mapping)]
    diagnostics = result.get("metadata", {}).get("model_turn_diagnostics", [])
    diagnostics = [item for item in diagnostics if isinstance(item, Mapping)]
    events: list[dict[str, Any]] = []
    prior_history: list[dict[str, Any]] = []
    prior_evidence: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        action = str(diagnostic.get("tool_call") or "")
        action_result = trace[index] if index < len(trace) else None
        legal_tools = [str(tool) for tool in diagnostic.get("allowed_tools", [])]
        executed = action if action in legal_tools else ""
        if not executed:
            events.append(
                {
                    "step": index,
                    "kind": "error",
                    "error": "V2 produced no legal tool call for the visible decision state",
                    "legal_tools": legal_tools,
                    "model_output": diagnostic.get("text"),
                    "diagnostic_error": diagnostic.get("error"),
                }
            )
            continue
        state_name = _state_name(
            action,
            action_result,
            first=index == 0,
            prior_evidence=prior_evidence,
            result=result,
        )
        decision = {
            "step": index,
            "kind": "decision",
            "agent_state": {"state_name": state_name, "source": "fitz-sage-v2"},
            "history": list(prior_history),
            "plan": result.get("plan", {}),
            "matrix_context": dict(scenario.get("matrix_cell") or {}),
            "observed_evidence": [{"evidence_id": item} for item in prior_evidence],
            "governance": _governance(result, index),
            "legal_tools": legal_tools,
            "proposed_tool": executed or None,
            "executed_tool": executed or None,
            "acceptable_tools": [executed] if accepted and executed else [],
            "hard_negative_tools": [executed] if not accepted and executed else [],
            "ranked_tools": [executed] if executed else [],
        }
        events.append(decision)
        if action_result is not None:
            result_ids = [str(item) for item in action_result.get("evidence_ids", [])]
            events.append(
                {
                    "step": index,
                    "kind": "tool_result",
                    "tool": executed or action,
                    "status": action_result.get("status", "ok"),
                    "evidence_ids": result_ids,
                    "error": action_result.get("error"),
                }
            )
            prior_history.append(
                {
                    "tool": executed or action,
                    "status": action_result.get("status", "ok"),
                    "result_count": len(result_ids),
                }
            )
            for evidence_id in result_ids:
                if evidence_id not in prior_evidence:
                    prior_evidence.append(evidence_id)
    events.append(
        {
            "step": len(diagnostics),
            "kind": "terminal",
            "terminal_result": {
                "status": result.get("status", "error"),
                "selected_evidence": result.get("selected_evidence", []),
            },
        }
    )
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "schema_version": "trajectory.v1",
        "trajectory_id": "v2-" + stable_hash({"scenario_id": scenario.get("scenario_id"), "run_id": result.get("run_id")})[:24],
        "scenario_id": str(scenario["scenario_id"]),
        "question": str(scenario["question"]),
        "runner": {
            "name": RUNNER_NAME,
            "version": RUNNER_VERSION,
            "contract_version": CONTRACT_VERSION,
        },
        "events": events,
        "terminal_result": {
            "status": result.get("status", "error"),
            "run_id": result.get("run_id"),
            "selected_evidence": result.get("selected_evidence", []),
        },
        "validation": {
            "trajectory_accepted": accepted,
            "rejection_reasons": rejection_reasons,
        },
        "provenance": {
            "captured_at": captured_at,
            "v2_run_id": result.get("run_id"),
            "v2_model": result.get("metadata", {}).get("model"),
            "v2_backend": result.get("metadata", {}).get("inference_backend"),
            "v2_base_url": result.get("metadata", {}).get("inference_base_url"),
        },
    }


def _error_trajectory(scenario: Mapping[str, Any], error: str) -> dict[str, Any]:
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "schema_version": "trajectory.v1",
        "trajectory_id": "v2-error-" + stable_hash({"scenario_id": scenario.get("scenario_id"), "error": error})[:24],
        "scenario_id": str(scenario["scenario_id"]),
        "question": str(scenario.get("question") or ""),
        "runner": {"name": RUNNER_NAME, "version": RUNNER_VERSION, "contract_version": CONTRACT_VERSION},
        "events": [{"step": 0, "kind": "error", "error": error}],
        "terminal_result": {"status": "runner_error"},
        "validation": {"trajectory_accepted": False, "rejection_reasons": [error]},
        "provenance": {"captured_at": captured_at},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps < 5:
        raise SystemExit("--max-steps must be at least 5")
    cards = _read_cards(args.source_card)
    for line in sys.stdin:
        if not line.strip():
            continue
        scenario = json.loads(line)
        if not isinstance(scenario, dict):
            raise SystemExit("runner input must contain scenario objects")
        try:
            output = _trajectory(scenario, _run_v2(scenario, args=args), cards=cards)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            output = _error_trajectory(scenario, f"{type(exc).__name__}: {exc}")
        sys.stdout.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
