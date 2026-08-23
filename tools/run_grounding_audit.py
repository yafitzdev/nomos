"""Audit a stratified scenario sample against an external V2 retriever.

This is a grounding check, not a trajectory labeler. It verifies that the
scenario's expected source-card facts can be retrieved from the immutable
corpus. Tool-choice labels still require the runner.v1 contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.contracts import validate_scenario, validate_source_card
from fitz_tool.uniqueness import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", default="fitz_tool_grounding_audit")
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _content_hashes(path: Path) -> set[str]:
    raw = path.read_bytes()
    hashes = {hashlib.sha256(raw).hexdigest()}
    if path.suffix.casefold() in {".md", ".txt", ".rst", ".yaml", ".yml", ".json", ".py", ".js", ".ts"}:
        normalized = path.read_text(encoding="utf-8", errors="replace").strip()
        hashes.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return hashes


def _fact_matches(
    scenario: dict[str, Any],
    result: dict[str, Any],
    facts: dict[str, str],
    source_hashes: set[str],
    probe_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    direct_evidence = [
        match.get("evidence", {})
        for match in result.get("matches", [])
        if isinstance(match, dict) and isinstance(match.get("evidence"), dict)
    ]
    direct_text = " ".join(
        normalize_text(str(item.get(key, "")))
        for item in direct_evidence
        for key in ("content", "excerpt")
    )
    expected_ids = [
        str(item.get("fact_id"))
        for item in scenario.get("expected_facts", [])
        if isinstance(item, dict) and item.get("fact_id")
    ]
    direct_matched_ids = [
        fact_id
        for fact_id in expected_ids
        if normalize_text(facts[fact_id]) in direct_text
    ]
    probe_evidence = [
        match.get("evidence", {})
        for probe in probe_results.values()
        for match in probe.get("matches", [])
        if isinstance(match, dict) and isinstance(match.get("evidence"), dict)
    ]
    probe_text = " ".join(
        normalize_text(str(item.get(key, "")))
        for item in probe_evidence
        for key in ("content", "excerpt")
    )
    matched_ids = [
        fact_id
        for fact_id in expected_ids
        if fact_id in direct_matched_ids or normalize_text(facts[fact_id]) in probe_text
    ]
    evidence = [*direct_evidence, *probe_evidence]
    evidence_hashes = {
        str(item.get("content_sha256") or metadata.get("content_sha256"))
        for item in evidence
        for metadata in [item.get("metadata") if isinstance(item.get("metadata"), dict) else {}]
        if item.get("content_sha256") or metadata.get("content_sha256")
    }
    provenance_ok = not evidence or bool(evidence_hashes & source_hashes)
    topology = scenario.get("matrix_cell", {}).get("evidence_topology")
    if topology == "absent":
        passed = not evidence and not expected_ids
    else:
        passed = len(matched_ids) == len(expected_ids) and provenance_ok
    return {
        "expected_fact_ids": expected_ids,
        "question_matched_fact_ids": direct_matched_ids,
        "matched_fact_ids": matched_ids,
        "missing_fact_ids": [fact_id for fact_id in expected_ids if fact_id not in matched_ids],
        "question_retrieved_evidence": len(direct_evidence),
        "probe_retrieved_evidence": len(probe_evidence),
        "retrieved_evidence": len(evidence),
        "retrieved_source_ids": sorted({str(item.get("source_id")) for item in evidence}),
        "retrieved_content_hashes": sorted(evidence_hashes),
        "provenance_ok": provenance_ok,
        "status": result.get("status"),
        "question_retrieval_passed": len(direct_matched_ids) == len(expected_ids),
        "grounding_passed": passed,
    }


def _run_retriever(
    *,
    v2_root: Path,
    source_root: Path,
    collection: str,
    question: str,
    candidate_k: int,
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-c",
        "from fitz_agent.retrieval_cli import main; raise SystemExit(main())",
        "--source",
        str(source_root),
        "--collection",
        collection,
        "--candidate-k",
        str(candidate_k),
        "--top-k",
        str(top_k),
        "--no-prewarm",
        "--question",
        question,
        "--json",
    ]
    completed = subprocess.run(
        command,
        cwd=v2_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"retrieval CLI exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"retrieval CLI did not return JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("retrieval CLI result must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.source_root = args.source_root.resolve()
    args.v2_root = args.v2_root.resolve()
    scenarios = _read_jsonl(args.scenarios)
    manifest = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    selected_indices = [int(item["row_index"]) for item in manifest.get("rows", [])]
    selected = [scenarios[index] for index in selected_indices]
    source_card = json.loads(args.source_card.read_text(encoding="utf-8"))
    source_report = validate_source_card(source_card)
    if not source_report.valid:
        raise SystemExit(json.dumps(source_report.as_dict(), indent=2))
    facts = {
        str(item["fact_id"]): str(item["statement"])
        for item in source_card["facts"]
    }
    source_hashes = {str(source_card["content_sha256"])}
    if source_card.get("normalized_content_sha256"):
        source_hashes.add(str(source_card["normalized_content_sha256"]))
    source_path_value = source_card.get("source_path") or source_card.get("metadata", {}).get("source_path")
    source_path = args.source_root / str(source_path_value or "")
    if source_path.is_file():
        source_hashes |= _content_hashes(source_path)

    audit_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, scenario in zip(selected_indices, selected, strict=True):
        report = validate_scenario(scenario)
        if not report.valid:
            errors.append({"row_index": index, "scenario_id": scenario.get("scenario_id"), "validation": report.as_dict()})
            continue
        try:
            result = _run_retriever(
                v2_root=args.v2_root,
                source_root=args.source_root,
                collection=args.collection,
                question=str(scenario["question"]),
                candidate_k=args.candidate_k,
                top_k=args.top_k,
                timeout=args.timeout,
            )
            direct_text = " ".join(
                normalize_text(str(match.get("evidence", {}).get(key, "")))
                for match in result.get("matches", [])
                if isinstance(match, dict) and isinstance(match.get("evidence"), dict)
                for key in ("content", "excerpt")
            )
            probe_results: dict[str, dict[str, Any]] = {}
            for fact in scenario.get("expected_facts", []):
                fact_id = str(fact.get("fact_id"))
                statement = facts.get(fact_id)
                if statement and normalize_text(statement) not in direct_text:
                    probe_results[fact_id] = _run_retriever(
                        v2_root=args.v2_root,
                        source_root=args.source_root,
                        collection=args.collection,
                        question=statement,
                        candidate_k=args.candidate_k,
                        top_k=args.top_k,
                        timeout=args.timeout,
                    )
            grounding = _fact_matches(scenario, result, facts, source_hashes, probe_results)
            audit_rows.append(
                {
                    "row_index": index,
                    "scenario_id": scenario["scenario_id"],
                    "matrix_cell": scenario["matrix_cell"],
                    **grounding,
                }
            )
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            errors.append(
                {
                    "row_index": index,
                    "scenario_id": scenario.get("scenario_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    passed = not errors and all(row["grounding_passed"] for row in audit_rows)
    payload = {
        "manifest_version": "grounding-audit.v1",
        "source_card_id": source_card["source_id"],
        "source_card_hashes": sorted(source_hashes),
        "collection": args.collection,
        "requested": len(selected),
        "audited": len(audit_rows),
        "passed": passed,
        "errors": errors,
        "rows": audit_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"requested": len(selected), "audited": len(audit_rows), "errors": len(errors), "passed": passed, "output": str(args.output)}, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
