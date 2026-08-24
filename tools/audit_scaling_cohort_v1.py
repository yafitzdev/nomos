"""Run a stratified structural and naturalness audit over accepted scaling rows."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.scaling_matrix_v1 import PROJECT_MARKER, normalized_question, semantic_signature
from fitz_tool.tool_registry import ToolRegistry


ABSTENTION_LEAKAGE = re.compile(
    r"\b(abstain|no (?:available )?(?:tool|operation)|none of the available|cannot perform|unsuitable candidate)\b",
    re.I,
)
RECOVERY_CUE = re.compile(
    r"\b(again|another|different|earlier|failed|fresh|previous|prior|rejected|avoid)\b",
    re.I,
)
META_LANGUAGE = re.compile(r"\b(assignment id|benchmark|ground truth|hidden label|training (?:row|data)|routing matrix)\b", re.I)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size >= len(rows):
        return list(rows)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["matrix_cell"]["scenario_family"]), []).append(row)
    rng = random.Random(seed)
    for values in by_family.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < size:
        for family in sorted(by_family):
            if len(selected) >= size:
                break
            if offset < len(by_family[family]):
                selected.append(by_family[family][offset])
        offset += 1
    return selected


def _audit_row(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    report = validate_decision_state_v2(row)
    if not report.valid:
        reasons.append("invalid_decision_state_contract")
    question = str(row.get("question") or "")
    current_step = str(row.get("teacher_paraphrase") or (row.get("plan") or {}).get("remaining_step") or "")
    combined = f"{question}\n{current_step}"
    words = re.findall(r"[a-z0-9]+", question.casefold())
    if len(words) < 7 or len(set(words)) < 5:
        reasons.append("unnatural_or_underspecified_question")
    if question[-1:] not in {".", "?", "!"}:
        reasons.append("question_missing_terminal_punctuation")
    if PROJECT_MARKER.search(combined):
        reasons.append("project_specific_language")
    if META_LANGUAGE.search(combined):
        reasons.append("generation_meta_language")
    registry = ToolRegistry.from_dict(row["tool_registry"])
    if any(tool_id.casefold() in combined.casefold() for tool_id in registry.by_id):
        reasons.append("tool_id_leakage")
    legal = set(map(str, row.get("legal_candidate_ids") or []))
    label = row.get("label") or {}
    positives = list(map(str, label.get("acceptable_tools") or []))
    negatives = list(map(str, label.get("hard_negative_tools") or []))
    if not set(positives) <= legal or not set(negatives) <= legal:
        reasons.append("label_outside_legal_set")
    if positives and len(negatives) < 2:
        reasons.append("insufficient_hard_negatives")
    if positives and any(registry.require(value).description == registry.require(positives[0]).description for value in negatives[:2]):
        reasons.append("indistinguishable_positive_and_negative")
    if not positives and ABSTENTION_LEAKAGE.search(combined):
        reasons.append("explicit_abstention_answer_leakage")
    previous = set(map(str, row.get("previous_candidate_ids") or []))
    if row.get("task_kind") == "recover":
        if not previous or previous & legal:
            reasons.append("invalid_recovery_candidate_history")
        if not RECOVERY_CUE.search(combined):
            reasons.append("recovery_not_observable_in_wording")
    expected_history = {"empty": 0, "short": 1, "long": 3}.get(
        str((row.get("matrix_cell") or {}).get("history_length"))
    )
    if expected_history is not None and len(row.get("history") or []) != expected_history:
        reasons.append("history_length_inconsistent")
    if positives:
        target = registry.require(positives[0])
        available = set(map(str, (row.get("source_state") or {}).get("available_modalities") or []))
        if not set(target.input_modalities) & available:
            reasons.append("target_modality_unavailable")
    if len(normalized_question(question)) < 20:
        reasons.append("question_normalization_too_short")
    return reasons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load(args.input)
    if args.sample_size < 100:
        raise SystemExit("quality audit sample-size must be at least 100")
    sample = _sample(rows, args.sample_size, args.seed)
    records = []
    reason_counts: Counter[str] = Counter()
    for row in sample:
        reasons = _audit_row(row)
        reason_counts.update(reasons)
        records.append(
            {
                "decision_state_id": row.get("decision_state_id"),
                "scenario_family": (row.get("matrix_cell") or {}).get("scenario_family"),
                "question": row.get("question"),
                "current_step": row.get("teacher_paraphrase"),
                "passed": not reasons,
                "reasons": reasons,
            }
        )
    normalized = [normalized_question(str(row["question"])) for row in rows]
    semantic = [semantic_signature(str(row["question"]), str(row.get("teacher_paraphrase") or "")) for row in rows]
    systemic = {
        reason: count
        for reason, count in reason_counts.items()
        if count / len(sample) >= 0.05
    }
    report = {
        "input": str(args.input),
        "input_rows": len(rows),
        "sample_size": len(sample),
        "seed": args.seed,
        "families_covered": sorted({str(row["matrix_cell"]["scenario_family"]) for row in sample}),
        "family_count": len({str(row["matrix_cell"]["scenario_family"]) for row in sample}),
        "passed_rows": sum(value["passed"] for value in records),
        "failed_rows": sum(not value["passed"] for value in records),
        "reason_counts": dict(sorted(reason_counts.items())),
        "systemic_defects": systemic,
        "unique_questions": len(set(normalized)),
        "unique_semantic_signatures": len(set(semantic)),
        "training_allowed": not systemic,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, sort_keys=True))
    return 0 if not systemic else 2


if __name__ == "__main__":
    raise SystemExit(main())
