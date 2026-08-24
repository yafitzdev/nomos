"""Evaluate Nomos actions, ranking, abstention, verification, and recovery safety."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.coprocessor import coprocessor_response
from fitz_tool.router_v2 import load_router_v2, rank_tools_v2


def _finish(counter: Counter[str]) -> dict[str, float | int]:
    states = int(counter["states"])
    return {
        "states": states,
        "action_accuracy": counter["action_correct"] / states if states else 0.0,
        "top3_recall": counter["top3_correct"] / counter["ranking_states"]
        if counter["ranking_states"]
        else 0.0,
        "ranking_states": int(counter["ranking_states"]),
        "illegal_candidate_rate": counter["illegal_candidates"] / max(1, counter["recommendations"]),
        "recommendations": int(counter["recommendations"]),
    }


def evaluate(
    model: Any,
    metadata: dict[str, Any],
    path: Path,
    *,
    partition: str,
) -> dict[str, Any]:
    overall: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = {}
    recovery: Counter[str] = Counter()
    abstention: Counter[str] = Counter()
    verification: Counter[str] = Counter()
    description_reduction = []
    total_candidate_descriptions = sent_candidate_descriptions = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evaluation_partition") != partition:
                continue
            request = deepcopy(row)
            request["schema_version"] = "runner-request.v2"
            request["request_id"] = row["decision_state_id"]
            ranked = (
                []
                if row["task_kind"] == "verify"
                else rank_tools_v2(
                    model,
                    metadata,
                    row,
                    top_k=len(row["legal_candidate_ids"]),
                )
            )
            response = coprocessor_response(
                request,
                ranked,
                router_version=str(metadata.get("router_version", "unknown")),
                calibration=metadata.get("confidence_calibration"),
            )
            expected_action = str(row["matrix_cell"]["expected_action"])
            kind = str(row["task_kind"])
            pool = str(len(row["legal_candidate_ids"]))
            counters = (
                overall,
                groups.setdefault(f"task:{kind}", Counter()),
                groups.setdefault(f"pool:{pool}", Counter()),
            )
            action_correct = response["action"] == expected_action
            acceptable = {str(value) for value in row["label"]["acceptable_tools"]}
            recommended = [str(item["tool_id"]) for item in response["recommendations"]]
            top3_correct = bool(set(recommended) & acceptable)
            for counter in counters:
                counter["states"] += 1
                counter["action_correct"] += int(action_correct)
                counter["recommendations"] += len(recommended)
                counter["illegal_candidates"] += sum(
                    tool_id not in set(row["legal_candidate_ids"]) for tool_id in recommended
                )
                if acceptable:
                    counter["ranking_states"] += 1
                    counter["top3_correct"] += int(top3_correct)
            if kind == "verify":
                verification["states"] += 1
                verification["correct"] += int(action_correct)
            if kind == "recover":
                recovery["states"] += 1
                recovery["no_repeat"] += int(
                    set(recommended).isdisjoint(row["previous_candidate_ids"])
                )
                recovery["top3_correct"] += int(top3_correct) if acceptable else 0
                recovery["ranking_states"] += int(bool(acceptable))
            if expected_action == "abstain":
                abstention["negative_states"] += 1
                abstention["correct_abstentions"] += int(response["action"] == "abstain")
            elif expected_action == "recommend_tools":
                abstention["positive_states"] += 1
                abstention["false_abstentions"] += int(response["action"] == "abstain")
            if kind != "verify":
                total_candidate_descriptions += len(row["legal_candidate_ids"])
                sent_candidate_descriptions += len(recommended)
                description_reduction.append(
                    1.0 - len(recommended) / len(row["legal_candidate_ids"])
                )
    return {
        "overall": _finish(overall),
        "groups": {key: _finish(value) for key, value in sorted(groups.items())},
        "verification": {
            "states": int(verification["states"]),
            "accuracy": verification["correct"] / verification["states"]
            if verification["states"]
            else 0.0,
        },
        "recovery": {
            "states": int(recovery["states"]),
            "no_repeat_rate": recovery["no_repeat"] / recovery["states"]
            if recovery["states"]
            else 0.0,
            "top3_recall": recovery["top3_correct"] / recovery["ranking_states"]
            if recovery["ranking_states"]
            else 0.0,
        },
        "abstention": {
            "negative_states": int(abstention["negative_states"]),
            "abstention_recall": abstention["correct_abstentions"] / abstention["negative_states"]
            if abstention["negative_states"]
            else 0.0,
            "positive_states": int(abstention["positive_states"]),
            "false_abstention_rate": abstention["false_abstentions"] / abstention["positive_states"]
            if abstention["positive_states"]
            else 0.0,
        },
        "mean_tool_description_reduction": sum(description_reduction) / len(description_reduction)
        if description_reduction
        else 0.0,
        "weighted_tool_description_reduction": (
            1.0 - sent_candidate_descriptions / total_candidate_descriptions
            if total_candidate_descriptions
            else 0.0
        ),
        "candidate_descriptions_available": total_candidate_descriptions,
        "candidate_descriptions_sent": sent_candidate_descriptions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, metadata = load_router_v2(str(args.artifact))
    report = {
        "artifact": str(args.artifact),
        "input": str(args.input),
        "partition": args.partition,
        "metrics": evaluate(model, metadata, args.input, partition=args.partition),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
