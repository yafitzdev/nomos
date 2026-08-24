"""Evaluate the production dense coprocessor on frozen route/verify/recovery states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.coprocessor import coprocessor_response
from fitz_tool.dense_selector import DenseToolRanker


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / denominator if denominator else 0.0


def evaluate(
    ranker: DenseToolRanker, path: Path, *, partition: str
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    validation_reasons: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evaluation_partition") != partition:
                continue
            request = deepcopy(row)
            request["schema_version"] = "runner-request.v2"
            request["request_id"] = str(row["decision_state_id"])
            kind = str(row["task_kind"])
            ranked = [] if kind == "verify" else ranker.rank(request)
            response = coprocessor_response(
                request,
                ranked,
                router_version=ranker.version,
                calibration=ranker.calibration,
            )
            acceptable = {
                str(value) for value in row["label"]["acceptable_tools"]
            }
            expected_action = str(
                row.get("matrix_cell", {}).get("expected_action")
                or (
                    "accept_tool_call"
                    if kind == "verify" and row.get("validation_label", {}).get("valid")
                    else "reject_tool_call"
                    if kind == "verify"
                    else "recommend_tools"
                    if acceptable
                    else "abstain"
                )
            )
            recommended = [
                str(item["tool_id"]) for item in response["recommendations"]
            ]
            counts["states"] += 1
            counts["action_correct"] += int(response["action"] == expected_action)
            counts["recommendations"] += len(recommended)
            counts["available_descriptions"] += (
                0 if kind == "verify" else len(row["legal_candidate_ids"])
            )
            counts["sent_descriptions"] += len(recommended)
            counts["illegal_recommendations"] += sum(
                tool_id not in set(row["legal_candidate_ids"])
                for tool_id in recommended
            )
            if kind != "verify" and acceptable:
                counts["ranking_states"] += 1
                counts["top3_correct"] += int(bool(set(recommended) & acceptable))
            if kind == "recover":
                counts["recovery_states"] += 1
                counts["recovery_no_repeat"] += int(
                    set(recommended).isdisjoint(row["previous_candidate_ids"])
                )
                if acceptable:
                    counts["recovery_ranking_states"] += 1
                    counts["recovery_top3_correct"] += int(
                        bool(set(recommended) & acceptable)
                    )
            if kind == "verify":
                expected_validation = row["validation_label"]
                actual_validation = response["validation"]
                exact = all(
                    actual_validation[key] == expected_validation[key]
                    for key in ("valid", "tool_id", "repairable", "checked")
                ) and set(actual_validation["failure_reasons"]) == set(
                    expected_validation["failure_reasons"]
                )
                counts["verification_states"] += 1
                counts["verification_exact"] += int(exact)
                if not expected_validation["valid"]:
                    counts["invalid_calls"] += 1
                    counts["false_accepts"] += int(
                        response["action"] == "accept_tool_call"
                    )
                    validation_reasons.update(expected_validation["failure_reasons"])
                else:
                    counts["valid_calls"] += 1
                    counts["false_rejects"] += int(
                        response["action"] == "reject_tool_call"
                    )
            if expected_action == "abstain":
                counts["no_suitable_states"] += 1
                counts["correct_abstentions"] += int(response["action"] == "abstain")
            elif expected_action == "recommend_tools":
                counts["suitable_states"] += 1
                counts["false_abstentions"] += int(response["action"] == "abstain")

    return {
        "states": int(counts["states"]),
        "action_accuracy": _ratio(counts["action_correct"], counts["states"]),
        "ranking": {
            "states": int(counts["ranking_states"]),
            "recall_at_3": _ratio(
                counts["top3_correct"], counts["ranking_states"]
            ),
            "illegal_candidate_rate": _ratio(
                counts["illegal_recommendations"], counts["recommendations"]
            ),
        },
        "verification": {
            "states": int(counts["verification_states"]),
            "exact_accuracy": _ratio(
                counts["verification_exact"], counts["verification_states"]
            ),
            "invalid_calls": int(counts["invalid_calls"]),
            "false_accept_rate": _ratio(
                counts["false_accepts"], counts["invalid_calls"]
            ),
            "valid_calls": int(counts["valid_calls"]),
            "false_reject_rate": _ratio(
                counts["false_rejects"], counts["valid_calls"]
            ),
            "invalid_reason_counts": dict(sorted(validation_reasons.items())),
        },
        "recovery": {
            "states": int(counts["recovery_states"]),
            "no_repeat_rate": _ratio(
                counts["recovery_no_repeat"], counts["recovery_states"]
            ),
            "recall_at_3": _ratio(
                counts["recovery_top3_correct"], counts["recovery_ranking_states"]
            ),
        },
        "abstention": {
            "no_suitable_states": int(counts["no_suitable_states"]),
            "abstention_recall": _ratio(
                counts["correct_abstentions"], counts["no_suitable_states"]
            ),
            "suitable_states": int(counts["suitable_states"]),
            "false_abstention_rate": _ratio(
                counts["false_abstentions"], counts["suitable_states"]
            ),
        },
        "weighted_tool_description_reduction": 1.0
        - _ratio(counts["sent_descriptions"], counts["available_descriptions"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument(
        "--query-strategy", choices=("single", "multiview"), default="multiview"
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=("single", "multiview"),
        default="multiview",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ranker = DenseToolRanker.from_path(
        args.model,
        device=args.device,
        query_strategy=args.query_strategy,
        candidate_strategy=args.candidate_strategy,
    )
    report = {
        "model": str(args.model),
        "input": str(args.input),
        "partition": args.partition,
        "device": args.device,
        "query_strategy": args.query_strategy,
        "candidate_strategy": args.candidate_strategy,
        "metrics": evaluate(ranker, args.input, partition=args.partition),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
