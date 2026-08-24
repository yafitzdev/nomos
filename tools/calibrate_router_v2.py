"""Fit and evaluate top-3 confidence calibration for a router.v2 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from fitz_tool.calibration import (
    calibration_metrics,
    choose_abstention_threshold,
    fit_logistic_calibration,
)
from fitz_tool.coprocessor import score_diagnostics
from fitz_tool.router_v2 import load_router_v2, rank_tools_v2, save_router_v2


def _rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _records(model: Any, metadata: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    for row in rows:
        partition = str(row.get("evaluation_partition"))
        if partition not in output or row.get("task_kind") == "verify":
            continue
        acceptable = {str(value) for value in (row.get("label") or {}).get("acceptable_tools") or []}
        if not acceptable:
            continue
        ranked = rank_tools_v2(model, metadata, row, top_k=len(row["legal_candidate_ids"]))
        previous = (
            {str(value) for value in row.get("previous_candidate_ids") or []}
            if row.get("task_kind") == "recover"
            else set()
        )
        ranked = [item for item in ranked if str(item["tool_id"]) not in previous]
        output[partition].append(
            {
                "correct": bool({str(item["tool_id"]) for item in ranked[:3]} & acceptable),
                "features": score_diagnostics([float(item["score"]) for item in ranked]),
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--maximum-selective-risk", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, metadata = load_router_v2(str(args.artifact))
    records = _records(model, metadata, _rows(args.input))
    calibration = fit_logistic_calibration(records["validation"])
    selection = choose_abstention_threshold(
        records["validation"],
        calibration,
        maximum_selective_risk=args.maximum_selective_risk,
    )
    calibration["abstention_threshold"] = selection["threshold"]
    report = {
        "artifact": str(args.artifact),
        "maximum_selective_risk": args.maximum_selective_risk,
        "calibration": calibration,
        "threshold_selection": selection,
        "validation": calibration_metrics(records["validation"], calibration),
        "test": calibration_metrics(records["test"], calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_artifact:
        args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
        metadata["confidence_calibration"] = calibration
        save_router_v2(str(args.output_artifact), model, metadata)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
