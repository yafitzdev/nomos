"""Consolidate locked backbone-bakeoff reports into one machine-readable table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


LABELS = (
    "accepted_bge",
    "fresh_bge",
    "gte_modernbert",
    "lfm_embedding_350m",
    "lfm_encoder_230m",
    "lfm_colbert_350m",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _input_metrics(report: dict[str, Any], needle: str) -> dict[str, Any]:
    return next(
        value["metrics"]
        for path, value in report["inputs"].items()
        if needle in path
    )


def summarize(run_dir: Path) -> dict[str, Any]:
    models = {}
    for label in LABELS:
        retrieval = _load(run_dir / f"{label}_retrieval.json")
        final = _load(run_dir / f"{label}_final.json")
        promotion = _load(run_dir / f"{label}_promotion.json")
        toolret = _load(run_dir / f"{label}_toolret.json")
        cpu = _load(run_dir / f"{label}_cpu.json")
        top3 = {
            "frozen": _input_metrics(retrieval, "agentic_v2")["recall_at_3"],
            "generic": _input_metrics(retrieval, "generic_baseline")["recall_at_3"],
            "sealed": _input_metrics(retrieval, "post_scaling")["recall_at_3"],
            "final": final["metrics"]["recall_at_3"],
            "promotion": promotion["metrics"]["recall_at_3"],
            "toolret": toolret["metrics"]["recall_at_3"],
        }
        models[label] = {
            "top1": {
                "frozen": _input_metrics(retrieval, "agentic_v2")["recall_at_1"],
                "generic": _input_metrics(retrieval, "generic_baseline")["recall_at_1"],
                "sealed": _input_metrics(retrieval, "post_scaling")["recall_at_1"],
                "final": final["metrics"]["recall_at_1"],
                "promotion": promotion["metrics"]["recall_at_1"],
                "toolret": toolret["metrics"]["recall_at_1"],
            },
            "top3": top3,
            "macro_top3": sum(top3.values()) / len(top3),
            "artifact_mebibytes": cpu["artifact"]["mebibytes"],
            "rss_load_delta_mebibytes": cpu["load"]["rss_load_delta_mebibytes"],
            "load_seconds": cpu["load"]["seconds"],
            "warm_query_p50_ms": {
                pool: cpu["ranking"][pool]["warm_query_p50_ms"]
                for pool in ("10", "30", "100")
            },
            "warm_query_p95_ms": {
                pool: cpu["ranking"][pool]["warm_query_p95_ms"]
                for pool in ("10", "30", "100")
            },
            "cold_registry_and_query_ms": {
                pool: cpu["ranking"][pool]["cold_registry_and_query_ms"]
                for pool in ("10", "30", "100")
            },
        }
    return {
        "suite": "nomos-backbone-bakeoff.v1",
        "primary_metric": "raw top-3 tool recall",
        "models": models,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/backbone_bakeoff_v1"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = summarize(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                label: round(value["macro_top3"] * 100, 2)
                for label, value in report["models"].items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
