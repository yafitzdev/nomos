"""Run the locked Nomos backbone bakeoff across raw retrieval and CPU suites."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_MODELS = {
    "accepted_bge": Path("artifacts/nomos_bge_scaling_balanced_mnrl_soup10_v2"),
    "fresh_bge": Path("artifacts/nomos_backbone_bge_control_v1"),
    "gte_modernbert": Path("artifacts/nomos_backbone_gte_modernbert_v1"),
    "lfm_embedding_350m": Path("artifacts/nomos_backbone_lfm25_embedding_350m_v1"),
    "lfm_encoder_230m": Path("artifacts/nomos_backbone_lfm25_encoder_230m_v1"),
    "lfm_colbert_350m": Path("artifacts/nomos_backbone_lfm25_colbert_350m_v1"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Model to evaluate; omit to run the locked six-model comparison.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/backbone_bakeoff_v1"))
    parser.add_argument(
        "--stage",
        action="append",
        choices=("retrieval", "final", "promotion", "toolret", "cpu"),
        default=[],
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def _models(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_MODELS)
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must use LABEL=PATH syntax: {value}")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError(f"model must use LABEL=PATH syntax: {value}")
        parsed[label] = Path(path)
    return parsed


def _run(command: list[str], output: Path, *, resume: bool) -> None:
    if resume and output.exists():
        print(f"skip {output}", flush=True)
        return
    print(f"run  {output}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        if result.stdout:
            print(result.stdout[-4000:], file=sys.stderr)
        if result.stderr:
            print(result.stderr[-8000:], file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    report = json.loads(output.read_text(encoding="utf-8"))
    metrics = report.get("metrics")
    if metrics is None and "inputs" in report:
        metrics = {
            Path(path).stem: value["metrics"]
            for path, value in report["inputs"].items()
        }
    print(json.dumps(metrics or {"completed": True}, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = _models(args.model)
    stages = set(args.stage or ("retrieval", "final", "promotion", "toolret", "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, model in models.items():
        if not model.is_dir():
            raise FileNotFoundError(model)
        prefix = args.output_dir / label
        if "retrieval" in stages:
            output = prefix.with_name(f"{label}_retrieval.json")
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.evaluate_dense_router",
                    "--model",
                    str(model),
                    "--input",
                    "data/generated/nomos_agentic_v2_frozen_10000.jsonl",
                    "--input",
                    "data/generated/nomos_generic_baseline_holdout_1000.jsonl",
                    "--input",
                    "data/generated/nomos_post_scaling_holdout_v1.jsonl",
                    "--partition",
                    "test",
                    "--partition",
                    "post_scaling_sealed",
                    "--limit",
                    "10000",
                    "--seed",
                    "20260824",
                    "--batch-size",
                    "64",
                    "--device",
                    args.device,
                    "--output",
                    str(output),
                ],
                output,
                resume=args.resume,
            )
        for suite in ("final", "promotion"):
            if suite not in stages:
                continue
            output = prefix.with_name(f"{label}_{suite}.json")
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.audit_real_session_retrieval",
                    "--model",
                    str(model),
                    "--suite",
                    suite,
                    "--strategy",
                    "multiview",
                    "--candidate-strategy",
                    "multiview",
                    "--batch-size",
                    "64",
                    "--device",
                    args.device,
                    "--output",
                    str(output),
                ],
                output,
                resume=args.resume,
            )
        if "toolret" in stages:
            output = prefix.with_name(f"{label}_toolret.json")
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.evaluate_toolret",
                    "--model",
                    str(model),
                    "--task",
                    "toolalpaca",
                    "--task",
                    "metatool",
                    "--task",
                    "appbench",
                    "--limit-per-task",
                    "20",
                    "--seed",
                    "20260828",
                    "--batch-size",
                    "32",
                    "--max-seq-length",
                    "512",
                    "--candidate-strategy",
                    "multiview",
                    "--device",
                    args.device,
                    "--output",
                    str(output),
                ],
                output,
                resume=args.resume,
            )
        if "cpu" in stages:
            output = prefix.with_name(f"{label}_cpu.json")
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.benchmark_dense_coprocessor",
                    "--model",
                    str(model),
                    "--input",
                    "data/generated/nomos_agentic_v2_frozen_10000.jsonl",
                    "--partition",
                    "test",
                    "--pool",
                    "10",
                    "--pool",
                    "30",
                    "--pool",
                    "100",
                    "--warm-repetitions",
                    "30",
                    "--verifier-repetitions",
                    "1000",
                    "--output",
                    str(output),
                ],
                output,
                resume=args.resume,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
