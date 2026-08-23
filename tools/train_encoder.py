"""Train and evaluate the first state-aware Fitz-Tool router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.contracts import validate_decision_state
from fitz_tool.router import RouterConfig, save_router, train_router


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def _read_states(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict):
            errors.append({"line": line_number, "error": "row must be an object"})
            continue
        report = validate_decision_state(value)
        if report.valid:
            states.append(value)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    return states, errors


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    states, errors = _read_states(args.input)
    if errors:
        raise SystemExit(json.dumps({"invalid_rows": len(errors), "examples": errors[:5]}, indent=2))
    model, metadata = train_router(
        states,
        config=RouterConfig(
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_router(str(args.output), model, metadata)
    print(json.dumps({"output": str(args.output), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
