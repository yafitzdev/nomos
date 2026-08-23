"""Train and evaluate the registry-aware router.v2 encoder."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.generic_contracts import validate_decision_state_v2
from fitz_tool.pilot_v2 import validate_pilot_state
from fitz_tool.router_v2 import RouterV2Config, save_router_v2, train_router_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Use only this many deterministic training states while retaining all frozen evaluation rows.",
    )
    return parser


def _read_states(path: Path) -> list[dict[str, Any]]:
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
        report = validate_pilot_state(value) if value.get("pilot_version") else validate_decision_state_v2(value)
        if report.valid:
            states.append(value)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    if errors:
        raise SystemExit(json.dumps({"invalid_rows": len(errors), "examples": errors[:5]}, indent=2))
    return states


def _balanced_training_subset(states: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        target = str((state.get("sampling_context") or {}).get("target_capability", "unknown"))
        groups[target].append(state)
    selected: list[dict[str, Any]] = []
    positions = {target: 0 for target in groups}
    targets = sorted(groups)
    while len(selected) < count:
        progressed = False
        for target in targets:
            position = positions[target]
            if position >= len(groups[target]):
                continue
            selected.append(groups[target][position])
            positions[target] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    states = _read_states(args.input)
    if args.train_count is not None:
        if args.train_count < 1:
            raise SystemExit("--train-count must be positive")
        train_rows = [state for state in states if state.get("evaluation_partition") == "train"]
        if args.train_count > len(train_rows):
            raise SystemExit(f"--train-count exceeds available training rows: {len(train_rows)}")
        states = _balanced_training_subset(train_rows, args.train_count) + [
            state for state in states if state.get("evaluation_partition") != "train"
        ]
    model, metadata = train_router_v2(
        states,
        config=RouterV2Config(
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_router_v2(str(args.output), model, metadata)
    print(json.dumps({"output": str(args.output), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
