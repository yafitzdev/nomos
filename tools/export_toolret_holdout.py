"""Export deterministic ToolRet query holdouts for training exclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.evaluate_toolret import TASK_CATEGORY, _sample_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--limit-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/toolret-cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from datasets import load_dataset

    args = build_parser().parse_args(argv)
    output = []
    for task_index, task in enumerate(args.task):
        if task not in TASK_CATEGORY:
            raise SystemExit(f"unknown ToolRet task: {task}")
        dataset = load_dataset(
            "mangopy/ToolRet-Queries",
            task,
            split="queries",
            cache_dir=str(args.cache_dir),
        )
        for row in _sample_rows(dataset, args.limit_per_task, args.seed + task_index):
            output.append(
                {
                    "task": task,
                    "query_id": str(row["id"]),
                    "query": str(row["query"]),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "queries": len(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
