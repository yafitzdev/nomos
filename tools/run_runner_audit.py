"""Run a stratified scenario audit through an external runner.v1 process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

from fitz_tool.contracts import validate_scenario, validate_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--runner-command",
        nargs=argparse.REMAINDER,
        required=True,
        help=(
            "Executable and arguments for a process implementing runner.v1 "
            "stdin/stdout JSONL; place this option last."
        ),
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.runner_command:
        raise SystemExit("--runner-command requires an executable")
    scenarios = _read_jsonl(args.scenarios)
    manifest = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    selected_indices = [int(item["row_index"]) for item in manifest.get("rows", [])]
    selected = [scenarios[index] for index in selected_indices]
    for scenario in selected:
        report = validate_scenario(scenario)
        if not report.valid:
            raise SystemExit(json.dumps(report.as_dict(), indent=2))
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    environment = dict(os.environ)
    environment["FITZ_TOOL_RUNNER_CONTRACT"] = "runner.v1"
    completed = subprocess.run(
        args.runner_command,
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=args.timeout,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"runner exited {completed.returncode}: {completed.stderr[-4000:]}"
        )
    traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            trace = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": str(exc)})
            continue
        report = validate_trajectory(trace)
        if report.valid:
            traces.append(trace)
        else:
            errors.append({"line": line_number, "validation": report.as_dict()})
    expected_ids = [str(row["scenario_id"]) for row in selected]
    actual_ids = [str(trace.get("scenario_id")) for trace in traces]
    if actual_ids != expected_ids:
        errors.append({"error": "runner output IDs/order do not match selected scenarios"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "requested": len(selected),
                "traces": len(traces),
                "errors": len(errors),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        args.output.with_suffix(".errors.jsonl").write_text(
            "".join(json.dumps(error, sort_keys=True) + "\n" for error in errors),
            encoding="utf-8",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
