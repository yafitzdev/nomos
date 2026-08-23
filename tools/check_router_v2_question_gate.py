"""Apply the router.v2 question-generalization and regression gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


REGRESSION_COHORTS = (
    "unseen_tool_ids",
    "id_renames",
    "heldout_family",
    "alternate_registry",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-frozen", type=Path, required=True)
    parser.add_argument("--candidate-frozen", type=Path, required=True)
    parser.add_argument("--candidate-generalized", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--question-r1-min", type=float, default=0.70)
    parser.add_argument("--max-r1-drop", type=float, default=0.05)
    return parser


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _r1(report: dict[str, Any], section: str, name: str | None = None) -> float:
    value = report[section] if name is None else report[section][name]
    return float(value["recall_at_1"])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = _read(args.baseline_frozen)
    candidate_frozen = _read(args.candidate_frozen)
    candidate_generalized = _read(args.candidate_generalized)
    checks: dict[str, Any] = {}
    question_r1 = _r1(candidate_generalized, "cohorts", "heldout_questions")
    checks["heldout_questions_r1"] = {
        "actual": question_r1,
        "minimum": args.question_r1_min,
        "passed": question_r1 >= args.question_r1_min,
    }
    for cohort in REGRESSION_COHORTS:
        baseline_value = _r1(baseline, "cohorts", cohort)
        candidate_value = _r1(candidate_frozen, "cohorts", cohort)
        checks[f"cohort_{cohort}"] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "drop": baseline_value - candidate_value,
            "maximum_drop": args.max_r1_drop,
            "passed": candidate_value >= baseline_value - args.max_r1_drop,
        }
    baseline_metadata = _r1(baseline, "ablations", "tool_metadata_removed")
    candidate_metadata = _r1(candidate_frozen, "ablations", "tool_metadata_removed")
    checks["metadata_ablation"] = {
        "baseline": baseline_metadata,
        "candidate": candidate_metadata,
        "drop": baseline_metadata - candidate_metadata,
        "maximum_drop": args.max_r1_drop,
        "passed": candidate_metadata >= baseline_metadata - args.max_r1_drop,
    }
    checks["overall_frozen"] = {
        "baseline": _r1(baseline, "overall"),
        "candidate": _r1(candidate_frozen, "overall"),
        "maximum_drop": args.max_r1_drop,
        "passed": _r1(candidate_frozen, "overall")
        >= _r1(baseline, "overall") - args.max_r1_drop,
    }
    checks["invalid_candidate_rate"] = {
        "frozen": candidate_frozen.get("overall", {}).get("invalid_candidate_rate"),
        "generalized": candidate_generalized.get("overall", {}).get("invalid_candidate_rate"),
        "passed": (
            candidate_frozen.get("overall", {}).get("invalid_candidate_rate") == 0.0
            and candidate_generalized.get("overall", {}).get("invalid_candidate_rate") == 0.0
        ),
    }
    checks["frozen_invariance"] = {
        "passed": bool(candidate_frozen.get("invariance", {}).get("passed"))
    }
    checks["generalized_invariance"] = {
        "passed": bool(candidate_generalized.get("invariance", {}).get("passed"))
    }
    valid = all(bool(value.get("passed")) for value in checks.values())
    report = {
        "valid": valid,
        "question_r1_min": args.question_r1_min,
        "max_r1_drop": args.max_r1_drop,
        "checks": checks,
        "recommend_scale": valid,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
