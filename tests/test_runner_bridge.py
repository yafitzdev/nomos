from __future__ import annotations

from tools.run_v2_runner import _governance


def test_runner_marks_terminal_state_fresh_after_assessment() -> None:
    result = {
        "metadata": {
            "governance_trajectory": [{"step_number": 5, "verdict": "SUFFICIENT"}],
            "requirement_progress": [],
        }
    }
    assert _governance(result, 5, prior_actions=["update_requirement_progress"])[
        "assessment_fresh"
    ] is False
    assert _governance(result, 5, prior_actions=["assess_evidence"])[
        "assessment_fresh"
    ] is True
