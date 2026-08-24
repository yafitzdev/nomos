from __future__ import annotations

from tools.evaluate_dense_confidence import _metrics


def test_confidence_metrics_keep_ranking_and_abstention_separate() -> None:
    calibration = {
        "intercept": 0.0,
        "coefficients": {},
        "abstention_threshold": 0.5,
    }
    records = [
        {
            "answer_present": True,
            "correct": True,
            "first_rank": 1,
            "positive_margin": 0.4,
            "features": {},
        },
        {
            "answer_present": True,
            "correct": True,
            "first_rank": 2,
            "positive_margin": -0.1,
            "features": {},
        },
        {
            "answer_present": False,
            "correct": False,
            "first_rank": None,
            "positive_margin": None,
            "features": {},
        },
    ]

    metrics = _metrics(records, calibration)

    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["mean_positive_margin"] == 0.15000000000000002
    assert metrics["answer_present"] == 2
    assert metrics["no_suitable_tool"] == 1
