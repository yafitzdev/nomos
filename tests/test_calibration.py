from __future__ import annotations

from fitz_tool.calibration import (
    calibration_metrics,
    choose_abstention_threshold,
    fit_logistic_calibration,
)
from fitz_tool.coprocessor import score_diagnostics


def test_calibration_fits_and_selects_low_risk_subset() -> None:
    records = []
    for index in range(40):
        correct = index >= 15
        scores = [4.0, 0.0, -1.0] if correct else [0.2, 0.1, 0.0]
        records.append({"correct": correct, "features": score_diagnostics(scores)})
    calibration = fit_logistic_calibration(records)
    selection = choose_abstention_threshold(
        records, calibration, maximum_selective_risk=0.01
    )
    calibration["abstention_threshold"] = selection["threshold"]
    metrics = calibration_metrics(records, calibration)
    assert selection["coverage"] > 0.5
    assert metrics["selective_risk"] <= 0.01
