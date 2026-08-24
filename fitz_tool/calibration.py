"""Calibration and selective-prediction utilities for Nomos top-k routing."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


CALIBRATION_FEATURES = (
    "raw_top_probability",
    "probability_margin",
    "normalized_entropy",
    "log_candidate_count",
    "top_score",
    "score_margin",
    "top3_mean",
    "score_standard_deviation",
)


def predict_confidence(calibration: Mapping[str, Any], features: Mapping[str, float]) -> float:
    value = float(calibration["intercept"])
    coefficients = calibration["coefficients"]
    for name, coefficient in coefficients.items():
        value += float(coefficient) * float(features.get(str(name), 0.0))
    return 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, value))))


def fit_logistic_calibration(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    if not records or len({int(record["correct"]) for record in records}) < 2:
        raise ValueError("calibration requires both correct and incorrect validation examples")
    features = [
        [float(record["features"][name]) for name in CALIBRATION_FEATURES]
        for record in records
    ]
    labels = [int(record["correct"]) for record in records]
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=0)
    model.fit(features, labels)
    return {
        "method": "logistic.v1",
        "target": "top3_contains_acceptable_tool",
        "intercept": float(model.intercept_[0]),
        "coefficients": {
            name: float(value) for name, value in zip(CALIBRATION_FEATURES, model.coef_[0])
        },
        "fitted_examples": len(records),
    }


def choose_abstention_threshold(
    records: list[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    *,
    maximum_selective_risk: float,
) -> dict[str, float | int]:
    scored = sorted(
        (
            predict_confidence(calibration, record["features"]),
            int(record["correct"]),
        )
        for record in records
    )
    best = {"threshold": 1.0, "coverage": 0.0, "selective_risk": 0.0, "selected": 0}
    for threshold in sorted({score for score, _correct in scored}):
        selected = [correct for score, correct in scored if score >= threshold]
        if not selected:
            continue
        risk = 1.0 - sum(selected) / len(selected)
        coverage = len(selected) / len(scored)
        if risk <= maximum_selective_risk and coverage >= float(best["coverage"]):
            best = {
                "threshold": threshold,
                "coverage": coverage,
                "selective_risk": risk,
                "selected": len(selected),
            }
    return best


def calibration_metrics(
    records: Iterable[Mapping[str, Any]], calibration: Mapping[str, Any], *, bins: int = 10
) -> dict[str, float | int]:
    rows = [
        (predict_confidence(calibration, record["features"]), int(record["correct"]))
        for record in records
    ]
    if not rows:
        return {"examples": 0, "accuracy": 0.0, "brier": 0.0, "ece": 0.0}
    brier = sum((confidence - correct) ** 2 for confidence, correct in rows) / len(rows)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [row for row in rows if lower <= row[0] < upper or (index == bins - 1 and row[0] == 1.0)]
        if bucket:
            average_confidence = sum(row[0] for row in bucket) / len(bucket)
            accuracy = sum(row[1] for row in bucket) / len(bucket)
            ece += len(bucket) / len(rows) * abs(average_confidence - accuracy)
    threshold = float(calibration.get("abstention_threshold", 0.0))
    selected = [correct for confidence, correct in rows if confidence >= threshold]
    return {
        "examples": len(rows),
        "accuracy": sum(correct for _confidence, correct in rows) / len(rows),
        "brier": brier,
        "ece": ece,
        "coverage": len(selected) / len(rows),
        "selective_risk": 1.0 - sum(selected) / len(selected) if selected else 0.0,
    }
