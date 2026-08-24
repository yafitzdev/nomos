"""Production-facing orchestration around a candidate-ranking model."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

from .call_validation import ToolCallValidation, validate_tool_call
from .tool_registry import ToolRegistry, ToolSpec


COPROCESSOR_VERSION = "nomos-coprocessor.v2"
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(value: Any) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(str(value or "")) if len(token) > 2}


def score_diagnostics(scores: list[float]) -> dict[str, float]:
    """Return raw ranking diagnostics used by the fitted confidence layer."""

    if not scores:
        return {
            "raw_top_probability": 0.0,
            "probability_margin": 0.0,
            "normalized_entropy": 1.0,
            "log_candidate_count": 0.0,
            "top_score": 0.0,
            "score_margin": 0.0,
            "top3_mean": 0.0,
            "score_standard_deviation": 0.0,
        }
    offset = max(scores)
    exponentials = [math.exp(min(80.0, score - offset)) for score in scores]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    ordered = sorted(probabilities, reverse=True)
    ordered_scores = sorted(scores, reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    if len(probabilities) == 1:
        entropy = 0.0
    else:
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
        entropy /= math.log(len(probabilities))
    score_mean = sum(scores) / len(scores)
    return {
        "raw_top_probability": ordered[0],
        "probability_margin": margin,
        "normalized_entropy": entropy,
        "log_candidate_count": math.log1p(len(scores)),
        "top_score": ordered_scores[0],
        "score_margin": ordered_scores[0] - (ordered_scores[1] if len(scores) > 1 else 0.0),
        "top3_mean": sum(ordered_scores[:3]) / min(3, len(ordered_scores)),
        "score_standard_deviation": math.sqrt(
            sum((score - score_mean) ** 2 for score in scores) / len(scores)
        ),
    }


def _calibrated_confidence(
    diagnostics: Mapping[str, float], calibration: Mapping[str, Any] | None
) -> float | None:
    if not calibration or calibration.get("method") != "logistic.v1":
        return None
    coefficients = calibration.get("coefficients")
    if not isinstance(coefficients, Mapping):
        return None
    value = float(calibration.get("intercept", 0.0))
    for name, coefficient in coefficients.items():
        value += float(coefficient) * float(diagnostics.get(str(name), 0.0))
    return 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, value))))


def _candidate_reasons(
    request: Mapping[str, Any], tool: ToolSpec, rank: int, *, recovery: bool
) -> dict[str, Any]:
    question_tokens = _tokens(request.get("question"))
    semantic_tokens = _tokens(
        " ".join(
            (
                tool.description,
                *tool.capabilities,
                *tool.input_modalities,
                *tool.output_modalities,
                *tool.evidence_roles,
            )
        )
    )
    source_state = request.get("source_state") or {}
    available = {
        str(value) for value in source_state.get("available_modalities") or []
    } if isinstance(source_state, Mapping) else set()
    governance = request.get("governance") or {}
    allowed = {
        str(value) for value in governance.get("allowed_side_effect_classes") or []
    } if isinstance(governance, Mapping) else set()
    reason_codes = ["legal_candidate", "router_ranked"]
    if recovery:
        reason_codes.append("not_previously_rejected")
    if available & set(tool.input_modalities):
        reason_codes.append("input_modality_available")
    if not allowed or tool.side_effect_class in allowed:
        reason_codes.append("side_effect_allowed")
    return {
        "rank": rank,
        "reason_codes": reason_codes,
        "matched_terms": sorted(question_tokens & semantic_tokens)[:12],
        "capabilities": list(tool.capabilities),
        "side_effect_class": tool.side_effect_class,
    }


def _schema_placeholder(name: str, schema: Mapping[str, Any]) -> Any:
    value_type = str(schema.get("type") or "value")
    if value_type == "string":
        return f"<string value for {name}>"
    if value_type == "integer":
        return f"<integer value for {name}>"
    if value_type == "number":
        return f"<number value for {name}>"
    if value_type == "boolean":
        return f"<boolean value for {name}>"
    if value_type == "array":
        return []
    if value_type == "object":
        return {}
    return f"<value for {name}>"


def validation_repair_guidance(
    registry: ToolRegistry, validation: ToolCallValidation
) -> dict[str, Any] | None:
    """Describe a safe retry shape without accepting or executing a bad call."""

    if validation.valid:
        return None
    tool = registry.by_id.get(validation.tool_id)
    if not validation.repairable or tool is None:
        return {
            "strategy": "choose_different_tool",
            "tool_id": validation.tool_id or None,
            "reason_codes": list(validation.failure_reasons),
        }
    properties = tool.argument_schema.get("properties") or {}
    properties = properties if isinstance(properties, Mapping) else {}
    required = [str(value) for value in tool.argument_schema.get("required") or []]
    allowed = [str(value) for value in properties]
    argument_shape = {
        name: _schema_placeholder(
            name,
            properties.get(name) if isinstance(properties.get(name), Mapping) else {},
        )
        for name in required
    }
    return {
        "strategy": "repair_same_tool_call",
        "tool_id": tool.tool_id,
        "required_argument_names": required,
        "allowed_argument_names": allowed,
        "additional_arguments_allowed": tool.argument_schema.get("additionalProperties")
        is not False,
        "call_shape": {"tool_id": tool.tool_id, "arguments": argument_shape},
        "reason_codes": list(validation.failure_reasons),
        "warning": "Replace every placeholder with a real value; this is guidance, not an accepted call.",
    }


def coprocessor_response(
    request: Mapping[str, Any],
    ranked: Iterable[Mapping[str, Any]],
    *,
    router_version: str,
    calibration: Mapping[str, Any] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Apply deterministic safety and recovery policy to model rankings."""

    registry = ToolRegistry.from_dict(request["tool_registry"])
    operation = str(request.get("operation") or "")
    if not operation:
        operation = (
            "verify_tool_call"
            if request.get("proposed_tool_call") is not None
            else "request_more_tool_candidates"
            if request.get("task_kind") == "recover"
            else "recommend_tools"
        )

    base = {
        "schema_version": "router-response.v2",
        "coprocessor_version": COPROCESSOR_VERSION,
        "request_id": str(request["request_id"]),
        "operation": operation,
        "runner": {
            "name": "nomos-router-contract",
            "version": "router-contract.v2",
            "router_version": router_version,
        },
    }
    if operation == "verify_tool_call":
        proposed = request.get("proposed_tool_call")
        if not isinstance(proposed, Mapping):
            raise ValueError("verify_tool_call requires proposed_tool_call")
        validation = validate_tool_call(registry, request, proposed)
        repair = validation_repair_guidance(registry, validation)
        return {
            **base,
            "action": "accept_tool_call" if validation.valid else "reject_tool_call",
            "selected_tool": validation.tool_id if validation.valid else None,
            "ranked_tools": [],
            "recommendations": [],
            "confidence": {
                "value": 1.0,
                "calibrated": True,
                "method": "deterministic_contract_validation",
                "uncertainty": 0.0,
            },
            "validation": validation.as_dict(),
            "repair": repair,
            "reason_codes": list(validation.checked if validation.valid else validation.failure_reasons),
        }

    legal_ids = {str(value) for value in request.get("legal_candidate_ids") or []}
    prior_ids = (
        {str(value) for value in request.get("previous_candidate_ids") or []}
        if operation == "request_more_tool_candidates"
        else set()
    )
    filtered = [
        dict(item)
        for item in ranked
        if str(item.get("tool_id")) in legal_ids and str(item.get("tool_id")) not in prior_ids
    ]
    if not filtered:
        return {
            **base,
            "action": "request_more_tool_candidates",
            "selected_tool": None,
            "ranked_tools": [],
            "recommendations": [],
            "confidence": {
                "value": None,
                "calibrated": False,
                "method": "no_eligible_candidates",
                "uncertainty": 1.0,
            },
            "reason_codes": ["candidate_set_exhausted", "registry_expansion_required"],
            "excluded_tool_ids": sorted(prior_ids),
        }

    scores = [float(item.get("score", 0.0)) for item in filtered]
    diagnostics = score_diagnostics(scores)
    confidence = _calibrated_confidence(diagnostics, calibration)
    threshold = float((calibration or {}).get("abstention_threshold", 0.0))
    should_abstain = confidence is not None and confidence < threshold
    recommendations = []
    for rank, item in enumerate(filtered[: max(1, top_k)], start=1):
        tool = registry.require(str(item["tool_id"]))
        recommendations.append(
            {
                **item,
                "reasons": _candidate_reasons(
                    request,
                    tool,
                    rank,
                    recovery=operation == "request_more_tool_candidates",
                ),
            }
        )
    return {
        **base,
        "action": "abstain" if should_abstain else "recommend_tools",
        "selected_tool": None if should_abstain else recommendations[0]["tool_id"],
        # Kept for backward-compatible runner evaluators. New clients should consume recommendations.
        "ranked_tools": filtered,
        "recommendations": [] if should_abstain else recommendations,
        "confidence": {
            "value": confidence,
            "calibrated": confidence is not None,
            "method": (calibration or {}).get("method", "uncalibrated_softmax_diagnostics"),
            "uncertainty": (
                1.0 - confidence
                if confidence is not None
                else diagnostics["normalized_entropy"]
            ),
            "diagnostics": diagnostics,
        },
        "reason_codes": (
            ["confidence_below_abstention_threshold"]
            if should_abstain
            else [
                "top_candidates_ranked",
                "previous_candidates_excluded"
                if operation == "request_more_tool_candidates"
                else "initial_candidate_set",
            ]
        ),
        "excluded_tool_ids": sorted(prior_ids),
    }
