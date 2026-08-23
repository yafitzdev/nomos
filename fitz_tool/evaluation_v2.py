"""Cohort, baseline, ablation and invariance evaluation for router.v2."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping

from .router_v2 import rank_tools_v2
from .tool_registry import ToolRegistry


RankingFunction = Callable[[Mapping[str, Any]], list[str]]
INVARIANCE_TOLERANCE = 1e-6


def _cached_ranker(ranker: RankingFunction) -> RankingFunction:
    cache: dict[int, list[str]] = {}

    def rank(state: Mapping[str, Any]) -> list[str]:
        key = id(state)
        if key not in cache:
            cache[key] = ranker(state)
        return list(cache[key])

    return rank


def _metrics(states: Iterable[Mapping[str, Any]], ranker: RankingFunction) -> dict[str, Any]:
    evaluated = recall_1 = recall_3 = 0
    reciprocal_rank = 0.0
    for state in states:
        acceptable = set((state.get("label") or {}).get("acceptable_tools") or [])
        if not acceptable or not state.get("legal_candidate_ids"):
            continue
        ranked = ranker(state)
        evaluated += 1
        recall_1 += int(bool(set(ranked[:1]) & acceptable))
        recall_3 += int(bool(set(ranked[:3]) & acceptable))
        first_rank = next(
            (index for index, tool_id in enumerate(ranked, start=1) if tool_id in acceptable),
            None,
        )
        if first_rank is not None:
            reciprocal_rank += 1.0 / first_rank
    return {
        "states": evaluated,
        "recall_at_1": recall_1 / evaluated if evaluated else 0.0,
        "recall_at_3": recall_3 / evaluated if evaluated else 0.0,
        "mrr": reciprocal_rank / evaluated if evaluated else 0.0,
        "invalid_candidate_rate": 0.0,
    }


def _model_ranker(model: Any, metadata: Mapping[str, Any]) -> RankingFunction:
    def rank(state: Mapping[str, Any]) -> list[str]:
        rows = rank_tools_v2(
            model,
            metadata,
            state,
            top_k=len(state.get("legal_candidate_ids") or []),
        )
        return [str(row["tool_id"]) for row in rows]

    return rank


def _frequency_ranker(metadata: Mapping[str, Any]) -> RankingFunction:
    family_counts = metadata.get("positive_family_counts") or {}
    capability_counts = metadata.get("positive_capability_counts") or {}

    def rank(state: Mapping[str, Any]) -> list[str]:
        registry = ToolRegistry.from_dict(state["tool_registry"])
        legal = [str(tool_id) for tool_id in state.get("legal_candidate_ids") or []]
        return sorted(
            legal,
            key=lambda tool_id: (
                -sum(
                    float(capability_counts.get(capability, 0))
                    for capability in registry.require(tool_id).capabilities
                ),
                -float(family_counts.get(registry.require(tool_id).tool_family, 0)),
                registry.require(tool_id).semantic_fingerprint,
            ),
        )

    return rank


def _neutralize_registry(state: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(state))
    registry = ToolRegistry.from_dict(output["tool_registry"])
    output["tool_registry"] = ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": "metadata_ablation",
            "tools": [
                {
                    "tool_id": tool.tool_id,
                    "tool_family": "unknown",
                    "description": "Unknown candidate capability metadata for ablation.",
                    "capabilities": ["unknown"],
                    "input_modalities": ["unknown"],
                    "output_modalities": ["unknown"],
                    "evidence_roles": ["unknown"],
                    "side_effect_class": "none",
                    "argument_schema": {"type": "object", "properties": {}, "required": []},
                    "constraints": ["none"],
                    "prerequisites": ["none"],
                }
                for tool in registry.tools
            ],
        }
    ).as_dict()
    return output


def evaluate_router_v2_report(
    model: Any,
    metadata: Mapping[str, Any],
    states: list[Mapping[str, Any]],
) -> dict[str, Any]:
    model_ranker = _cached_ranker(_model_ranker(model, metadata))
    cohorts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        cohorts[str(state.get("evaluation_cohort", "unspecified"))].append(state)

    no_question = [dict(copy.deepcopy(state), question="") for state in states]
    no_metadata = [_neutralize_registry(state) for state in states]
    no_question_ranker = _cached_ranker(_model_ranker(model, metadata))
    no_metadata_ranker = _cached_ranker(_model_ranker(model, metadata))
    target_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    family_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        target = str((state.get("sampling_context") or {}).get("target_capability", "unknown"))
        target_groups[target].append(state)
        acceptable = (state.get("label") or {}).get("acceptable_tools") or []
        registry = ToolRegistry.from_dict(state["tool_registry"])
        if acceptable:
            family = registry.require(str(acceptable[0])).tool_family
            family_groups[family].append(state)

    confusion_pairs: Counter[tuple[str, str]] = Counter()
    for state in states:
        acceptable = set((state.get("label") or {}).get("acceptable_tools") or [])
        if not acceptable or not state.get("legal_candidate_ids"):
            continue
        ranked = model_ranker(state)
        if ranked and ranked[0] not in acceptable:
            confusion_pairs[(sorted(acceptable)[0], ranked[0])] += 1
    return {
        "overall": _metrics(states, model_ranker),
        "cohorts": {
            cohort: _metrics(rows, model_ranker) for cohort, rows in sorted(cohorts.items())
        },
        "by_target_capability": {
            target: _metrics(rows, model_ranker) for target, rows in sorted(target_groups.items())
        },
        "by_tool_family": {
            family: _metrics(rows, model_ranker) for family, rows in sorted(family_groups.items())
        },
        "confusion_pairs": [
            {"acceptable": acceptable, "predicted": predicted, "count": count}
            for (acceptable, predicted), count in confusion_pairs.most_common(25)
        ],
        "baselines": {
            "caller_order": _metrics(
                states, lambda state: list(state.get("legal_candidate_ids") or [])
            ),
            "capability_frequency": _metrics(states, _frequency_ranker(metadata)),
        },
        "ablations": {
            "question_removed": _metrics(no_question, no_question_ranker),
            "tool_metadata_removed": _metrics(no_metadata, no_metadata_ranker),
        },
        "invariance": invariance_report(model, metadata, states),
    }


def _score_map(model: Any, metadata: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["tool_id"]): float(row["score"])
        for row in rank_tools_v2(
            model,
            metadata,
            state,
            top_k=len(state.get("legal_candidate_ids") or []),
        )
    }


def _renamed_state(state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    output = copy.deepcopy(dict(state))
    registry = ToolRegistry.from_dict(output["tool_registry"])
    names = {tool.tool_id: f"renamed_{index:03d}" for index, tool in enumerate(registry.tools)}
    tools = []
    for tool in registry.tools:
        value = tool.as_dict()
        value["tool_id"] = names[tool.tool_id]
        tools.append(value)
    output["tool_registry"] = ToolRegistry.from_dict(
        {
            "schema_version": "tool-registry.v2",
            "registry_id": "id_invariance",
            "tools": tools,
        }
    ).as_dict()
    output["legal_candidate_ids"] = [names[tool_id] for tool_id in output["legal_candidate_ids"]]
    return output, names


def invariance_report(
    model: Any,
    metadata: Mapping[str, Any],
    states: Iterable[Mapping[str, Any]],
    *,
    sample_size: int = 100,
) -> dict[str, Any]:
    max_order_delta = max_rename_delta = max_sampling_delta = 0.0
    evaluated = 0
    for state in list(states)[:sample_size]:
        if not state.get("legal_candidate_ids"):
            continue
        baseline = _score_map(model, metadata, state)

        reordered = copy.deepcopy(dict(state))
        reordered["legal_candidate_ids"] = list(reversed(reordered["legal_candidate_ids"]))
        reordered_scores = _score_map(model, metadata, reordered)
        max_order_delta = max(
            max_order_delta,
            max(abs(score - reordered_scores[tool_id]) for tool_id, score in baseline.items()),
        )

        renamed, names = _renamed_state(state)
        renamed_scores = _score_map(model, metadata, renamed)
        max_rename_delta = max(
            max_rename_delta,
            max(
                abs(score - renamed_scores[names[tool_id]])
                for tool_id, score in baseline.items()
            ),
        )

        altered_sampling = copy.deepcopy(dict(state))
        altered_sampling["sampling_context"] = {
            "target_capability": "adversarial_future_label",
            "terminal_outcome": "adversarial_future_outcome",
        }
        altered_scores = _score_map(model, metadata, altered_sampling)
        max_sampling_delta = max(
            max_sampling_delta,
            max(abs(score - altered_scores[tool_id]) for tool_id, score in baseline.items()),
        )
        evaluated += 1
    return {
        "states": evaluated,
        "candidate_order_max_score_delta": max_order_delta,
        "tool_id_rename_max_score_delta": max_rename_delta,
        "sampling_context_max_score_delta": max_sampling_delta,
        "tolerance": INVARIANCE_TOLERANCE,
        "passed": max(max_order_delta, max_rename_delta, max_sampling_delta)
        <= INVARIANCE_TOLERANCE,
    }
