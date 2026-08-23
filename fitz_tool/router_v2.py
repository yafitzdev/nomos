"""Registry-aware candidate ranker that does not learn literal tool identities."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .generic_contracts import observable_router_state
from .tool_registry import ToolRegistry, ToolSpec


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
ROUTER_VERSION = "router.v2"
FEATURE_VERSION = "registry-features.v2"


@dataclass(frozen=True)
class RouterV2Config:
    feature_dim: int = 4096
    hidden_dim: int = 128
    epochs: int = 20
    learning_rate: float = 2e-3
    seed: int = 20260823

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }


def _tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        return [token.casefold() for token in TOKEN_RE.findall(value)]
    if isinstance(value, Mapping):
        output: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            output.extend([str(key).casefold(), *_tokens(item)])
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_tokens(item))
        return output
    if value is None:
        return []
    return [str(value).casefold()]


def _schema_tokens(schema: Mapping[str, Any], prefix: str = "argument") -> list[str]:
    output: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        output.append(f"{prefix}_type={schema_type}")
    required = set(schema.get("required") or [])
    properties = schema.get("properties") or {}
    if isinstance(properties, Mapping):
        for name, value in sorted(properties.items(), key=lambda pair: str(pair[0])):
            property_prefix = f"{prefix}_property={str(name).casefold()}"
            output.append(property_prefix)
            output.append(f"{property_prefix}|required={str(name in required).lower()}")
            if isinstance(value, Mapping):
                property_type = value.get("type")
                if isinstance(property_type, str):
                    output.append(f"{property_prefix}|type={property_type}")
                enum = value.get("enum")
                if isinstance(enum, list):
                    output.extend(f"{property_prefix}|enum={token}" for token in _tokens(enum))
                items = value.get("items")
                if isinstance(items, Mapping) and isinstance(items.get("type"), str):
                    output.append(f"{property_prefix}|item_type={items['type']}")
    return output


def tool_semantic_tokens(tool: ToolSpec) -> list[str]:
    """Represent one candidate without exposing its concrete tool_id."""

    tokens = [f"candidate_family={tool.tool_family}"]
    tokens.extend(f"candidate_description={token}" for token in _tokens(tool.description))
    tokens.extend(f"candidate_capability={value}" for value in tool.capabilities)
    tokens.extend(f"candidate_input_modality={value}" for value in tool.input_modalities)
    tokens.extend(f"candidate_output_modality={value}" for value in tool.output_modalities)
    tokens.extend(f"candidate_evidence_role={value}" for value in tool.evidence_roles)
    tokens.append(f"candidate_side_effect={tool.side_effect_class}")
    tokens.extend(f"candidate_constraint={value}" for value in tool.constraints)
    tokens.extend(f"candidate_prerequisite={value}" for value in tool.prerequisites)
    tokens.extend(_schema_tokens(tool.argument_schema))
    return tokens


def _state_value(state: Mapping[str, Any], parent: str, key: str, default: str) -> str:
    value = state.get(parent) or {}
    if not isinstance(value, Mapping):
        return default
    item = value.get(key, default)
    if isinstance(item, list):
        return "+".join(sorted(str(part) for part in item)) or default
    return str(item)


def state_candidate_tokens(
    state: Mapping[str, Any],
    candidate: ToolSpec,
    legal_candidates: Iterable[ToolSpec],
) -> list[str]:
    """Encode observable state and semantic candidate metadata.

    `sampling_context`, `matrix_context`, labels, future governance paths and
    terminal outcomes are intentionally ignored.
    """

    observable = observable_router_state(state)
    tokens: list[str] = []
    question_tokens = _tokens(observable.get("question"))
    tokens.extend(f"question={token}" for token in question_tokens)
    for field in (
        "agent_state",
        "history",
        "plan",
        "observed_evidence",
        "governance",
        "resource_state",
        "source_state",
        "query_state",
    ):
        tokens.extend(f"state_{field}={token}" for token in _tokens(observable.get(field)))

    tokens.extend(tool_semantic_tokens(candidate))

    legal_candidates = tuple(legal_candidates)
    tokens.append(f"candidate_set_size={len(legal_candidates)}")
    for tool in legal_candidates:
        tokens.append(f"candidate_set_family={tool.tool_family}")
        tokens.extend(f"candidate_set_capability={value}" for value in tool.capabilities)
        tokens.append(f"candidate_set_side_effect={tool.side_effect_class}")

    interaction_context = {
        "agent_state": _state_value(observable, "agent_state", "state_name", "unknown"),
        "agent_phase": _state_value(observable, "agent_state", "phase", "unknown"),
        "operation": _state_value(observable, "query_state", "operation", "unknown"),
        "match_strategy": _state_value(
            observable, "query_state", "match_strategy", "unknown"
        ),
        "query_specificity": _state_value(
            observable, "query_state", "specificity", "unknown"
        ),
        "inventory_state": _state_value(
            observable, "source_state", "inventory_state", "unknown"
        ),
        "inspection_state": _state_value(
            observable, "source_state", "inspection_state", "unknown"
        ),
        "available_modalities": _state_value(
            observable, "source_state", "available_modalities", "unknown"
        ),
        "assessment_fresh": _state_value(
            observable, "governance", "assessment_fresh", "unknown"
        ),
        "remaining_steps": _state_value(
            observable, "resource_state", "remaining_steps", "unknown"
        ),
    }
    for capability in candidate.capabilities:
        for key, value in interaction_context.items():
            tokens.append(f"capability_context={capability}|{key}={value}")
        for question_token in sorted(set(question_tokens))[:64]:
            tokens.append(f"capability_question={capability}|{question_token}")

    description_terms = set(_tokens(candidate.description))
    for overlap in sorted(description_terms & set(question_tokens)):
        tokens.append(f"question_description_overlap={overlap}")

    available_modalities = set(
        _tokens((observable.get("source_state") or {}).get("available_modalities"))
        if isinstance(observable.get("source_state"), Mapping)
        else []
    )
    if available_modalities:
        matches = bool(available_modalities & set(candidate.input_modalities))
        tokens.append(f"candidate_modality_match={str(matches).lower()}")

    allowed_side_effects = set(
        (observable.get("governance") or {}).get("allowed_side_effect_classes") or []
    )
    if allowed_side_effects:
        tokens.append(
            "candidate_side_effect_allowed="
            + str(candidate.side_effect_class in allowed_side_effects).lower()
        )
    return tokens


def featurize_v2(
    state: Mapping[str, Any],
    candidate: ToolSpec,
    legal_candidates: Iterable[ToolSpec],
    feature_dim: int,
) -> list[float]:
    if feature_dim < 16:
        raise ValueError("feature_dim must be at least 16")
    vector = [0.0] * feature_dim
    for token in state_candidate_tokens(state, candidate, legal_candidates):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest, "big") % feature_dim] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def split_name_v2(state: Mapping[str, Any]) -> str:
    explicit = state.get("evaluation_partition")
    if explicit in {"train", "validation", "test"}:
        return str(explicit)
    provenance = state.get("provenance") or {}
    if isinstance(provenance, Mapping) and provenance.get("split") in {
        "train",
        "validation",
        "test",
    }:
        return str(provenance["split"])
    group_id = str(
        state.get("trajectory_id")
        or state.get("scenario_id")
        or state.get("decision_state_id")
    )
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _registry(state: Mapping[str, Any]) -> ToolRegistry:
    raw = state.get("tool_registry")
    if not isinstance(raw, Mapping):
        raise ValueError("state must embed a tool_registry")
    return ToolRegistry.from_dict(raw)


def pair_examples_v2(
    states: Iterable[Mapping[str, Any]], *, feature_dim: int
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for state in states:
        registry = _registry(state)
        legal_ids = [str(tool_id) for tool_id in state.get("legal_candidate_ids") or []]
        legal_specs = registry.resolve(legal_ids)
        label = state.get("label") or {}
        acceptable = set(label.get("acceptable_tools") or [])
        hard_negatives = set(label.get("hard_negative_tools") or [])
        accepted = bool(state.get("accepted", bool(acceptable)))
        candidate_ids = legal_ids if accepted else [
            tool_id for tool_id in legal_ids if tool_id in hard_negatives
        ]
        by_id = registry.by_id
        for tool_id in candidate_ids:
            tool = by_id[tool_id]
            examples.append(
                {
                    "state_id": state.get("decision_state_id"),
                    "split": split_name_v2(state),
                    "tool_id": tool_id,
                    "tool_semantic_fingerprint": tool.semantic_fingerprint,
                    "label": float(tool_id in acceptable),
                    "features": featurize_v2(state, tool, legal_specs, feature_dim),
                }
            )
    return examples


def _rank(
    model: Any,
    state: Mapping[str, Any],
    registry: ToolRegistry,
    feature_dim: int,
) -> list[dict[str, Any]]:
    import torch

    legal_ids = [str(tool_id) for tool_id in state.get("legal_candidate_ids") or []]
    if not legal_ids:
        raise ValueError("legal_candidate_ids must not be empty")
    legal_specs = registry.resolve(legal_ids)
    features = torch.tensor(
        [featurize_v2(state, tool, legal_specs, feature_dim) for tool in legal_specs],
        dtype=torch.float32,
    )
    scores = model(features).flatten().tolist()
    rows = [
        {
            "tool_id": tool.tool_id,
            "tool_family": tool.tool_family,
            "semantic_fingerprint": tool.semantic_fingerprint,
            "score": float(score),
        }
        for score, tool in zip(scores, legal_specs)
    ]
    return sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            str(row["semantic_fingerprint"]),
            str(row["tool_id"]),
        ),
    )


def evaluate_router_v2(
    model: Any,
    metadata: Mapping[str, Any],
    states: Iterable[Mapping[str, Any]],
) -> dict[str, float | int]:
    import torch

    evaluated = recall_1 = recall_3 = 0
    reciprocal_rank = 0.0
    feature_dim = int(metadata["config"]["feature_dim"])
    model.eval()
    with torch.no_grad():
        for state in states:
            acceptable = set((state.get("label") or {}).get("acceptable_tools") or [])
            if not acceptable or not state.get("legal_candidate_ids"):
                continue
            ranked = _rank(model, state, _registry(state), feature_dim)
            ranked_ids = [str(row["tool_id"]) for row in ranked]
            evaluated += 1
            recall_1 += int(bool(set(ranked_ids[:1]) & acceptable))
            recall_3 += int(bool(set(ranked_ids[:3]) & acceptable))
            first_rank = next(
                (index for index, tool_id in enumerate(ranked_ids, start=1) if tool_id in acceptable),
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


def train_router_v2(
    states: list[Mapping[str, Any]],
    *,
    config: RouterV2Config = RouterV2Config(),
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn

    if not states:
        raise ValueError("cannot train router.v2 without decision states")
    torch.manual_seed(config.seed)
    examples = pair_examples_v2(states, feature_dim=config.feature_dim)
    positives = sum(example["label"] for example in examples)
    negatives = len(examples) - positives
    if not positives or not negatives:
        raise ValueError("training requires positive and negative candidate labels")

    class RouterV2MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    model = RouterV2MLP()
    train_examples = [row for row in examples if row["split"] == "train"] or examples
    features = torch.tensor(
        [row["features"] for row in train_examples], dtype=torch.float32
    )
    labels = torch.tensor([[row["label"]] for row in train_examples], dtype=torch.float32)
    positive_weight = max(
        1.0,
        (len(train_examples) - labels.sum().item()) / max(1.0, labels.sum().item()),
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for _epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(features), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    metadata: dict[str, Any] = {
        "router_version": ROUTER_VERSION,
        "feature_version": FEATURE_VERSION,
        "config": config.as_dict(),
        "state_count": len(states),
        "pair_count": len(examples),
        "positive_pairs": int(positives),
        "negative_pairs": int(negatives),
        "registry_fingerprints": sorted({_registry(state).fingerprint for state in states}),
        "feature_description": (
            "blake2b hashed observable-state and identity-free tool-registry metadata"
        ),
        "losses": losses,
    }
    positive_family_counts: Counter[str] = Counter()
    positive_capability_counts: Counter[str] = Counter()
    for state in states:
        registry = _registry(state)
        for tool_id in (state.get("label") or {}).get("acceptable_tools") or []:
            tool = registry.require(str(tool_id))
            positive_family_counts[tool.tool_family] += 1
            positive_capability_counts.update(tool.capabilities)
    metadata["positive_family_counts"] = dict(sorted(positive_family_counts.items()))
    metadata["positive_capability_counts"] = dict(
        sorted(positive_capability_counts.items())
    )
    model.eval()
    metadata["metrics"] = {
        split: evaluate_router_v2(
            model,
            metadata,
            [state for state in states if split_name_v2(state) == split],
        )
        for split in ("train", "validation", "test")
    }
    return model, metadata


def save_router_v2(path: str, model: Any, metadata: Mapping[str, Any]) -> None:
    import torch

    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, path)


def load_router_v2(path: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(artifact["metadata"])
    if metadata.get("router_version") != ROUTER_VERSION:
        raise ValueError(f"expected {ROUTER_VERSION} artifact")
    config = RouterV2Config(**metadata["config"])

    class RouterV2MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    model = RouterV2MLP()
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, metadata


def rank_tools_v2(
    model: Any,
    metadata: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    registry: ToolRegistry | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    import torch

    if top_k < 1:
        raise ValueError("top_k must be positive")
    registry = registry or _registry(state)
    feature_dim = int(metadata["config"]["feature_dim"])
    model.eval()
    with torch.no_grad():
        return _rank(model, state, registry, feature_dim)[:top_k]
