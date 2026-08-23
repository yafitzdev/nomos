"""A small state-aware tool-ranking encoder for the first Fitz-Tool baseline."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
ROUTER_VERSION = "router.v1"


@dataclass(frozen=True)
class RouterConfig:
    feature_dim: int = 2048
    hidden_dim: int = 128
    epochs: int = 12
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
        for key, item in value.items():
            output.extend([str(key).casefold(), *_tokens(item)])
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(_tokens(item))
        return output
    if value is None:
        return []
    return [str(value).casefold()]


def state_feature_tokens(state: Mapping[str, Any], candidate_tool: str) -> list[str]:
    """Serialize the compact decision state into prefixed feature tokens."""

    tokens: list[str] = [f"candidate_tool={candidate_tool}"]
    tokens.extend(f"question={token}" for token in _tokens(state.get("question")))
    agent_state = state.get("agent_state") or {}
    if isinstance(agent_state, Mapping):
        tokens.append(f"agent_state={agent_state.get('state_name', 'unknown')}")
        tokens.extend(f"agent_{token}" for token in _tokens(agent_state))
    tokens.extend(f"legal_tool={tool}" for tool in state.get("legal_tools") or [])
    for field in ("history", "plan", "matrix_context"):
        tokens.extend(f"{field}_{token}" for token in _tokens(state.get(field)))
    matrix_context = state.get("matrix_context") or {}
    if isinstance(matrix_context, Mapping):
        context_values = {
            str(key): str(value)
            for key, value in matrix_context.items()
            if key != "next_tool_target"
        }
        # Candidate-conditioned interactions let a compact MLP learn rules
        # such as state+modality+operation -> legal next tool without exposing
        # the target label itself.  The target is intentionally absent from
        # matrix_context and therefore cannot leak into these features.
        for key, value in context_values.items():
            tokens.append(f"candidate_context={candidate_tool}|{key}={value}")
        for left, right in (
            ("agent_state", "information_operation"),
            ("agent_state", "source_modality"),
            ("information_operation", "source_modality"),
            ("agent_state", "retrieval_obstacle"),
            ("terminal_condition", "agent_state"),
            ("governance_path", "agent_state"),
        ):
            if left in context_values and right in context_values:
                tokens.append(
                    "candidate_context_pair="
                    f"{candidate_tool}|{left}={context_values[left]}|{right}={context_values[right]}"
                )
    governance = state.get("governance") or {}
    tokens.extend(f"governance_{token}" for token in _tokens(governance))
    for evidence in state.get("observed_evidence") or []:
        if isinstance(evidence, Mapping):
            tokens.extend(f"evidence_{token}" for token in _tokens(evidence.get("evidence_id")))
            tokens.extend(f"evidence_source_{token}" for token in _tokens(evidence.get("source_id")))
    return tokens


def featurize(state: Mapping[str, Any], candidate_tool: str, feature_dim: int) -> list[float]:
    """Hash state/candidate tokens into a normalized dense feature vector."""

    if feature_dim < 16:
        raise ValueError("feature_dim must be at least 16")
    vector = [0.0] * feature_dim
    for token in state_feature_tokens(state, candidate_tool):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % feature_dim
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def split_name(group_id: str) -> str:
    """Assign a trajectory/group to a stable train/validation/test split."""

    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def pair_examples(
    states: Iterable[Mapping[str, Any]],
    *,
    feature_dim: int,
) -> list[dict[str, Any]]:
    """Expand multi-label decision states into candidate-tool training pairs."""

    examples: list[dict[str, Any]] = []
    for state in states:
        label = state.get("label") or {}
        acceptable = set(label.get("acceptable_tools") or [])
        hard_negatives = set(label.get("hard_negative_tools") or [])
        legal_tools = [str(tool) for tool in state.get("legal_tools") or []]
        if state.get("accepted"):
            candidates = legal_tools
        else:
            candidates = [tool for tool in legal_tools if tool in hard_negatives]
        for tool in candidates:
            examples.append(
                {
                    "state_id": state.get("decision_state_id"),
                    "trajectory_id": state.get("trajectory_id"),
                    "scenario_id": state.get("scenario_id"),
                    "split": split_name(str(state.get("trajectory_id") or state.get("scenario_id"))),
                    "tool": tool,
                    "label": float(tool in acceptable),
                    "features": featurize(state, tool, feature_dim),
                }
            )
    return examples


def _metrics(model: Any, states: list[Mapping[str, Any]], *, feature_dim: int) -> dict[str, float | int]:
    import torch

    model.eval()
    evaluated = 0
    recall_1 = 0
    recall_3 = 0
    with torch.no_grad():
        for state in states:
            acceptable = set((state.get("label") or {}).get("acceptable_tools") or [])
            legal_tools = [str(tool) for tool in state.get("legal_tools") or []]
            if not acceptable or not legal_tools:
                continue
            features = torch.tensor(
                [featurize(state, tool, feature_dim) for tool in legal_tools], dtype=torch.float32
            )
            scores = model(features).flatten().tolist()
            ranked = [tool for _, tool in sorted(zip(scores, legal_tools), reverse=True)]
            evaluated += 1
            recall_1 += int(bool(set(ranked[:1]) & acceptable))
            recall_3 += int(bool(set(ranked[:3]) & acceptable))
    return {
        "states": evaluated,
        "recall_at_1": recall_1 / evaluated if evaluated else 0.0,
        "recall_at_3": recall_3 / evaluated if evaluated else 0.0,
        "invalid_candidate_rate": 0.0,
    }


def train_router(
    states: list[Mapping[str, Any]],
    *,
    config: RouterConfig = RouterConfig(),
) -> tuple[Any, dict[str, Any]]:
    """Train the first router and return a PyTorch module plus metadata."""

    import torch
    from torch import nn

    if not states:
        raise ValueError("cannot train a router without decision states")
    torch.manual_seed(config.seed)
    examples = pair_examples(states, feature_dim=config.feature_dim)
    positives = sum(example["label"] for example in examples)
    negatives = len(examples) - positives
    if not positives or not negatives:
        raise ValueError("training requires both positive and negative candidate labels")

    class RouterMLP(nn.Module):
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

    model = RouterMLP()
    train_examples = [example for example in examples if example["split"] == "train"]
    if not train_examples:
        train_examples = examples
    features = torch.tensor([example["features"] for example in train_examples], dtype=torch.float32)
    labels = torch.tensor([[example["label"]] for example in train_examples], dtype=torch.float32)
    positive_weight = max(1.0, (len(train_examples) - labels.sum().item()) / max(1.0, labels.sum().item()))
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

    by_split = {
        split: [state for state in states if split_name(str(state.get("trajectory_id") or state.get("scenario_id"))) == split]
        for split in ("train", "validation", "test")
    }
    metrics = {split: _metrics(model, rows, feature_dim=config.feature_dim) for split, rows in by_split.items()}
    metadata = {
        "router_version": ROUTER_VERSION,
        "config": config.as_dict(),
        "tool_vocab": sorted({tool for state in states for tool in state.get("legal_tools") or []}),
        "state_count": len(states),
        "pair_count": len(examples),
        "positive_pairs": int(positives),
        "negative_pairs": int(negatives),
        "losses": losses,
        "metrics": metrics,
        "feature_description": "blake2b hashed prefixed state/candidate tokens",
    }
    return model, metadata


def save_router(path: str, model: Any, metadata: Mapping[str, Any]) -> None:
    import torch

    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, path)


def load_router(path: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(artifact["metadata"])
    config = RouterConfig(**metadata["config"])

    class RouterMLP(nn.Module):
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

    model = RouterMLP()
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, metadata


def rank_tools(model: Any, metadata: Mapping[str, Any], state: Mapping[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    import torch

    feature_dim = int(metadata["config"]["feature_dim"])
    legal_tools = [str(tool) for tool in state.get("legal_tools") or []]
    with torch.no_grad():
        scores = model(
            torch.tensor([featurize(state, tool, feature_dim) for tool in legal_tools], dtype=torch.float32)
        ).flatten().tolist()
    ranked = sorted(zip(scores, legal_tools), reverse=True)[:top_k]
    return [{"tool": tool, "score": float(score)} for score, tool in ranked]
