"""Runtime dense ranker for arbitrary runner-supplied tool registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .dense_router import (
    DENSE_TEXT_VERSION,
    candidate_views,
    eligible_tools,
    query_document,
    weighted_query_views,
)
from .embedding_backend import (
    encode_documents,
    encode_queries,
    load_embedding_model,
    similarity_matrix,
)


class DenseToolRanker:
    """Rank legal tools from their semantics, never from concrete tool IDs."""

    def __init__(
        self,
        model: Any,
        *,
        model_name: str = "injected",
        query_strategy: str = "multiview",
        candidate_strategy: str = "multiview",
    ) -> None:
        if query_strategy not in {"single", "multiview"}:
            raise ValueError("query_strategy must be single or multiview")
        if candidate_strategy not in {"single", "multiview"}:
            raise ValueError("candidate_strategy must be single or multiview")
        self.model = model
        self.model_name = model_name
        self.query_strategy = query_strategy
        self.candidate_strategy = candidate_strategy
        self._candidate_cache: dict[str, Any] = {}
        self.calibration: Mapping[str, Any] | None = None

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        device: str = "cpu",
        query_strategy: str = "multiview",
        candidate_strategy: str = "multiview",
    ) -> "DenseToolRanker":
        artifact = Path(path)
        runtime_path = artifact / "nomos_runtime.json"
        if runtime_path.exists():
            from .onnx_encoder import OnnxSentenceEncoder

            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            if runtime.get("backend") != "onnxruntime.cls.v1":
                raise ValueError("unsupported Nomos runtime backend")
            if device != "cpu":
                raise ValueError("the packaged ONNX runtime currently supports CPU only")
            model = OnnxSentenceEncoder(artifact)
        else:
            model = load_embedding_model(artifact, device=device)
        ranker = cls(
            model,
            model_name=artifact.name,
            query_strategy=query_strategy,
            candidate_strategy=candidate_strategy,
        )
        calibration_path = artifact / "nomos_calibration.json"
        if calibration_path.exists():
            value = json.loads(calibration_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                ranker.calibration = value
        return ranker

    @property
    def version(self) -> str:
        return f"dense:{self.model_name}:{DENSE_TEXT_VERSION}"

    def _encode(self, texts: list[str], *, query: bool) -> Any:
        encoder = encode_queries if query else encode_documents
        return encoder(self.model, texts)

    def rank(
        self, request: Mapping[str, Any], *, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        tools = eligible_tools(request)
        missing = [
            tool for tool in tools if tool.semantic_fingerprint not in self._candidate_cache
        ]
        if missing:
            missing_views = [
                candidate_views(tool)
                if self.candidate_strategy == "multiview"
                else (candidate_views(tool)[0],)
                for tool in missing
            ]
            embeddings = self._encode(
                [text for views in missing_views for text in views], query=False
            )
            offset = 0
            for tool, views in zip(missing, missing_views):
                self._candidate_cache[tool.semantic_fingerprint] = embeddings[
                    offset : offset + len(views)
                ]
                offset += len(views)

        query_views = (
            weighted_query_views(request)
            if self.query_strategy == "multiview"
            else ((query_document(request), 1.0),)
        )
        query_embeddings = self._encode(
            [text for text, _weight in query_views], query=True
        )
        scored = []
        for tool in tools:
            candidate_embeddings = self._candidate_cache[tool.semantic_fingerprint]
            score = sum(
                weight
                * float(
                    similarity_matrix(
                        self.model,
                        [query_embedding],
                        candidate_embeddings,
                    ).max()
                )
                for query_embedding, (_text, weight) in zip(
                    query_embeddings, query_views
                )
            )
            scored.append((score, tool))
        ranked = sorted(
            scored, key=lambda item: (-item[0], item[1].semantic_fingerprint)
        )
        if top_k is not None:
            ranked = ranked[: max(0, top_k)]
        return [
            {
                "tool_id": tool.tool_id,
                "tool_family": tool.tool_family,
                "semantic_fingerprint": tool.semantic_fingerprint,
                "score": score,
            }
            for score, tool in ranked
        ]
