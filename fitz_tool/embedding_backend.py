"""Shared dense and late-interaction embedding operations for Nomos."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence


def artifact_model_kind(path: Path | str) -> str:
    """Return ``dense`` or ``multivector`` from a local ST artifact."""

    config_path = Path(path) / "config_sentence_transformers.json"
    if not config_path.exists():
        return "dense"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type") or "").lower()
    similarity = str(config.get("similarity_fn_name") or "").lower()
    if "multivector" in model_type or model_type == "colbert" or similarity == "maxsim":
        return "multivector"
    return "dense"


def copy_remote_code_files(source: Path | str, destination: Path | str) -> list[str]:
    """Copy local Transformers ``auto_map`` modules required to reload an artifact."""

    source = Path(source)
    destination = Path(destination)
    config_path = source / "config.json"
    if not config_path.exists():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    references = (config.get("auto_map") or {}).values()
    copied: list[str] = []
    for reference in references:
        values = reference if isinstance(reference, list) else [reference]
        for value in values:
            if not isinstance(value, str) or "." not in value:
                continue
            module = value.split("--")[-1].rsplit(".", 1)[0]
            relative_path = Path(*module.split(".")).with_suffix(".py")
            source_path = source / relative_path
            destination_path = destination / relative_path
            if not source_path.exists():
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            copied.append(relative_path.as_posix())
    return sorted(set(copied))


def is_multivector_model(model: Any) -> bool:
    return callable(getattr(model, "encode_query", None)) and callable(
        getattr(model, "encode_document", None)
    )


def load_embedding_model(
    path: Path | str,
    *,
    device: str,
    local_files_only: bool = True,
) -> Any:
    """Load a local dense SentenceTransformer or MultiVectorEncoder."""

    if artifact_model_kind(path) == "multivector":
        from sentence_transformers import MultiVectorEncoder

        return MultiVectorEncoder(
            str(path),
            local_files_only=local_files_only,
            trust_remote_code=True,
            device=device,
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(path),
        local_files_only=local_files_only,
        trust_remote_code=True,
        device=device,
    )


def _prompt_name(model: Any, name: str) -> str | None:
    prompts = getattr(model, "prompts", {})
    return name if name in prompts else None


def encode_queries(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> Any:
    if is_multivector_model(model):
        return model.encode_query(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        prompt_name=_prompt_name(model, "query"),
    )


def encode_documents(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> Any:
    if is_multivector_model(model):
        return model.encode_document(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        prompt_name=_prompt_name(model, "document"),
    )


def similarity_matrix(model: Any, queries: Any, documents: Any) -> Any:
    """Return a CPU NumPy query-by-document score matrix."""

    import numpy as np

    if is_multivector_model(model):
        scores = model.similarity(queries, documents)
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        return np.asarray(scores)
    return np.asarray(queries) @ np.asarray(documents).T
