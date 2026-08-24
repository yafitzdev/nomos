from __future__ import annotations

import json

import numpy as np
import torch

from fitz_tool.embedding_backend import (
    artifact_model_kind,
    copy_remote_code_files,
    encode_documents,
    encode_queries,
    similarity_matrix,
)


class _DenseModel:
    prompts = {"query": "query: ", "document": "document: "}

    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray([[1.0, 0.0] for _text in texts])


class _MultiVectorModel:
    def encode_query(self, texts, **kwargs):
        return [torch.tensor([[1.0, 0.0]]) for _text in texts]

    def encode_document(self, texts, **kwargs):
        return [torch.tensor([[1.0, 0.0], [0.0, 1.0]]) for _text in texts]

    def similarity(self, queries, documents):
        return torch.tensor(
            [[float((query @ document.T).max(dim=1).values.sum()) for document in documents]
             for query in queries]
        )


def test_artifact_model_kind_detects_colbert(tmp_path):
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "ColBERT", "similarity_fn_name": "MaxSim"}),
        encoding="utf-8",
    )
    assert artifact_model_kind(tmp_path) == "multivector"


def test_artifact_model_kind_detects_saved_multivector_encoder(tmp_path):
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "MultiVectorEncoder"}), encoding="utf-8"
    )
    assert artifact_model_kind(tmp_path) == "multivector"


def test_copy_remote_code_files_from_auto_map(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling_custom.CustomModel"}}),
        encoding="utf-8",
    )
    (source / "modeling_custom.py").write_text("class CustomModel: pass\n", encoding="utf-8")

    assert copy_remote_code_files(source, destination) == ["modeling_custom.py"]
    assert (destination / "modeling_custom.py").is_file()


def test_dense_encoders_apply_both_asymmetric_prompts():
    model = _DenseModel()
    encode_queries(model, ["state"])
    encode_documents(model, ["tool"])
    assert model.calls[0][1]["prompt_name"] == "query"
    assert model.calls[1][1]["prompt_name"] == "document"


def test_multivector_similarity_uses_native_model():
    model = _MultiVectorModel()
    queries = encode_queries(model, ["state"])
    documents = encode_documents(model, ["tool"])
    scores = similarity_matrix(model, queries, documents)
    assert scores.shape == (1, 1)
    assert scores[0, 0] == 1.0
