"""Minimal SentenceTransformer-compatible ONNX encoder for Nomos deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class OnnxSentenceEncoder:
    """Encode text with a quantized BERT graph, CLS pooling, and L2 normalization."""

    def __init__(self, artifact: Path | str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.artifact = Path(artifact)
        config = json.loads(
            (self.artifact / "nomos_runtime.json").read_text(encoding="utf-8")
        )
        if config.get("backend") != "onnxruntime.cls.v1":
            raise ValueError("unsupported Nomos ONNX runtime configuration")
        self.max_length = int(config.get("max_length", 512))
        self.max_seq_length = self.max_length
        self.default_batch_size = int(config.get("batch_size", 32))
        self.prompts = {"query": "", "document": ""}
        self.tokenizer = Tokenizer.from_file(str(self.artifact / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.artifact / str(config["model_file"])),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {value.name for value in self.session.get_inputs()}

    def encode(
        self,
        texts: list[str] | tuple[str, ...],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = False,
        **_kwargs: Any,
    ) -> Any:
        import numpy as np

        values = [str(text) for text in texts]
        if not values:
            return np.empty((0, 384), dtype=np.float32)
        effective_batch = max(1, int(batch_size or self.default_batch_size))
        outputs = []
        for offset in range(0, len(values), effective_batch):
            encoded = self.tokenizer.encode_batch(values[offset : offset + effective_batch])
            feeds = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in encoded], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [item.type_ids for item in encoded], dtype=np.int64
                ),
            }
            token_embeddings = self.session.run(
                None, {name: feeds[name] for name in self.input_names}
            )[0]
            pooled = token_embeddings[:, 0, :]
            if normalize_embeddings:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = pooled / np.clip(norms, 1e-12, None)
            outputs.append(pooled.astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)
