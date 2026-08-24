"""Small optional local chat backend used by real-session evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class OnnxGenAIChat:
    """Run a local ONNX GenAI causal model without an HTTP server."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        model_filename: str = "onnx/model_q4f16.onnx",
        provider: str = "CPU",
    ) -> None:
        import onnxruntime_genai as og
        from transformers import AutoTokenizer

        self.og = og
        self.path = Path(model_path)
        config = og.Config(str(self.path))
        config.overlay(
            json.dumps({"model": {"decoder": {"filename": model_filename}}})
        )
        config.clear_providers()
        config.append_provider(provider)
        self.model = og.Model(config)
        self.tokenizer = og.Tokenizer(self.model)
        self.chat_tokenizer = AutoTokenizer.from_pretrained(
            str(self.path), local_files_only=True
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:
        prompt = self.chat_tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = self.tokenizer.encode(prompt)
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=len(input_ids) + max_new_tokens,
            do_sample=False,
        )
        generator = self.og.Generator(self.model, params)
        generator.append_tokens(input_ids)
        while not generator.is_done():
            generator.generate_next_token()
        sequence = generator.get_sequence(0)
        output_ids = sequence[len(input_ids) :]
        return {
            "text": self.tokenizer.decode(output_ids),
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
        }
