"""Export a Nomos dense encoder as a standalone dynamically quantized ONNX artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence


RUNTIME_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "nomos_calibration.json",
    "nomos_training_manifest.json",
)


def export(
    source: Path,
    output: Path,
    *,
    max_length: int,
    batch_size: int,
    quantization: str,
) -> dict:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sentence_transformers import SentenceTransformer

    class TransformerGraph(torch.nn.Module):
        def __init__(self, transformer: torch.nn.Module) -> None:
            super().__init__()
            self.transformer = transformer

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
        ) -> torch.Tensor:
            return self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            ).last_hidden_state

    model = SentenceTransformer(str(source), local_files_only=True, device="cpu")
    graph = TransformerGraph(model[0].auto_model).eval()
    output.mkdir(parents=True, exist_ok=True)
    model_name = "model_int8_avx2.onnx" if quantization == "int8" else "model_fp32.onnx"
    with tempfile.TemporaryDirectory(prefix="nomos-onnx-") as temporary:
        fp32_path = Path(temporary) / "model_fp32.onnx"
        inputs = (
            torch.ones((2, 16), dtype=torch.long),
            torch.ones((2, 16), dtype=torch.long),
            torch.zeros((2, 16), dtype=torch.long),
        )
        torch.onnx.export(
            graph,
            inputs,
            str(fp32_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        if quantization == "int8":
            quantize_dynamic(
                fp32_path,
                output / model_name,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
        else:
            shutil.copy2(fp32_path, output / model_name)
    for name in RUNTIME_FILES:
        source_path = source / name
        if source_path.exists():
            shutil.copy2(source_path, output / name)
    runtime = {
        "backend": "onnxruntime.cls.v1",
        "model_file": model_name,
        "pooling": "cls",
        "normalize": True,
        "max_length": max_length,
        "batch_size": batch_size,
        "quantization": "dynamic_int8_avx2" if quantization == "int8" else "none_fp32",
        "source_model": source.name,
    }
    (output / "nomos_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    return {**runtime, "output": str(output), "artifact_bytes": artifact_bytes}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quantization", choices=("none", "int8"), default="int8")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export(
        args.source,
        args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        quantization=args.quantization,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
