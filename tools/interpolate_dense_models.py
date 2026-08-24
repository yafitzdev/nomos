"""Interpolate compatible SentenceTransformer checkpoints into one artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--specialist-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from safetensors.torch import load_file, save_file

    args = build_parser().parse_args(argv)
    weight = args.specialist_weight
    if not 0.0 <= weight <= 1.0:
        raise SystemExit("specialist weight must be between zero and one")
    base_path = args.base / "model.safetensors"
    specialist_path = args.specialist / "model.safetensors"
    base_state = load_file(str(base_path), device="cpu")
    specialist_state = load_file(str(specialist_path), device="cpu")
    if set(base_state) != set(specialist_state):
        raise ValueError("checkpoint parameter sets differ")
    merged = {
        key: base_state[key].mul(1.0 - weight).add(specialist_state[key], alpha=weight)
        for key in base_state
    }
    shutil.copytree(args.base, args.output, dirs_exist_ok=False)
    save_file(merged, str(args.output / "model.safetensors"))
    manifest = {
        "method": "linear_weight_interpolation",
        "base": str(args.base),
        "specialist": str(args.specialist),
        "base_weight": 1.0 - weight,
        "specialist_weight": weight,
        "output": str(args.output),
    }
    (args.output / "nomos_interpolation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
