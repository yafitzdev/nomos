"""Validate and fingerprint a router.v2 tool registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fitz_tool.tool_registry import RegistryValidationError, load_tool_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_tool_registry(args.registry)
    except RegistryValidationError as exc:
        raise SystemExit(json.dumps({"valid": False, "issues": exc.issues}, indent=2)) from exc
    print(
        json.dumps(
            {
                "valid": True,
                "registry_id": registry.registry_id,
                "registry_fingerprint": registry.fingerprint,
                "tool_count": len(registry.tools),
                "tool_families": sorted({tool.tool_family for tool in registry.tools}),
                "capabilities": sorted(
                    {capability for tool in registry.tools for capability in tool.capabilities}
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
