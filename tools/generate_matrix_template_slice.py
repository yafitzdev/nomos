"""Generate deterministic matrix-bound scenarios for encoder bootstrap runs.

This tool is deliberately not a teacher replacement.  It creates grounded,
matrix-shaped scenarios from immutable source-card manifests so the runner and
encoder can be exercised while an external teacher is unavailable.  Rows are
marked ``teacher=matrix-template`` and must remain separate from NInfer and
DeepSeek analyses.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.contracts import validate_scenario, validate_source_card
from fitz_tool.matrix import materialize_cells
from fitz_tool.uniqueness import annotate_signatures


PROMPT_VERSION = "matrix-template.v1"
SUPPORTED_MODALITIES = {"text", "pdf", "csv", "excel", "sqlite", "code"}
BOOTSTRAP_TOPOLOGIES = {
    "one_passage",
    "multiple_passages",
    "complementary_sources",
    "cross_format",
}
BOOTSTRAP_TERMINALS = {
    "ongoing",
    "selection",
    "abstention",
    "clarification",
    "step_limit_termination",
}
OPERATION_PROMPTS = {
    "lookup": "locate the documented rule for",
    "enumerate": "enumerate the documented values for",
    "compare": "compare the documented claims about",
    "join": "join the relevant evidence for",
    "latest_value_selection": "identify which documented value should be selected for",
    "compatibility": "determine compatibility for",
    "contradiction": "reconcile the competing evidence about",
    "absence": "check the available documentation for",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--source-card-manifest",
        action="append",
        type=Path,
        required=True,
        help="JSONL source-card manifest; repeat for additional manifests.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-slice",
        action="append",
        type=Path,
        default=[],
        help="Previously accepted scenario JSONL; reserve its cells/signatures.",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def _load_cards(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"source-card manifest row is not an object: {path}:{line_number}")
            report = validate_source_card(value)
            if not report.valid:
                raise ValueError(f"invalid source card {path}:{line_number}: {report.as_dict()}")
            source_id = str(value["source_id"])
            if source_id in cards:
                raise ValueError(f"duplicate source card ID: {source_id}")
            cards[source_id] = value
    if not cards:
        raise ValueError("source-card manifests contain no cards")
    return cards


def _excluded(paths: Sequence[Path]) -> tuple[set[str], set[str], set[str]]:
    cells: set[str] = set()
    types: set[str] = set()
    instances: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"excluded row is not an object: {path}:{line_number}")
            matrix_cell = value.get("matrix_cell")
            if isinstance(matrix_cell, Mapping) and isinstance(matrix_cell.get("cell_id"), str):
                cells.add(str(matrix_cell["cell_id"]))
            if isinstance(value.get("type_signature"), str):
                types.add(str(value["type_signature"]))
            if isinstance(value.get("instance_signature"), str):
                instances.add(str(value["instance_signature"]))
    return cells, types, instances


def _card_groups(cards: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {modality: [] for modality in SUPPORTED_MODALITIES}
    for card in cards.values():
        modality = str(card["modality"])
        if modality in groups:
            groups[modality].append(dict(card))
    return groups


def _pick_cards(
    cell: Mapping[str, str],
    groups: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    modality = str(cell["source_modality"])
    digest = hashlib.sha256(f"{seed}:{cell['cell_id']}".encode("utf-8")).digest()
    if modality == "mixed":
        available = [
            card
            for group_modality, group in groups.items()
            if group_modality in SUPPORTED_MODALITIES
            for card in group
        ]
        modalities: list[str] = []
        for item in available:
            item_modality = str(item["modality"])
            if item_modality not in modalities:
                modalities.append(item_modality)
        if len(modalities) < 2:
            raise ValueError("mixed matrix cells require at least two source modalities")
        first_modality = modalities[digest[0] % len(modalities)]
        second_modality = modalities[digest[1] % len(modalities)]
        if second_modality == first_modality:
            second_modality = modalities[(modalities.index(first_modality) + 1) % len(modalities)]
        first_group = groups[first_modality]
        second_group = groups[second_modality]
        return [
            first_group[digest[2] % len(first_group)],
            second_group[digest[3] % len(second_group)],
        ]
    group = groups.get(modality, [])
    if not group:
        raise ValueError(f"no source card is available for modality {modality!r}")
    return [group[digest[0] % len(group)]]


def _selected_facts(cards: Sequence[Mapping[str, Any]], cell: Mapping[str, str], seed: int) -> list[dict[str, str]]:
    all_facts = [
        {"source_id": str(card["source_id"]), "fact_id": str(fact["fact_id"]), "statement": str(fact["statement"])}
        for card in cards
        for fact in card["facts"]
    ]
    if not all_facts:
        raise ValueError("selected source cards contain no facts")
    digest = hashlib.sha256(f"facts:{seed}:{cell['cell_id']}".encode("utf-8")).digest()
    first = all_facts[digest[0] % len(all_facts)]
    selected = [first]
    if cell["evidence_topology"] in {"multiple_passages", "complementary_sources", "cross_format"}:
        second = all_facts[digest[1] % len(all_facts)]
        if (second["source_id"], second["fact_id"]) != (first["source_id"], first["fact_id"]):
            selected.append(second)
    return selected


def _scenario(
    *,
    ordinal: int,
    cell: Mapping[str, str],
    cards: Sequence[Mapping[str, Any]],
    seed: int,
    generated_at: str,
) -> dict[str, Any]:
    selected = _selected_facts(cards, cell, seed)
    fact_text = " and ".join(fact["statement"] for fact in selected)
    operation = OPERATION_PROMPTS[str(cell["information_operation"])]
    domain = str(cell["integration_domain"])
    obstacle = str(cell["retrieval_obstacle"])
    question = (
        f"For the {domain} integration, {operation} the evidence represented by: "
        f"{fact_text} Address the {obstacle} retrieval condition."
    )
    paraphrase = (
        f"In a {domain} API investigation with {obstacle} retrieval, what should be "
        f"done to answer this issue: {fact_text}"
    )
    state = str(cell["agent_state"])
    source_ids = [str(card["source_id"]) for card in cards]
    expected_facts = [
        {"source_id": fact["source_id"], "fact_id": fact["fact_id"]}
        for fact in selected
    ]
    history_by_state = {
        "initial": [],
        "no_hits": [{"tool": "search_bm25", "status": "ok", "result_count": 0}],
        "noisy_hits": [{"tool": "search_bm25", "status": "ok", "result_count": 5}],
        "partial_evidence": [{"tool": "search_bm25", "status": "ok", "result_count": 3}],
        "expansion_needed": [{"tool": "inspect_evidence", "status": "ok", "result_count": 1}],
        "insufficient": [{"tool": "search_bm25", "status": "ok", "result_count": 0}],
        "contradiction": [{"tool": "compare_evidence", "status": "ok", "result_count": 2}],
        "disputed": [{"tool": "compare_evidence", "status": "ok", "result_count": 2}],
        "fresh_sufficient": [{"tool": "assess_evidence", "status": "ok", "result_count": len(selected)}],
    }
    state_setup = {
        "state_name": state,
        "history": history_by_state.get(state, []),
        "observed_evidence": [
            {
                "evidence_id": f"template-E{index}",
                "source_id": fact["source_id"],
                "fact_ids": [fact["fact_id"]],
            }
            for index, fact in enumerate(selected[:2], start=1)
        ],
        "requirements": [{"requirement_id": "R1", "status": "tracked" if state != "initial" else "missing"}],
        "governance": {
            "assessment_fresh": state == "fresh_sufficient",
            "path": cell["governance_path"],
        },
    }
    source_hashes: list[str] = []
    for card in cards:
        source_hashes.append(str(card["content_sha256"]))
        if card.get("normalized_content_sha256"):
            source_hashes.append(str(card["normalized_content_sha256"]))
    scenario = {
        "schema_version": "scenario.v1",
        "scenario_id": f"matrix_template_{seed}_{ordinal:06d}_{cell['cell_id']}",
        "matrix_version": "matrix.v1",
        "matrix_cell": dict(cell),
        "source_card_ids": source_ids,
        "state_setup": state_setup,
        "question": question,
        "difficult_paraphrase": paraphrase,
        "expected_facts": expected_facts,
        "expected_tools": [str(cell["next_tool_target"])],
        "expected_terminal_state": str(cell["terminal_condition"]),
        "provenance": {
            "teacher": "matrix-template",
            "model": "deterministic-template-v1",
            "prompt_version": PROMPT_VERSION,
            "seed": seed,
            "source_card_hashes": source_hashes,
            "generated_at": generated_at,
        },
    }
    return annotate_signatures(scenario)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        raise SystemExit("count must be positive")
    cards = _load_cards(args.source_card_manifest)
    groups = _card_groups(cards)
    available_modalities = {modality for modality, group in groups.items() if group}
    if len(available_modalities) < 1:
        raise SystemExit("no supported source modalities are available")
    excluded_cells, excluded_types, excluded_instances = _excluded(args.exclude_slice)
    cells = materialize_cells(
        args.count,
        seed=args.seed,
        excluded_cell_ids=excluded_cells,
        allowed_source_modalities=available_modalities | {"mixed"},
        allowed_evidence_topologies=BOOTSTRAP_TOPOLOGIES,
        allowed_terminal_conditions=BOOTSTRAP_TERMINALS,
    )
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    scenarios: list[dict[str, Any]] = []
    seen_types = set(excluded_types)
    seen_instances = set(excluded_instances)
    rejected: list[dict[str, Any]] = []
    for ordinal, cell in enumerate(cells):
        try:
            selected_cards = _pick_cards(cell.as_dict(), groups, seed=args.seed)
            scenario = _scenario(
                ordinal=ordinal,
                cell=cell.as_dict(),
                cards=selected_cards,
                seed=args.seed,
                generated_at=generated_at,
            )
            report = validate_scenario(scenario)
            if scenario["type_signature"] in seen_types:
                report.add("type_signature", "duplicate type signature in excluded or current slices")
            if scenario["instance_signature"] in seen_instances:
                report.add("instance_signature", "duplicate instance signature in excluded or current slices")
            if report.valid:
                scenarios.append(scenario)
                seen_types.add(scenario["type_signature"])
                seen_instances.add(scenario["instance_signature"])
            else:
                rejected.append({"cell_id": cell.cell_id, "issues": report.as_dict()})
        except (KeyError, ValueError, IndexError) as exc:
            rejected.append({"cell_id": cell.cell_id, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario, ensure_ascii=False, sort_keys=True) + "\n")
    if rejected:
        args.output.with_suffix(".errors.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in rejected),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "requested": args.count,
                "accepted": len(scenarios),
                "rejected": len(rejected),
                "excluded_cell_ids": len(excluded_cells),
                "reserved_type_signatures": len(excluded_types),
                "reserved_instance_signatures": len(excluded_instances),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if len(scenarios) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
