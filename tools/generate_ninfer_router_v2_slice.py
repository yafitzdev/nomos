"""Generate matrix.v2-bound, unverified NInfer proposals for router.v2."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from fitz_tool.matrix_v2 import MatrixV2Cell, load_matrix_v2_spec, validate_matrix_v2_cell
from fitz_tool.pilot_v2 import (
    DOMAIN_FOCUS,
    _candidate_specs,
    _valid_cell_for_target,
    load_pilot_registries,
    load_pilot_source_cards,
)
from fitz_tool.router_v2 import FEATURE_VERSION
from fitz_tool.tool_registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "ninfer-router-v2-prompt.v1"
PROPOSAL_VERSION = "teacher-proposal.v2"
DEFAULT_BASE_URL = os.environ.get(
    "FITZ_TOOL_NINFER_BASE_URL",
    os.environ.get("FITZ_TOOL_TEACHER_BASE_URL", "http://127.0.0.1:19003/v1"),
)
DEFAULT_MODEL = os.environ.get(
    "FITZ_TOOL_NINFER_MODEL",
    os.environ.get("FITZ_TOOL_TEACHER_MODEL", "Qwen/Qwen3.8-27B"),
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _api_key(*, disabled: bool) -> str | None:
    if disabled:
        return None
    key = os.environ.get("FITZ_TOOL_TEACHER_API_KEY") or os.environ.get(
        "FITZ_AGENT_TEACHER_API_KEY"
    )
    if key:
        return key
    return getpass.getpass("ninfer teacher API key: ")


def _read_excluded(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    excluded: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid excluded pilot row {path}:{line_number}: {exc}") from exc
        cell_id = row.get("matrix_cell_id")
        if isinstance(cell_id, str):
            excluded.add(cell_id)
    return excluded


def _source_choices(
    cell: MatrixV2Cell,
    cards: Mapping[str, Any],
    rng: random.Random,
) -> list[str]:
    modality = str(cell.values["source_modality"])
    matching = [card for card in cards.values() if card.modality == modality]
    if modality == "mixed":
        matching = list(cards.values())
    if not matching:
        matching = list(cards.values())
    rng.shuffle(matching)
    chosen = [matching[0]]
    if modality == "mixed":
        second = next((card for card in matching[1:] if card.modality != chosen[0].modality), None)
        if second is not None:
            chosen.append(second)
    return [card.source_id for card in chosen]


def _source_payload(source_ids: list[str], cards: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for source_id in source_ids:
        card = cards[source_id]
        content = Path(card.path).read_text(encoding="utf-8")
        output.append(
            {
                "source_card_id": source_id,
                "modality": card.modality,
                "content_sha256": card.content_sha256,
                "content": content,
            }
        )
    return output


def _assignment(
    *,
    ordinal: int,
    target_capability: str,
    cell: MatrixV2Cell,
    source_ids: list[str],
    registry: ToolRegistry,
    legal_ids: list[str],
    cards: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "assignment_id": f"assignment-{ordinal:05d}",
        "ordinal": ordinal,
        "target_capability": target_capability,
        "matrix_cell": cell.as_dict(),
        "matrix_cell_id": cell.cell_id,
        "source_card_ids": source_ids,
        "source_card_hashes": [cards[source_id].content_sha256 for source_id in source_ids],
        "source_documents": _source_payload(source_ids, cards),
        "registry_id": registry.registry_id,
        "registry_fingerprint": registry.fingerprint,
        "tool_registry": registry.as_dict(),
        "legal_candidate_ids": legal_ids,
        "candidate_descriptors": [registry.require(tool_id).as_dict() for tool_id in legal_ids],
    }


def build_assignments(
    *,
    count: int,
    seed: int,
    excluded_cell_ids: set[str],
    root: Path,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    spec = load_matrix_v2_spec(root / "configs" / "matrix.v2.json")
    cards = load_pilot_source_cards(root / "tests" / "fixtures" / "pilot_v2_corpus")
    registry = load_pilot_registries(root)["fitz_sage_v2"]
    targets = list(spec["dimensions"]["target_capability"])
    rng = random.Random(seed)
    used = set(excluded_cell_ids)
    assignments: list[dict[str, Any]] = []
    for ordinal in range(count):
        target = str(targets[ordinal % len(targets)])
        cell = _valid_cell_for_target(rng, target, used, spec)
        used.add(cell.cell_id)
        source_ids = _source_choices(cell, cards, rng)
        legal_specs, _acceptable = _candidate_specs(
            registry,
            target,
            str(cell.values["candidate_set_difficulty"]),
            rng,
        )
        legal_ids = [tool.tool_id for tool in legal_specs]
        assignments.append(
            _assignment(
                ordinal=ordinal,
                target_capability=target,
                cell=cell,
                source_ids=source_ids,
                registry=registry,
                legal_ids=legal_ids,
                cards=cards,
            )
        )
    return assignments


def _prompt(batch: list[dict[str, Any]]) -> str:
    items = []
    for assignment in batch:
        cell = assignment["matrix_cell"]
        items.append(
            {
                "assignment_id": assignment["assignment_id"],
                "target_capability": assignment["target_capability"],
                "domain_focus": DOMAIN_FOCUS.get(
                    str(cell["integration_domain"]), "the API behavior"
                ),
                "matrix_context": {
                    key: cell[key]
                    for key in (
                        "integration_domain",
                        "information_operation",
                        "source_modality",
                        "evidence_topology",
                        "retrieval_obstacle",
                        "agent_phase",
                        "query_specificity",
                        "match_strategy",
                        "evidence_inspection_state",
                        "requirement_progress",
                        "assessment_freshness",
                        "terminal_outcome",
                    )
                },
                "legal_candidate_ids": assignment["legal_candidate_ids"],
                "candidate_descriptors": [
                    {
                        "tool_id": descriptor["tool_id"],
                        "tool_family": descriptor["tool_family"],
                        "description": descriptor["description"],
                        "capabilities": descriptor["capabilities"],
                        "input_modalities": descriptor["input_modalities"],
                        "side_effect_class": descriptor["side_effect_class"],
                    }
                    for descriptor in assignment["candidate_descriptors"]
                ],
                "source_documents": assignment["source_documents"],
            }
        )
    return f"""Generate exactly one grounded teacher proposal for each assigned item below.

This is a synthetic-data proposal pass for a generic tool router. The matrix cell
is an assignment constraint, not evidence. Use only the supplied source-document
content for the user question. Do not invent facts, source IDs, tool IDs, or
terminal outcomes. The proposal is not a verified label.

For each item return:
- assignment_id: exactly the supplied assignment ID
- question: one concrete technical integration/API research question
- difficult_paraphrase: a materially different wording of the same question
- proposed_tool_ids: one or more candidate IDs from legal_candidate_ids only;
  choose the candidate(s) that appear useful, but do not claim execution verified them
- proposed_capabilities: capability names corresponding to the proposed candidates
- grounding_terms: two to five exact distinctive terms copied from the supplied source
  content, such as an endpoint, field, event, error code, or section phrase

Return a JSON array with exactly {len(batch)} objects and no Markdown.

Assigned items:
{json.dumps(items, ensure_ascii=False, sort_keys=True)}"""


def _decode(content: str) -> Any:
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()
    return json.loads(content)


def _request(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    batch: list[dict[str, Any]],
    timeout: float,
    retries: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], str | None]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You generate grounded synthetic data as strict JSON only.",
            },
            {"role": "user", "content": _prompt(batch)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "enable_thinking": False,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error = "unknown teacher error"
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            content = str((body["choices"][0].get("message") or {}).get("content") or "")
            parsed = _decode(content)
            if not isinstance(parsed, list) or len(parsed) != len(batch):
                raise ValueError("teacher returned the wrong number of proposal objects")
            if not all(isinstance(item, dict) for item in parsed):
                raise ValueError("teacher returned a non-object proposal")
            return parsed, None
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTPError {exc.code}: {detail}"
            else:
                last_error = f"{type(exc).__name__}: {exc}"
    return [], last_error


def _proposal(
    *,
    assignment: Mapping[str, Any],
    generated: Mapping[str, Any],
    model: str,
    seed: int,
    generated_at: str,
) -> dict[str, Any]:
    registry = ToolRegistry.from_dict(assignment["tool_registry"])
    proposed_tool_ids = generated.get("proposed_tool_ids")
    if not isinstance(proposed_tool_ids, list):
        proposed_tool_ids = []
    proposed_capabilities = generated.get("proposed_capabilities")
    if not isinstance(proposed_capabilities, list):
        proposed_capabilities = []
    grounding_terms = generated.get("grounding_terms")
    if not isinstance(grounding_terms, list):
        grounding_terms = []
    row = {
        "schema_version": PROPOSAL_VERSION,
        "scenario_id": f"ninfer-router-v2-{seed}-{int(assignment['ordinal']):05d}",
        "matrix_version": "matrix.v2",
        "matrix_cell": dict(assignment["matrix_cell"]),
        "matrix_cell_id": assignment["matrix_cell_id"],
        "target_capability": assignment["target_capability"],
        "registry_id": assignment["registry_id"],
        "registry_fingerprint": assignment["registry_fingerprint"],
        "legal_candidate_ids": list(assignment["legal_candidate_ids"]),
        "tool_registry": registry.as_dict(),
        "source_card_ids": list(assignment["source_card_ids"]),
        "source_card_hashes": list(assignment["source_card_hashes"]),
        "question": str(generated.get("question") or ""),
        "difficult_paraphrase": str(generated.get("difficult_paraphrase") or ""),
        "proposed_tool_ids": [str(value) for value in proposed_tool_ids],
        "proposed_capabilities": [str(value) for value in proposed_capabilities],
        "grounding_terms": [str(value) for value in grounding_terms],
        "source_kind": "ninfer_proposal",
        "execution_status": "not_executed",
        "label_status": "unverified_teacher_proposal",
        "provenance": {
            "corpus": "nomos_router_v2_pilot",
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "artifact": "NInfer-local-endpoint",
            "teacher": "ninfer",
            "seed": seed,
            "validator_version": "ninfer-proposal-validator.v2",
            "feature_version": FEATURE_VERSION,
            "registry_fingerprint": assignment["registry_fingerprint"],
            "source_card_hashes": list(assignment["source_card_hashes"]),
            "matrix_cell_id": assignment["matrix_cell_id"],
            "generated_at": generated_at,
        },
    }
    row["type_signature"] = _digest(
        {
            "matrix_cell": row["matrix_cell"],
            "registry_fingerprint": row["registry_fingerprint"],
            "legal_candidate_ids": sorted(row["legal_candidate_ids"]),
            "source_card_ids": sorted(row["source_card_ids"]),
        }
    )
    row["instance_signature"] = _digest(
        {"type_signature": row["type_signature"], "question": row["question"]}
    )
    return row


def validate_proposal(
    row: Mapping[str, Any],
    *,
    cards: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if row.get("schema_version") != PROPOSAL_VERSION:
        errors.append("schema_version must be teacher-proposal.v2")
    cell = row.get("matrix_cell")
    if not isinstance(cell, Mapping):
        errors.append("matrix_cell must be an object")
    else:
        errors.extend(validate_matrix_v2_cell(cell))
        if row.get("matrix_cell_id") != _digest(dict(cell)):
            errors.append("matrix_cell_id does not match matrix_cell")
    try:
        registry = ToolRegistry.from_dict(row["tool_registry"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid tool_registry: {exc}")
        registry = None
    if registry is not None:
        if row.get("registry_fingerprint") != registry.fingerprint:
            errors.append("registry_fingerprint does not match tool_registry")
        legal = row.get("legal_candidate_ids")
        if not isinstance(legal, list) or not legal:
            errors.append("legal_candidate_ids must be non-empty")
        elif len(legal) != len(set(legal)):
            errors.append("legal_candidate_ids must be unique")
        elif not set(legal) <= set(registry.by_id):
            errors.append("legal_candidate_ids contains an unknown tool ID")
        proposed = row.get("proposed_tool_ids")
        if not isinstance(proposed, list) or not proposed:
            errors.append("proposed_tool_ids must be non-empty")
        elif not set(proposed) <= set(legal or []):
            errors.append("proposed_tool_ids must be a subset of legal_candidate_ids")
        capabilities = set()
        for tool_id in proposed or []:
            if tool_id in registry.by_id:
                capabilities.update(registry.require(tool_id).capabilities)
        if not set(row.get("proposed_capabilities") or []) <= capabilities:
            errors.append("proposed_capabilities must match proposed candidate metadata")
    for key in ("scenario_id", "question", "difficult_paraphrase"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"{key} must be non-empty")
    source_ids = row.get("source_card_ids")
    source_hashes = row.get("source_card_hashes")
    if not isinstance(source_ids, list) or not source_ids:
        errors.append("source_card_ids must be non-empty")
    elif not all(source_id in cards for source_id in source_ids):
        errors.append("source_card_ids contains an unknown source card")
    if not isinstance(source_hashes, list) or source_hashes != [cards[source_id].content_sha256 for source_id in source_ids if source_id in cards]:
        errors.append("source_card_hashes do not match source cards")
    terms = row.get("grounding_terms")
    if not isinstance(terms, list) or not terms:
        errors.append("grounding_terms must be non-empty")
    else:
        source_text = "\n".join(
            Path(cards[source_id].path).read_text(encoding="utf-8")
            for source_id in source_ids
            if source_id in cards
        ).casefold()
        if not any(str(term).strip().casefold() in source_text for term in terms if str(term).strip()):
            errors.append("grounding_terms contain no exact source match")
    if row.get("source_kind") != "ninfer_proposal":
        errors.append("source_kind must identify an NInfer proposal")
    if row.get("execution_status") != "not_executed":
        errors.append("execution_status must remain not_executed")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        for key in (
            "corpus", "prompt_version", "model", "artifact", "teacher", "seed",
            "validator_version", "feature_version", "registry_fingerprint",
            "source_card_hashes", "matrix_cell_id", "generated_at",
        ):
            if key not in provenance:
                errors.append(f"provenance.{key} is missing")
        if provenance.get("registry_fingerprint") != row.get("registry_fingerprint"):
            errors.append("provenance registry fingerprint mismatch")
        if provenance.get("matrix_cell_id") != row.get("matrix_cell_id"):
            errors.append("provenance matrix cell mismatch")
    expected_type = _digest(
        {
            "matrix_cell": row.get("matrix_cell"),
            "registry_fingerprint": row.get("registry_fingerprint"),
            "legal_candidate_ids": sorted(row.get("legal_candidate_ids") or []),
            "source_card_ids": sorted(row.get("source_card_ids") or []),
        }
    )
    if row.get("type_signature") != expected_type:
        errors.append("type_signature mismatch")
    if row.get("instance_signature") != _digest(
        {"type_signature": expected_type, "question": row.get("question")}
    ):
        errors.append("instance_signature mismatch")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-pilot", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-api-key", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1 or args.batch_size < 1 or args.concurrency < 1:
        raise SystemExit("count, batch-size, and concurrency must be positive")
    root = PROJECT_ROOT
    cards = load_pilot_source_cards(root / "tests" / "fixtures" / "pilot_v2_corpus")
    assignments = build_assignments(
        count=args.count,
        seed=args.seed,
        excluded_cell_ids=_read_excluded(args.exclude_pilot),
        root=root,
    )
    batches = [
        assignments[start : start + args.batch_size]
        for start in range(0, len(assignments), args.batch_size)
    ]
    api_key = _api_key(disabled=args.no_api_key)
    results: dict[int, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _request,
                base_url=args.base_url,
                model=args.model,
                api_key=api_key,
                batch=batch,
                timeout=args.timeout,
                retries=args.retries,
                max_tokens=args.max_tokens,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            generated, error = future.result()
            if error:
                failures.append({"batch_index": batch_index, "error": error})
            else:
                results[batch_index] = generated

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    seen_instances: set[str] = set()
    for batch_index, batch in enumerate(batches):
        generated = results.get(batch_index)
        if generated is None:
            continue
        for assignment, item in zip(batch, generated):
            row = _proposal(
                assignment=assignment,
                generated=item,
                model=args.model,
                seed=args.seed,
                generated_at=generated_at,
            )
            errors = validate_proposal(row, cards=cards)
            if row["type_signature"] in seen_types:
                errors.append("duplicate type_signature")
            if row["instance_signature"] in seen_instances:
                errors.append("duplicate instance_signature")
            if errors:
                rejected.append(
                    {"assignment_id": assignment["assignment_id"], "errors": errors}
                )
            else:
                rows.append(row)
                seen_types.add(row["type_signature"])
                seen_instances.add(row["instance_signature"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    if failures or rejected:
        args.output.with_suffix(".errors.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in [*failures, *rejected]
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "requested": args.count,
                "accepted": len(rows),
                "failed_batches": len(failures),
                "rejected": len(rejected),
                "output": str(args.output),
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if len(rows) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
