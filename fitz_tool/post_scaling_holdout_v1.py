"""Frozen post-scaling holdout with unseen registries, schemas, and workflows."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .generic_contracts import validate_decision_state_v2
from .router_v2 import FEATURE_VERSION
from .scaling_matrix_v1 import (
    DATASET_VERSION,
    PURPOSES,
    _allowed_effects,
    build_registry,
    digest,
)


HOLDOUT_VERSION = "nomos-post-scaling-holdout.v1"
HOLDOUT_MATRIX_VERSION = "matrix.post-scaling-holdout.v1"
HOLDOUT_SEED = 20_269_991
HOLDOUT_STYLES = ("lattice", "meridian", "prism", "vanguard")
HOLDOUT_SCHEMA_STYLES = ("holdout_tuple", "holdout_discriminated", "holdout_map")
HOLDOUT_POOL_SIZES = (34, 50, 100)
HOLDOUT_TEMPLATE_PREFIX = "postscale-holdout-v1"
FROZEN_AT = "2026-08-24T00:00:00+00:00"

WORKFLOWS = (
    ("symbol_lineage_trace", "inspect_code_structure", ("read_content", "exact_pattern_search")),
    ("resource_surface_inventory", "list_sources", ("search_metadata", "search_content")),
    ("catalog_attribute_narrowing", "search_metadata", ("list_sources", "search_content")),
    ("literal_contract_locator", "exact_pattern_search", ("search_content", "search_metadata")),
    ("concept_passage_discovery", "search_content", ("exact_pattern_search", "read_content")),
    ("selected_asset_open", "read_content", ("search_content", "search_metadata")),
    ("table_shape_discovery", "inspect_structured_schema", ("search_structured_records", "read_content")),
    ("qualified_record_retrieval", "search_structured_records", ("inspect_structured_schema", "search_content")),
    ("manual_navigation_map", "inspect_document_structure", ("search_document_pages", "read_content")),
    ("manual_location_lookup", "search_document_pages", ("inspect_document_structure", "search_content")),
    ("observation_lineage_audit", "inspect_provenance", ("inspect_evidence", "search_content")),
    ("clipped_result_recovery", "expand_context", ("search_content", "read_content")),
    ("support_record_audit", "inspect_evidence", ("compare_evidence", "assess_evidence")),
    ("cross_source_reconciliation", "compare_evidence", ("assess_evidence", "finalize_selection")),
    ("condition_ledger_refresh", "update_requirements", ("assess_evidence", "finalize_selection")),
    ("support_sufficiency_gate", "assess_evidence", ("finalize_selection", "update_requirements")),
    ("justified_decision_commit", "finalize_selection", ("assess_evidence", "plan_retrieval")),
    ("visual_asset_inspection", "analyze_image", ("transcribe_audio", "read_content")),
    ("spoken_record_conversion", "transcribe_audio", ("analyze_image", "read_content")),
    ("bounded_change_preflight", "validate_change", ("publish_external", "delete_external")),
    ("unsupported_remote_mutation", None, ("publish_external", "delete_external")),
)


def _wording(workflow: str, target: str | None, stage: int) -> tuple[str, str, list[str]]:
    label = workflow.replace("_", " ")
    if target is None:
        history = ["Confirmed the exact externally hosted account named by the user."]
        if stage == 2:
            history.extend(
                [
                    "Reviewed the requested deletion scope and retained the unresolved outcome.",
                    "Rejected an inspection-only route because it could not perform the requested change.",
                ]
            )
        return (
            "Permanently remove the externally hosted account and all associated remote data now.",
            "The immediate requested outcome is permanent deletion of the remote account and its data.",
            history if stage else [],
        )
    purpose = PURPOSES[target]
    if stage == 0:
        return (
            f"For the {label} workflow, choose the first bounded action that will move the investigation forward.",
            f"The immediate outcome must {purpose}.",
            [],
        )
    if stage == 1:
        return (
            f"Continue the {label} workflow now that its input has been identified; select the operation for the active evidence step.",
            f"Use the current input to {purpose}.",
            ["Identified the relevant input and preserved its stable handle."],
        )
    return (
        f"The prior option in the {label} workflow did not resolve the requirement. Return a fresh route that directly advances it.",
        f"Avoid the rejected candidates and instead {purpose}.",
        [
            "Identified the relevant input and preserved its stable handle.",
            "Tried two superficially plausible operations without resolving the requirement.",
            "Kept the unresolved condition open for recovery.",
        ],
    )


def generate_post_scaling_holdout() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {
        "registry_style": Counter(),
        "schema_style": Counter(),
        "candidate_pool_size": Counter(),
        "task_kind": Counter(),
        "workflow": Counter(),
    }
    index = 0
    for workflow, target, negatives in WORKFLOWS:
        for style_index, style in enumerate(HOLDOUT_STYLES):
            for pool_index, pool_size in enumerate(HOLDOUT_POOL_SIZES):
                for stage in range(3):
                    schema_style = HOLDOUT_SCHEMA_STYLES[(style_index + pool_index + stage) % len(HOLDOUT_SCHEMA_STYLES)]
                    cell = {
                        "matrix_version": HOLDOUT_MATRIX_VERSION,
                        "scenario_family": f"sealed_{workflow}",
                        "workflow": workflow,
                        "registry_description_style": style,
                        "argument_schema_style": schema_style,
                        "candidate_pool_size": pool_size,
                        "source_modality": "mixed" if target in {"compare_evidence", "assess_evidence"} else "text",
                        "side_effect_policy": "read_only",
                        "session_position": ("initial", "middle", "late")[stage],
                        "history_length": ("empty", "short", "long")[stage],
                        "wording_style": "sealed_narrative",
                        "capability_visibility": "opaque",
                        "initial_target_position": ("third", "later", "second")[stage],
                        "history_transition": "failed_candidates" if stage == 2 else "none",
                        "slot": index,
                        "replacement_ordinal": 0,
                    }
                    assignment = {
                        "assignment_id": f"postscale-holdout-{index:04d}",
                        "slot_id": f"postscale-holdout-{index:04d}",
                        "slot": index,
                        "replacement_ordinal": 0,
                        "seed": HOLDOUT_SEED + index * 130_363,
                        "matrix_cell": cell,
                        "matrix_cell_id": digest(cell),
                        "target_capability": target,
                        "hard_negative_capabilities": list(negatives),
                        "illegal_capabilities": [value for value in negatives if value in {"publish_external", "delete_external"}],
                        "routing_hint": f"sealed workflow {workflow}",
                        "force_recovery": stage == 2,
                    }
                    registry, legal, target_id, hard_negatives, previous = build_registry(assignment, holdout=True)
                    question, current_step, completed = _wording(workflow, target, stage)
                    allowed = sorted(_allowed_effects("read_only"))
                    row = {
                        "schema_version": "decision-state.v2",
                        "dataset_version": HOLDOUT_VERSION,
                        "decision_state_id": assignment["assignment_id"],
                        "trajectory_id": f"sealed-trajectory-{workflow}-{style}-{pool_size}",
                        "scenario_id": f"sealed-scenario-{workflow}-{style}-{pool_size}",
                        "step": stage,
                        "question": question,
                        "task_kind": "recover" if previous else "route",
                        "agent_state": {"state_name": "active", "phase": "execution", "session_position": cell["session_position"]},
                        "history": [{"completed_step": value, "status": "complete"} for value in completed],
                        "plan": {"remaining_step": current_step},
                        "observed_evidence": [{"result_id": f"sealed_observation_{value}", "inspection_status": "inspected"} for value in range(len(completed))],
                        "governance": {"allowed_side_effect_classes": allowed, "call_allowed_side_effect_classes": allowed},
                        "resource_state": {"remaining_steps": 4 - stage, "unresolved_requirement_count": 1},
                        "source_state": {"source_ids": [f"sealed_asset_{index % 31:02d}"], "available_modalities": [cell["source_modality"]], "inventory_state": "known", "inspection_state": "partial_context", "schema_known": target != "inspect_structured_schema"},
                        "query_state": {"query_terms": current_step.casefold().split()[:24], "schema_known": target != "inspect_structured_schema"},
                        "previous_candidate_ids": previous,
                        "expansion_context": {"expansion_allowed": bool(previous), "expansion_round": 1 if previous else 0, "trigger": "wrong_tool" if previous else "none", "prior_candidate_ids": previous, "excluded_candidate_ids": previous, "unresolved_requirement": current_step},
                        "tool_registry": registry.as_dict(),
                        "legal_candidate_ids": legal,
                        "label": {"acceptable_tools": [target_id] if target_id else [], "ranked_tools": ([target_id] if target_id else []) + [value for value in legal if value != target_id], "hard_negative_tools": hard_negatives if target_id else list(legal), "label_source": "post-scaling-sealed-validator.v1"},
                        "accepted": True,
                        "evaluation_partition": "post_scaling_sealed",
                        "split_group_id": f"sealed-scenario-{workflow}-{style}-{pool_size}",
                        "question_template_id": f"{HOLDOUT_TEMPLATE_PREFIX}-{workflow}-stage-{stage}",
                        "matrix_cell": cell,
                        "matrix_cell_id": assignment["matrix_cell_id"],
                        "provenance": {
                            "corpus": HOLDOUT_VERSION,
                            "dataset_version": HOLDOUT_VERSION,
                            "matrix_version": HOLDOUT_MATRIX_VERSION,
                            "prompt_version": f"{HOLDOUT_TEMPLATE_PREFIX}.v1",
                            "model": "deterministic-sealed-fixture",
                            "artifact": "post-scaling-sealed-holdout",
                            "generated_at": FROZEN_AT,
                            "seed": assignment["seed"],
                            "validator_version": "post-scaling-sealed-validator.v1",
                            "feature_version": FEATURE_VERSION,
                            "registry_fingerprint": registry.fingerprint,
                            "trajectory_hash": digest({"assignment": assignment, "question": question, "step": current_step}),
                            "matrix_cell_id": assignment["matrix_cell_id"],
                            "source_lineage": f"{HOLDOUT_VERSION}:{workflow}:{stage}",
                            "source_row_hash": digest({"workflow": workflow, "style": style, "pool": pool_size, "stage": stage}),
                            "teacher_fallback_used": False,
                        },
                    }
                    report = validate_decision_state_v2(row)
                    if not report.valid:
                        raise RuntimeError(f"invalid sealed state {index}: {report.as_dict()}")
                    if set(previous) & set(legal):
                        raise RuntimeError("sealed recovery repeats previous candidates")
                    rows.append(row)
                    count_values = {
                        "registry_style": style,
                        "schema_style": schema_style,
                        "candidate_pool_size": pool_size,
                        "task_kind": row["task_kind"],
                        "workflow": workflow,
                    }
                    for name, value in count_values.items():
                        counts[name][str(value)] += 1
                    index += 1
    manifest = {
        "holdout_version": HOLDOUT_VERSION,
        "matrix_version": HOLDOUT_MATRIX_VERSION,
        "frozen_at": FROZEN_AT,
        "seed": HOLDOUT_SEED,
        "count": len(rows),
        "answer_present_count": sum(bool(row["label"]["acceptable_tools"]) for row in rows),
        "abstention_count": sum(not row["label"]["acceptable_tools"] for row in rows),
        "unique_registry_fingerprints": len({row["provenance"]["registry_fingerprint"] for row in rows}),
        "unique_matrix_cells": len({row["matrix_cell_id"] for row in rows}),
        "unique_templates": len({row["question_template_id"] for row in rows}),
        "rows_sha256": digest(rows),
        "source_hashes_sha256": digest(sorted(row["provenance"]["source_row_hash"] for row in rows)),
        "registry_hashes_sha256": digest(sorted(row["provenance"]["registry_fingerprint"] for row in rows)),
        "template_hashes_sha256": digest(sorted(row["question_template_id"] for row in rows)),
        "scenario_hashes_sha256": digest(sorted(row["scenario_id"] for row in rows)),
        "dimension_counts": {name: dict(sorted(values.items())) for name, values in counts.items()},
        "training_dataset_forbidden": DATASET_VERSION,
    }
    return rows, manifest
