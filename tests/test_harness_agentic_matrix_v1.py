from __future__ import annotations

from collections import defaultdict

from fitz_tool.harness_agentic_matrix_v1 import generate_matrix_cells, load_matrix, validate_cell


def test_matrix_covers_every_capability_with_a_hard_contrast() -> None:
    spec = load_matrix()
    capabilities = set(spec["dimensions"]["target_capability"])
    covered = {value for pair in spec["contrast_pairs"] for value in pair}

    assert covered == capabilities
    assert {"find_paths", "search_content", "read_content", "execute_command"} <= capabilities
    assert {"request_user_input", "discover_capabilities", "expand_candidates"} <= capabilities


def test_generated_cells_are_balanced_split_safe_and_valid() -> None:
    cells, report = generate_matrix_cells(1900, seed=53)
    pools_by_kind: defaultdict[str, set[int]] = defaultdict(set)

    assert report["unique_cell_ids"] == len(cells)
    assert set(report["target_counts"]) == set(load_matrix()["dimensions"]["target_capability"])
    assert all(value > 0 for value in report["task_pool_combinations"].values())
    for cell in cells:
        partition = cell["evaluation_partition"]
        pools_by_kind[cell["task_kind"]].add(cell["candidate_pool_size"])
        assert validate_cell(cell) == []
        assert cell["registry_profile"].startswith(f"{partition}_")
        assert cell["question_template_group"].startswith(f"{partition}_")
        assert cell["scenario_group_id"].startswith(f"{partition}_")
        assert cell["tool_identity_group"].startswith(f"{partition}_")
    assert all(pools == {10, 25, 50, 100} for pools in pools_by_kind.values())


def test_recovery_cells_require_accumulated_no_repeat_memory() -> None:
    cells, _report = generate_matrix_cells(500, seed=59)
    recovery = [cell for cell in cells if cell["task_kind"] == "recover"]

    assert recovery
    assert all(cell["candidate_memory"] == "accumulated_no_repeat" for cell in recovery)
    assert all(cell["recovery_trigger"] != "none" for cell in recovery)
    assert all(cell["prior_candidate_count"] > 0 for cell in recovery)
