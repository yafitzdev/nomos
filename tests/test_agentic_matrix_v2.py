from __future__ import annotations

from collections import defaultdict

from fitz_tool.agentic_matrix_v2 import generate_matrix_cells, validate_cell


def test_v2_matrix_decouples_task_kind_and_pool_size() -> None:
    cells, report = generate_matrix_cells(1200)
    assert all(value > 0 for value in report["task_pool_combinations"].values())
    pools_by_kind = defaultdict(set)
    for cell in cells:
        pools_by_kind[cell["task_kind"]].add(cell["candidate_pool_size"])
        assert validate_cell(cell) == []
    assert all(pools == {5, 10, 30, 100} for pools in pools_by_kind.values())


def test_v2_matrix_freezes_registry_and_templates_by_split() -> None:
    cells, report = generate_matrix_cells(1200)
    assert report["unique_cell_ids"] == len(cells)
    for cell in cells:
        partition = cell["evaluation_partition"]
        assert cell["registry_profile"].startswith(f"{partition}_")
        assert cell["question_template_group"].startswith(f"{partition}_")
        assert cell["scenario_group_id"].startswith(f"{partition}_")


def test_v2_matrix_balances_valid_and_invalid_verification() -> None:
    cells, _report = generate_matrix_cells(1100)
    verify = [cell for cell in cells if cell["task_kind"] == "verify"]
    valid = [cell for cell in verify if cell["validation_case"] == "valid_call"]
    invalid = [cell for cell in verify if cell["validation_case"] != "valid_call"]
    assert valid
    assert invalid
    assert all(cell["expected_action"] == "accept_tool_call" for cell in valid)
    assert all(cell["expected_action"] == "reject_tool_call" for cell in invalid)
