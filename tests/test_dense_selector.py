from __future__ import annotations

import numpy as np

from fitz_tool.agentic_pilot import generate_agentic_states
from fitz_tool.dense_selector import DenseToolRanker


class KeywordEncoder:
    prompts: dict[str, str] = {}

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("search" in lowered or "retrieve" in lowered),
                    float("final" in lowered or "select" in lowered),
                    0.1,
                ]
            )
        values = np.asarray(vectors, dtype=float)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_dense_tool_ranker_returns_only_legal_non_repeated_tools() -> None:
    row = generate_agentic_states(1)[0][0]
    row["question"] = "Search and retrieve the relevant evidence."
    row["plan"] = {"remaining_step": "Search the source now."}
    row["task_kind"] = "recover"
    row["previous_candidate_ids"] = [row["legal_candidate_ids"][0]]
    ranker = DenseToolRanker(KeywordEncoder())

    ranked = ranker.rank(row)

    assert ranked
    assert {item["tool_id"] for item in ranked}.issubset(row["legal_candidate_ids"])
    assert row["previous_candidate_ids"][0] not in {
        item["tool_id"] for item in ranked
    }
    assert ranker.version.endswith("dense-text.v3")
