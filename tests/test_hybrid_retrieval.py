from __future__ import annotations

from fitz_tool.hybrid_retrieval import bm25_scores, reciprocal_rank_fusion


def test_bm25_rewards_exact_distinguishing_terms() -> None:
    scores = bm25_scores(
        "retrieve a weather forecast for berlin",
        ["current weather observations", "future weather forecast by city"],
    )
    assert scores[1] > scores[0]


def test_rank_fusion_can_break_dense_near_tie_with_lexical_evidence() -> None:
    fused = reciprocal_rank_fusion(
        [0.81, 0.80], [0.0, 4.0], lexical_weight=0.6
    )
    assert fused[1] > fused[0]
