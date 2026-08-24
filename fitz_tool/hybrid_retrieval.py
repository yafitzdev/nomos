"""Small dependency-free lexical scoring and rank fusion utilities."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "this",
    "tool",
    "use",
    "using",
    "with",
}


def lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 2 and token not in STOPWORDS
    )


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Score one query against a small candidate registry with BM25."""

    tokenized = [lexical_tokens(document) for document in documents]
    if not tokenized:
        return []
    query_terms = set(lexical_tokens(query))
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequencies.update(set(tokens))
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    scores = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length_normalization = 1.5 * (1.0 - 0.75 + 0.75 * len(tokens) / average_length)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (len(tokenized) - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5)
            )
            score += inverse_frequency * frequency * 2.5 / (
                frequency + length_normalization
            )
        scores.append(score)
    return scores


def reciprocal_rank_fusion(
    dense_scores: Sequence[float],
    lexical_scores: Sequence[float],
    *,
    lexical_weight: float,
    rank_constant: float = 60.0,
) -> list[float]:
    if len(dense_scores) != len(lexical_scores):
        raise ValueError("dense and lexical score lengths must match")
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("lexical_weight must be between 0 and 1")
    if not dense_scores or lexical_weight == 0.0 or max(lexical_scores, default=0.0) <= 0:
        return list(map(float, dense_scores))
    dense_order = sorted(range(len(dense_scores)), key=lambda index: -dense_scores[index])
    lexical_order = sorted(
        range(len(lexical_scores)), key=lambda index: -lexical_scores[index]
    )
    dense_rank = {index: rank for rank, index in enumerate(dense_order, start=1)}
    lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
    return [
        (1.0 - lexical_weight) / (rank_constant + dense_rank[index])
        + lexical_weight / (rank_constant + lexical_rank[index])
        for index in range(len(dense_scores))
    ]
