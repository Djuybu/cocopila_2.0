"""Rerankers score query/candidate pairs; selection is a separate stage."""
from abc import ABC, abstractmethod
import math
from src.data.schema import unique_ids


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query, candidates, top_k=None):
        """Return candidates with rerank_score, without threshold filtering."""
        raise NotImplementedError


def validate_reranked_candidates(original, ranked):
    """Canonical pipeline must score all chunks without rewriting input evidence."""
    if unique_ids(original, "chunk_id") != unique_ids(ranked, "chunk_id"):
        raise ValueError("Reranking changed candidate coverage")
    by_id = {row["chunk_id"]: row for row in original}
    previous_score = math.inf
    for row in ranked:
        if any(row.get(key) != value for key, value in by_id[row["chunk_id"]].items() if key != "rerank_score"):
            raise ValueError("Reranking changed input identity/text/provenance")
        score = row.get("rerank_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score > previous_score:
            raise ValueError("Reranking scores must be finite and sorted descending")
        previous_score = score
