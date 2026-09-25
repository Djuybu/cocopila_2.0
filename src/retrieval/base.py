"""Shared retrieval contract; legacy metadata is preserved."""
from abc import ABC, abstractmethod
import math
from typing import TypedDict


class Candidate(TypedDict):
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    source: str


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Candidate]:
        """Return unique candidates in rank order (rank starts at one)."""
        raise NotImplementedError


def normalize_candidates(rows, source, top_k):
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    result, seen = [], set()
    for row in rows:
        if len(result) >= top_k:
            break
        for key in ("chunk_id", "doc_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Invalid candidate {key}")
        if row["chunk_id"] in seen:
            continue
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError("Candidate score must be finite")
        seen.add(row["chunk_id"])
        result.append({**row, "score": score, "rank": len(result) + 1, "source": source})
    return result
