"""Fusion works on independent ranked lists and never reranks model pairs."""
from collections import defaultdict
from math import sqrt
from src.retrieval.base import normalize_candidates


def union_candidates(rankings):
    unique = {}
    for rows in rankings:
        for row in rows:
            previous = unique.get(row["chunk_id"])
            if previous and previous["doc_id"] != row["doc_id"]:
                raise ValueError("Conflicting document IDs for the same chunk")
            unique.setdefault(row["chunk_id"], dict(row))
    return list(unique.values())


def reciprocal_rank_fusion(rankings, k=60, top_k=None):
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    union = {row["chunk_id"]: row for row in union_candidates(rankings)}
    scores = defaultdict(float)
    for rows in rankings:
        seen = set()
        for position, row in enumerate(rows, 1):
            if row["chunk_id"] in seen:
                continue
            seen.add(row["chunk_id"])
            rank = row.get("rank", position)
            if not isinstance(rank, int) or rank < 1:
                raise ValueError("Rank must be a positive integer")
            scores[row["chunk_id"]] += 1.0 / (k + rank)
    rows = [{**row, "score": scores[key]} for key, row in union.items()]
    rows.sort(key=lambda row: (-row["score"], row["chunk_id"]))
    return normalize_candidates(rows, "rrf", len(rows) if top_k is None else top_k)


class HybridFusion:
    def __init__(self, alpha=0.7, fusion_method="cc"):
        if not 0 <= alpha <= 1 or fusion_method not in {"cc", "rrf"}:
            raise ValueError("Invalid fusion settings")
        self.alpha, self.fusion_method = alpha, fusion_method

    def normalize_scores(self, results, method="min_max"):
        if method not in {"min_max", "standard"}:
            raise ValueError("Unknown normalization")
        if not results:
            return []
        scores = [float(row["score"]) for row in results]
        if method == "min_max":
            offset, scale = min(scores), max(scores) - min(scores)
        else:
            offset = sum(scores) / len(scores)
            scale = sqrt(sum((s - offset) ** 2 for s in scores) / len(scores))
        return [{**row, "score": (float(row["score"]) - offset) / scale if scale else 0.0}
                for row in results]

    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        return reciprocal_rank_fusion([dense_results, sparse_results], k)

    def convex_combination(self, dense_results, sparse_results):
        union = {row["chunk_id"]: row for row in union_candidates([dense_results, sparse_results])}
        scores = defaultdict(float)
        for rows, weight in [(dense_results, self.alpha), (sparse_results, 1 - self.alpha)]:
            seen = set()
            for row in rows:
                if row["chunk_id"] not in seen:
                    scores[row["chunk_id"]] += weight * row["score"]
                    seen.add(row["chunk_id"])
        rows = [{**row, "score": scores[key]} for key, row in union.items()]
        rows.sort(key=lambda row: (-row["score"], row["chunk_id"]))
        return normalize_candidates(rows, "cc", len(rows))

    def fuse(self, dense_results, sparse_results, top_k=200):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if self.fusion_method == "rrf":
            return reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)
        return self.convex_combination(self.normalize_scores(dense_results),
                                       self.normalize_scores(sparse_results))[:top_k]
