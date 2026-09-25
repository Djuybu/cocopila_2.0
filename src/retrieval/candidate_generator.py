"""Retriever orchestration, optional query variants, and candidate fusion."""
from src.retrieval.base import normalize_candidates
from src.retrieval.fusion import HybridFusion, reciprocal_rank_fusion, union_candidates


class CandidateGenerator:
    def __init__(self, retrievers, *, method="rrf", rrf_k=60, alpha=0.7, query_expander=None):
        # retrievers: {source: (BaseRetriever instance, per-source top_k)}
        if not retrievers:
            raise ValueError("At least one retriever must be enabled")
        if method not in {"rrf", "cc", "union"}:
            raise ValueError(f"Unsupported fusion: {method}")
        self.retrievers = retrievers
        self.method, self.rrf_k, self.alpha = method, rrf_k, alpha
        self.query_expander = query_expander

    def generate(self, query, top_k):
        queries = list(dict.fromkeys([query] + (list(self.query_expander(query)) if self.query_expander else [])))
        rankings = []
        for source, (retriever, limit) in self.retrievers.items():
            variants = [normalize_candidates(retriever.retrieve(q, limit), source, limit) for q in queries]
            rows = variants[0] if len(variants) == 1 else reciprocal_rank_fusion(variants, self.rrf_k, limit)
            rankings.append(rows)
        if self.method == "rrf":
            return reciprocal_rank_fusion(rankings, self.rrf_k, top_k)
        if self.method == "cc":
            if set(self.retrievers) != {"dense", "bm25"}:
                raise ValueError("CC requires exactly dense and bm25 retrievers")
            by_source = dict(zip(self.retrievers, rankings))
            return HybridFusion(self.alpha).fuse(by_source["dense"], by_source["bm25"], top_k)
        return normalize_candidates(union_candidates(rankings), "union", top_k)
