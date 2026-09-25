"""Document aggregation independent of chunk selection."""
from collections import defaultdict
import math


def candidate_score(row):
    score = float(row["rerank_score"] if "rerank_score" in row else row["score"])
    if not math.isfinite(score):
        raise ValueError("Scores must be finite")
    return score


class DocumentAggregator:
    def __init__(self, method="max_p", top_docs=10, *, k=3, weight=0.5):
        aliases = {"max_p": "max", "top_k_mean": "mean_top_k"}
        self.method = aliases.get(method, method)
        if self.method not in {"max", "mean_top_k", "weighted"}:
            raise ValueError("Unknown aggregation method")
        if k < 1 or top_docs < 0 or not 0 <= weight <= 1:
            raise ValueError("Invalid aggregation parameters")
        self.top_docs, self.k, self.weight = top_docs, k, weight

    def score_documents(self, chunks, method=None, k=None):
        method, k = method or self.method, self.k if k is None else k
        if method not in {"max", "mean_top_k", "weighted"} or k < 1:
            raise ValueError("Invalid aggregation method or k")
        groups = defaultdict(dict)
        for chunk in chunks:
            # Duplicate chunks cannot bias the mean.
            old = groups[chunk["doc_id"]].get(chunk["chunk_id"])
            if old is None or candidate_score(chunk) > candidate_score(old):
                groups[chunk["doc_id"]][chunk["chunk_id"]] = chunk
        documents = []
        for doc_id, group in groups.items():
            ordered = sorted(group.values(), key=lambda row: (-candidate_score(row), row["chunk_id"]))
            maximum = candidate_score(ordered[0])
            mean = sum(candidate_score(row) for row in ordered[:k]) / len(ordered[:k])
            score = maximum if method == "max" else mean if method == "mean_top_k" else self.weight * maximum + (1 - self.weight) * mean
            documents.append({"doc_id": doc_id, "score": score, "best_chunk_id": ordered[0]["chunk_id"]})
        return sorted(documents, key=lambda row: (-row["score"], row["doc_id"]))

    def max_p_aggregation(self, chunks):
        return self.score_documents(chunks, "max")

    def top_k_mean_aggregation(self, chunks, k=3):
        if k < 1:
            raise ValueError("k must be positive")
        return self.score_documents(chunks, "mean_top_k", k)

    def deduplicate(self, doc_ids):
        return list(dict.fromkeys(doc_ids))

    def aggregate(self, reranked_chunks):
        return ([row["doc_id"] for row in self.score_documents(reranked_chunks)[:self.top_docs]],
                self.deduplicate([row["chunk_id"] for row in sorted(reranked_chunks, key=candidate_score, reverse=True)]))
