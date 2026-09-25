"""BGE cross-encoder adapter; leaves thresholds and fallback to scoring."""
import math
from config.settings import Settings
from src.reranking.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name=None, device=None, *, model=None, batch_size=32):
        legacy = Settings()
        self.model_name, self.device = model_name or legacy.RERANKER_MODEL, device or legacy.DEVICE
        self.model, self.batch_size = model, batch_size

    def load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query, candidates, top_k=None):
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not candidates or top_k == 0:
            return []
        self.load_model()
        scores = self.model.predict([(query, row["text"]) for row in candidates], batch_size=self.batch_size)
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned the wrong number of scores")
        rows = [{**row, "rerank_score": float(score)} for row, score in zip(candidates, scores)]
        if any(not math.isfinite(row["rerank_score"]) for row in rows):
            raise ValueError("Reranker score must be finite")
        rows.sort(key=lambda row: (-row["rerank_score"], row["chunk_id"]))
        # top_k retained for callers of the legacy API; pipeline always scores all.
        return rows if top_k is None else rows[:top_k]

    def batch_rerank(self, queries, candidates_list, top_k=None):
        if len(queries) != len(candidates_list):
            raise ValueError("Mismatched queries and candidate lists")
        return [self.rerank(q, rows, top_k) for q, rows in zip(queries, candidates_list)]


BGEReranker = CrossEncoderReranker
