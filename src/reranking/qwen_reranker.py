"""Qwen3-Reranker adapter with medical domain instruction support."""
import math
from typing import Any, List, Optional
from config.settings import Settings
from src.reranking.base import BaseReranker

DEFAULT_MEDICAL_INSTRUCTION = (
    "Given a medical query and a medical document passage, determine if the document is relevant to the query."
)


class QwenReranker(BaseReranker):
    """Adapter for Qwen3-Reranker-0.6B cross-encoder model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        *,
        model: Any = None,
        batch_size: int = 16,
        instruction: Optional[str] = DEFAULT_MEDICAL_INSTRUCTION,
    ):
        legacy = Settings()
        self.model_name = model_name or "Qwen/Qwen3-Reranker-0.6B"
        self.device = device or legacy.DEVICE
        self.model = model
        self.batch_size = batch_size
        self.instruction = instruction

    def load_model(self) -> None:
        """Lazily load sentence_transformers CrossEncoder for Qwen3 reranker."""
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Score candidates using Qwen3 reranker and sort descending by score."""
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not candidates or top_k == 0:
            return []

        self.load_model()
        formatted_query = f"{self.instruction} {query}" if self.instruction else query
        pairs = [(formatted_query, row["text"]) for row in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        if len(scores) != len(candidates):
            raise ValueError("Reranker returned the wrong number of scores")

        rows = [{**row, "rerank_score": float(score)} for row, score in zip(candidates, scores)]
        if any(not math.isfinite(row["rerank_score"]) for row in rows):
            raise ValueError("Reranker score must be finite")

        rows.sort(key=lambda row: (-row["rerank_score"], row["chunk_id"]))
        return rows if top_k is None else rows[:top_k]

    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[dict]],
        top_k: Optional[int] = None,
    ) -> List[List[dict]]:
        """Batch rerank multiple query-candidate sets."""
        if len(queries) != len(candidates_list):
            raise ValueError("Mismatched queries and candidate lists")
        return [self.rerank(q, rows, top_k) for q, rows in zip(queries, candidates_list)]
