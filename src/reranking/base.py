"""Rerankers score query/candidate pairs; selection is a separate stage."""
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query, candidates, top_k=None):
        """Return candidates with rerank_score, without threshold filtering."""
        raise NotImplementedError
