"""Compatibility import; new code should import src.retrieval.bm25."""
from src.retrieval.bm25 import SparseRetriever

__all__ = ["SparseRetriever"]
