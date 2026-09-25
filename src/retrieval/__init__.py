"""Retrieval pipeline: dense search, sparse search, hybrid fusion, reranking, and aggregation."""

__author__ = "Dev B (AI Pipeline & Model Specialist)"

from src.retrieval.base import BaseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever, SparseRetriever
from src.retrieval.fusion import HybridFusion
from src.reranking.bge import CrossEncoderReranker
from src.scoring.doc_aggregation import DocumentAggregator

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridFusion",
    "CrossEncoderReranker",
    "DocumentAggregator",
]
