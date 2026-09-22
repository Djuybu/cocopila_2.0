"""Data ingestion pipeline: segmentation, chunking, and indexing."""

__author__ = "Dev A (Data & Infrastructure Engineer)"

from .segmentor import VietnameseSegmentor
from .chunker import SlidingWindowChunker
from .indexer import QdrantIndexer

__all__ = ["VietnameseSegmentor", "SlidingWindowChunker", "QdrantIndexer"]
