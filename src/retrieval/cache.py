"""Corpus/model/query-bound cache; cache timings are distinct from compute timings."""
import hashlib
import json
from pathlib import Path
import time
from src.retrieval.base import BaseRetriever
from src.utils.io import read_json, write_json


class CachedRetriever(BaseRetriever):
    def __init__(self, retriever, cache_dir, signature):
        self.retriever, self.cache_dir, self.signature = retriever, Path(cache_dir), signature
        self.events = []

    def store(self, query, top_k, candidates, compute_seconds):
        """Persist measured fresh results without replacing an existing cache entry."""
        key = {"signature": self.signature, "query": query, "top_k": top_k}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        try:
            write_json(path, {"key": key, "candidates": candidates, "compute_seconds": compute_seconds})
        except FileExistsError:
            if read_json(path)["key"] != key:
                raise ValueError("Retrieval cache key mismatch")

    def retrieve(self, query, top_k):
        key = {"signature": self.signature, "query": query, "top_k": top_k}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        start = time.perf_counter()
        if path.exists():
            saved = read_json(path)
            if saved["key"] != key:
                raise ValueError("Retrieval cache key mismatch")
            result, compute_seconds, hit = saved["candidates"], saved["compute_seconds"], True
        else:
            result = self.retriever.retrieve(query, top_k)
            compute_seconds, hit = time.perf_counter() - start, False
            saved = {"key": key, "candidates": result, "compute_seconds": compute_seconds}
            try:
                write_json(path, saved)
            except FileExistsError:
                # Another worker may have written this exact deterministic key.
                if read_json(path)["key"] != key:
                    raise
        self.events.append({"cache_hit": hit, "wall_seconds": time.perf_counter() - start,
                            "compute_seconds": compute_seconds})
        return result
