"""BM25Okapi baseline with explicit tokenization and JSON index snapshots."""
from src.retrieval.base import BaseRetriever, normalize_candidates
from src.utils.io import read_json, write_json


class BM25Retriever(BaseRetriever):
    def __init__(self, index_path=None, *, k1=1.5, b=0.75, epsilon=0.25,
                 lowercase=False, text_key="segmented_text"):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.lowercase, self.text_key = lowercase, text_key
        self.corpus, self.model = [], None
        if index_path is not None:
            self.load_index(index_path)

    def tokenize_for_bm25(self, text):
        # Word-segmented input is supported; no implicit medical normalization.
        return (text.lower() if self.lowercase else text).split()

    def build_index(self, corpus, text_key=None):
        from rank_bm25 import BM25Okapi
        self.text_key = text_key or self.text_key
        self.corpus = [dict(row) for row in corpus]
        normalize_candidates([{**row, "score": 0} for row in corpus], "bm25", len(corpus))
        if len({row["chunk_id"] for row in corpus}) != len(corpus):
            raise ValueError("Duplicate chunk_id in BM25 corpus")
        tokens = [self.tokenize_for_bm25(row[self.text_key]) for row in self.corpus]
        self.model = (BM25Okapi(tokens, k1=self.k1, b=self.b, epsilon=self.epsilon)
                      if any(tokens) else None)

    def retrieve(self, query, top_k):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not self.model or not top_k:
            return []
        scores = self.model.get_scores(self.tokenize_for_bm25(query))
        ordered = sorted(zip(self.corpus, scores), key=lambda item: (-float(item[1]), item[0]["chunk_id"]))
        return normalize_candidates([{**row, "score": float(score)} for row, score in ordered], "bm25", top_k)

    def search(self, query, top_k=100):
        return self.retrieve(query, top_k)

    def save_index(self, path):
        write_json(path, {
            "format": "medical-bm25-v1", "corpus": self.corpus,
            "parameters": {"k1": self.k1, "b": self.b, "epsilon": self.epsilon,
                           "lowercase": self.lowercase, "text_key": self.text_key},
        })

    def load_index(self, path):
        data = read_json(path)
        if data.get("format") != "medical-bm25-v1":
            raise ValueError("Unsupported BM25 snapshot (legacy pickle was never implemented)")
        params = data["parameters"]
        self.k1, self.b, self.epsilon = params["k1"], params["b"], params["epsilon"]
        self.lowercase, self.text_key = params["lowercase"], params["text_key"]
        self.build_index(data["corpus"])


SparseRetriever = BM25Retriever
