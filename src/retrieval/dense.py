"""BGE dense retrieval with lazy model loading and injectable test backends."""
from config.settings import Settings
from src.retrieval.base import BaseRetriever, normalize_candidates


class DenseRetriever(BaseRetriever):
    def __init__(self, model_name=None, device=None, *, client=None, collection_name=None,
                 model=None, query_prefix="", batch_size=32, max_seq_length=None):
        legacy = Settings()
        self.model_name = model_name or legacy.EMBEDDING_MODEL
        self.device = device or legacy.DEVICE
        self.model, self.client = model, client
        self.collection_name = collection_name or legacy.QDRANT_COLLECTION_NAME
        self.query_prefix, self.batch_size = query_prefix, batch_size
        self.max_seq_length = max_seq_length

    def _model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length
        return self.model

    def encode_query(self, query):
        return self._model().encode(self.query_prefix + query, normalize_embeddings=True).tolist()

    def encode_documents(self, documents, batch_size=32, show_progress=True):
        return self._model().encode(documents, batch_size=batch_size,
                                    show_progress_bar=show_progress, normalize_embeddings=True).tolist()

    def search(self, query_vector, qdrant_client, collection_name, top_k=100, language_filter=None):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not top_k:
            return []
        query_filter = None
        if language_filter is not None:
            from qdrant_client import models
            query_filter = models.Filter(must=[models.FieldCondition(
                key="language", match=models.MatchValue(value=language_filter))])
        if hasattr(qdrant_client, "query_points"):
            points = qdrant_client.query_points(
                collection_name=collection_name, query=query_vector, limit=top_k,
                query_filter=query_filter, with_payload=True).points
        else:
            points = qdrant_client.search(
                collection_name=collection_name, query_vector=query_vector, limit=top_k,
                query_filter=query_filter, with_payload=True)
        return normalize_candidates([{**point.payload, "score": point.score} for point in points], "dense", top_k)

    def retrieve(self, query, top_k):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if top_k == 0:
            return []
        if self.client is None:
            raise ValueError("DenseRetriever requires a Qdrant client")
        return self.search(self.encode_query(query), self.client, self.collection_name, top_k)

    def batch_search(self, queries, qdrant_client, collection_name, top_k=100):
        return [self.search(self.encode_query(query), qdrant_client, collection_name, top_k) for query in queries]
