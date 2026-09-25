"""Persist Qdrant vectors and BM25 snapshots under configured artifact paths."""
from uuid import uuid5, NAMESPACE_URL


class QdrantIndexer:
    def __init__(self, qdrant_path, collection_name, embedding_dim=1024, *, client=None):
        if client is None:
            from qdrant_client import QdrantClient
            client = QdrantClient(path=str(qdrant_path))
        self.client, self.collection_name = client, collection_name
        self.embedding_dim = embedding_dim

    def create_collection(self, recreate=False):
        from qdrant_client import models
        exists = self.client.collection_exists(self.collection_name)
        if exists and not recreate:
            raise FileExistsError(f"Collection already exists: {self.collection_name}")
        if exists:
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True)),
        )

    def index_documents(self, chunks, embeddings, batch_size=256):
        from qdrant_client import models
        if len(chunks) != len(embeddings) or batch_size < 1:
            raise ValueError("Mismatched vectors/chunks or invalid batch size")
        if len({row["chunk_id"] for row in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs")
        if any(len(vector) != self.embedding_dim for vector in embeddings):
            raise ValueError("Embedding dimension mismatch")
        for start in range(0, len(chunks), batch_size):
            points = [
                models.PointStruct(id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])),
                                   vector=vector, payload=chunk)
                for chunk, vector in zip(chunks[start:start + batch_size], embeddings[start:start + batch_size])
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(chunks)

    def build_bm25_index(self, chunks, save_path):
        from src.retrieval.bm25 import BM25Retriever
        retriever = BM25Retriever()
        retriever.build_index(chunks)
        retriever.save_index(save_path)

    def get_collection_info(self):
        return self.client.get_collection(self.collection_name).model_dump()
