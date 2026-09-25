"""Real Qdrant storage checks; vectors are supplied without downloading models."""
from pathlib import Path
import pytest

qdrant = pytest.importorskip("qdrant_client")

from src.data.indexer import QdrantIndexer
from src.retrieval.dense import DenseRetriever
from src.pipeline.retrieve import build_dense_index
from src.pipeline.full_pipeline import run_full_pipeline
from src.utils.io import read_json


def test_qdrant_roundtrip_and_language_filter(tmp_path):
    indexer = QdrantIndexer(tmp_path / "qdrant", "test", embedding_dim=2)
    try:
        indexer.create_collection()
        chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "text": "one", "language": "vi"},
            {"chunk_id": "c2", "doc_id": "d2", "text": "two", "language": "en"},
        ]
        assert indexer.index_documents(chunks, [[1., 0.], [0., 1.]]) == 2
        assert indexer.get_collection_info()["points_count"] == 2
        retriever = DenseRetriever(client=indexer.client, collection_name="test")
        result = retriever.search([1., 0.], indexer.client, "test", top_k=1)
        assert result[0]["chunk_id"] == "c1"
        assert result[0]["rank"] == 1
        assert retriever.search([1., 0.], indexer.client, "test", language_filter="en")[0]["chunk_id"] == "c2"
        with pytest.raises(FileExistsError):
            indexer.create_collection()
    finally:
        indexer.client.close()


def test_dense_pipeline_with_real_index(experiment, tmp_path, monkeypatch):
    experiment["retrieval"]["bm25"]["enabled"] = False
    experiment["retrieval"]["dense"] = {
        "enabled": True, "model": "BAAI/bge-m3", "device": "cpu",
        "index_dir": str(tmp_path / "dense"), "collection": "test",
        "dimension": 2, "max_seq_length": 128, "batch_size": 2,
        "text_key": "text", "query_prefix": "", "top_k": 3,
    }
    vectors = {"aspirin treatment": [1., 0.], "insulin diabetes": [0., 1.], "cardiac pressure": [-1., 0.]}
    monkeypatch.setattr(DenseRetriever, "encode_documents",
                        lambda self, docs, batch_size: [vectors[doc] for doc in docs])
    monkeypatch.setattr(DenseRetriever, "encode_query",
                        lambda self, query: [1., 0.] if query == "aspirin" else [0., 1.])
    build_dense_index(experiment)
    result = read_json(run_full_pipeline(experiment) / "predictions.json")
    assert result[0]["relevant_chunks"] == ["d1_c000"]
    assert result[1]["relevant_chunks"] == ["d2_c000"]
    with pytest.raises(FileExistsError):
        build_dense_index(experiment)
