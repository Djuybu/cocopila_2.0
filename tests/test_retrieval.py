"""Behavior checks for retriever contracts and independent rank fusion."""
from types import SimpleNamespace
import pytest
from src.retrieval.base import BaseRetriever, normalize_candidates
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion, union_candidates, HybridFusion
from src.retrieval.candidate_generator import CandidateGenerator


def row(chunk_id, score=1, rank=1):
    return {"chunk_id": chunk_id, "doc_id": "d1", "score": score, "rank": rank, "text": chunk_id}


def test_bm25_contract_and_roundtrip(dataset, tmp_path):
    chunks = dataset[2]
    retriever = BM25Retriever(text_key="text")
    assert isinstance(retriever, BaseRetriever)
    retriever.build_index(chunks)
    result = retriever.retrieve("aspirin", 2)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "internal1"
    assert set(result[0]) >= {"chunk_id", "doc_id", "score", "rank", "source"}
    assert result[0]["rank"] == 1
    assert isinstance(result[0]["score"], float)
    assert retriever.retrieve("aspirin", 0) == []
    path = tmp_path / "bm25.json"
    retriever.save_index(path)
    assert BM25Retriever(path).retrieve("aspirin", 2) == result


def test_rrf_uses_rank_not_score():
    a = [row("a", 10000, 1), row("b", 20, 2)]
    b = [row("b", -900, 1), row("a", -1000, 2)]
    result = reciprocal_rank_fusion([a, b])
    assert len(result) == 2
    assert result[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert result[1]["score"] == result[0]["score"]


def test_fusion_deduplicates_within_and_across_sources():
    result = reciprocal_rank_fusion([[row("a"), row("a")], [row("a")]])
    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(2 / 61)
    assert len(union_candidates([[row("a")], [row("a"), row("b")]])) == 2


def test_cc_normalizes_and_does_not_mutate():
    dense = [row("a", .9), row("b", .1)]
    sparse = [row("b", 100), row("a", 0)]
    result = HybridFusion(alpha=.7).fuse(dense, sparse)
    assert result[0]["chunk_id"] == "a"
    assert result[0]["score"] == pytest.approx(.7)
    assert dense[0]["score"] == .9


def test_dense_contract_without_downloading_model():
    class Client:
        def query_points(self, **kwargs):
            assert kwargs["query"] == [1, 0]
            return SimpleNamespace(points=[SimpleNamespace(payload={"chunk_id": "a", "doc_id": "d1", "text": "text"}, score=.9)])
    retriever = DenseRetriever(client=Client())
    result = retriever.search([1, 0], retriever.client, "collection", top_k=1)
    assert result[0]["source"] == "dense"
    assert result[0]["rank"] == 1


def test_candidate_generator_query_expansion_and_top_k():
    class Retriever(BaseRetriever):
        def retrieve(self, query, top_k):
            return [row(query)][:top_k]
    generator = CandidateGenerator({"bm25": (Retriever(), 2)}, query_expander=lambda q: [q, "expanded"])
    result = generator.generate("original", 1)
    assert len(result) == 1
    assert "rerank_score" not in result[0]
    assert generator.generate("original", 0) == []


def test_invalid_candidate_or_top_k():
    with pytest.raises(ValueError):
        normalize_candidates([row("a", float("nan"))], "test", 1)
    with pytest.raises(ValueError):
        normalize_candidates([row("a")], "test", -1)
