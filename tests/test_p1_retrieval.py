"""P1 cache, multilingual prefixes, provenance and pre-reranking recall."""
import numpy as np
import pytest
from contextlib import closing

from src.data.indexer import QdrantIndexer
from src.evaluation.candidate_recall import evaluate_candidate_recall
from src.retrieval.cache import CachedRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import union_candidates, reciprocal_rank_fusion


def candidate(chunk, doc="d", rank=1, score=1):
    return {"chunk_id": chunk, "doc_id": doc, "rank": rank, "score": score}


def test_union_preserves_independent_source_ranks_and_scores():
    rankings = {"bm25": [candidate("a", rank=1, score=100), candidate("b", rank=2, score=80)],
                "dense": [candidate("b", rank=1, score=.95), candidate("c", rank=2, score=.8)]}
    union = union_candidates(rankings)
    assert len(union) == 3
    b = union[1]
    assert (b["bm25_rank"], b["dense_rank"], b["bm25_score"], b["dense_score"]) == (2, 1, 80, .95)
    assert b["source_ranks"] == {"bm25": 2, "dense": 1}
    fused = reciprocal_rank_fusion(rankings, k=20)
    assert fused[0]["chunk_id"] == "b"
    assert fused[0]["fused_score"] == pytest.approx(1/22 + 1/21)
    assert fused[0]["bm25_score"] == 80
    assert "source_ranks" not in rankings["bm25"][0]


def test_union_rejects_conflicting_parents():
    with pytest.raises(ValueError, match="Conflicting"):
        union_candidates({"bm25": [candidate("a", "d1")], "dense": [candidate("a", "d2")]})


def test_rrf_ties_are_stable_independent_of_source_order():
    a, b = [candidate("a")], [candidate("b")]
    assert reciprocal_rank_fusion({"bm25": a, "dense": b}) == reciprocal_rank_fusion({"dense": b, "bm25": a})


def test_cache_is_bound_to_query_k_and_signature(tmp_path):
    class Retriever:
        calls = 0
        def retrieve(self, query, top_k):
            self.calls += 1
            return [candidate(query)][:top_k]
    retriever = Retriever()
    cache = CachedRetriever(retriever, tmp_path, {"corpus": "v1", "model": "m1"})
    assert cache.retrieve("q", 1) == cache.retrieve("q", 1)
    assert retriever.calls == 1 and cache.events[-1]["cache_hit"]
    cache.retrieve("other", 1)
    cache.retrieve("q", 2)
    CachedRetriever(retriever, tmp_path, {"corpus": "v2", "model": "m1"}).retrieve("q", 1)
    CachedRetriever(retriever, tmp_path, {"corpus": "v1", "model": "m2"}).retrieve("q", 1)
    assert retriever.calls == 5


def test_store_persists_uncached_results_without_recomputing(tmp_path):
    class Retriever:
        def retrieve(self, *args):
            raise AssertionError("Cache hit must not invoke the backend")
    cache = CachedRetriever(Retriever(), tmp_path, {"corpus": "v1"})
    cache.store("query", 1, [candidate("a")], .25)
    cache.store("query", 1, [candidate("other")], .5)
    assert cache.retrieve("query", 1)[0]["chunk_id"] == "a"
    assert cache.events[-1]["compute_seconds"] == .25


def test_e5_prefixes_queries_and_passages_in_all_languages():
    class Model:
        def __init__(self):
            self.calls = []
        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return np.array([1., 0.]) if isinstance(texts, str) else np.array([[1., 0.]])
    model = Model()
    retriever = DenseRetriever(model=model, query_prefix="query: ", document_prefix="passage: ")
    retriever.encode_query("心脏")
    retriever.encode_documents(["心脏疾病"], show_progress=False)
    assert model.calls[0][0] == "query: 心脏"
    assert model.calls[1][0] == ["passage: 心脏疾病"]
    assert all(kwargs["normalize_embeddings"] for _, kwargs in model.calls)


@pytest.mark.parametrize("distance", ["cosine", "dot"])
def test_qdrant_distance_and_schema(distance):
    QdrantClient = pytest.importorskip("qdrant_client").QdrantClient
    with closing(QdrantClient(location=":memory:")) as client:
        index = QdrantIndexer(None, "test", embedding_dim=2, client=client, distance=distance)
        index.create_collection()
        chunks = [{**candidate("a"), "text": "medical"}, {**candidate("b"), "text": "other"}]
        index.index_documents(chunks, [[1., 0.], [0., 1.]])
        retriever = DenseRetriever(client=client, collection_name="test")
        result = retriever.search([1., 0.], client, "test", top_k=1)
        assert result[0]["chunk_id"] == "a" and result[0]["rank"] == 1
        assert result[0]["source"] == "dense"


def test_unicode_bm25_is_opt_in_and_persisted(tmp_path):
    default = BM25Retriever()
    unicode = BM25Retriever(tokenizer="unicode_cjk", lowercase=True, text_key="text")
    assert default.tokenizer == "whitespace"
    unicode.build_index([{**candidate("a"), "text": "心脏疾病"}, {**candidate("b"), "text": "皮肤疾病"},
                         {**candidate("c"), "text": "其他问题"}])
    assert unicode.retrieve("心脏", 1)[0]["chunk_id"] == "a"
    unicode.save_index(tmp_path / "index.json")
    assert BM25Retriever(tmp_path / "index.json").tokenizer == "unicode_cjk"


def test_recall_uses_chunk_budget_for_documents_and_lists_misses():
    records = [{"id": "q1", "candidates": [candidate("a", "d1"), candidate("b", "d1"), candidate("c", "d2")]},
               {"id": "q2", "candidates": []}]
    labels = [{"id": "q1", "relevant_chunks": ["c"], "relevant_docs": ["d2"]},
              {"id": "q2", "relevant_chunks": ["z"], "relevant_docs": ["d3"]}]
    report = evaluate_candidate_recall(records, labels, [2, 3])
    assert report["per_query"]["q1"]["documents"]["recall@2"] == 0
    assert report["macro"]["chunks"]["recall@3"] == .5
    assert report["miss_all_chunk_queries"] == ["q2"]
    assert report["miss_all_doc_queries"] == ["q2"]


def test_recall_deduplicates_official_ids_and_checks_query_coverage():
    records = [{"id": "q", "candidates": [candidate("i1"), candidate("i2"), candidate("c")]}]
    labels = [{"id": "q", "relevant_chunks": ["c"], "relevant_docs": ["d"]}]
    report = evaluate_candidate_recall(records, labels, [2], {"i1": "a", "i2": "a"})
    assert report["macro"]["chunks"]["recall@2"] == 1
    with pytest.raises(ValueError, match="same query IDs"):
        evaluate_candidate_recall(records, [], [2])


def test_real_bm25_benchmark_and_standalone_harness(experiment, tmp_path):
    from src.pipeline.benchmark import run_retrieval_benchmark, evaluate_retrieval_files
    from src.utils.io import write_json, read_json
    labels = tmp_path / "labels.json"
    write_json(labels, [{"id": "q1", "relevant_chunks": ["d1_c000"], "relevant_docs": ["d1"]},
                        {"id": "q2", "relevant_chunks": ["d2_c000"], "relevant_docs": ["d2"]}])
    experiment["benchmark"] = {"owner": "Mai Ngọc Duy", "labels_path": str(labels),
                               "cutoffs": [1, 3], "build_indexes_if_missing": True,
                               "rrf_constants": [20, 60], "experiment_log_path": str(tmp_path / "log.csv")}
    experiment["retrieval_cache"] = {"enabled": True, "cache_dir": str(tmp_path / "cache")}
    output = run_retrieval_benchmark(experiment)
    summary = read_json(output / "benchmark.json")
    assert summary["methods"]["bm25"]["macro"]["chunks"]["recall@1"] == 1
    assert summary["methods"]["bm25"]["performance"]["timing_mode"] == "uncached_compute"
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
    # Legacy internal IDs require an explicit registry, never implicit guessing.
    import json
    records = [json.loads(line) for line in (output / "bm25/candidates.jsonl").read_text().splitlines()]
    assert records[0]["candidates"][0]["chunk_id"] == "internal1"
    assert evaluate_retrieval_files(output / "bm25/candidates.jsonl", labels,
                                    tmp_path / "standalone", [3],
                                    internal_to_official=read_json(output / "registry.json")["internal_to_official"])["macro"]["chunks"]["recall@3"] == 1
