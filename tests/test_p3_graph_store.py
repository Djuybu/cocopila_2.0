"""P3-11/P3-12: graph store, GraphRetriever, recall and ablation tests."""
import pytest

from src.graph.retriever import (
    GraphRetriever,
    ablation_graph_vs_baseline,
    evaluate_graph_recall,
    rank_chunks_by_entities,
    union_candidate_records,
)
from src.graph.store import InMemoryGraphStore, Neo4jGraphStore, build_graph, load_graph, save_graph


def lexicon():
    return {"version": "test",
            "entities": {
                "nhoi_mau_co_tim": {"type": "disease", "name": "nhồi máu cơ tim",
                                    "aliases": ["nhồi máu cơ tim"], "abbreviations": ["NMCT"]},
                "statin": {"type": "treatment", "name": "statin", "aliases": ["statin"], "abbreviations": []},
                "dau_nguc": {"type": "symptom", "name": "đau ngực", "aliases": ["đau ngực"], "abbreviations": []}},
            "relations": {"TREATS": {"cues": ["điều trị"]}}}


def documents():
    return [{"doc_id": "d1", "title": "Doc 1"}, {"doc_id": "d2", "title": "Doc 2"}]


def chunks():
    return [{"chunk_id": "c1", "doc_id": "d1", "text": "NMCT điều trị bằng statin"},
            {"chunk_id": "c2", "doc_id": "d1", "text": "đau ngực khi gắng sức"},
            {"chunk_id": "c3", "doc_id": "d2", "text": "statin và chế độ ăn"}]


def entities():
    return [
        {"chunk_id": "c1", "doc_id": "d1", "surface": "NMCT", "canonical_id": "nhoi_mau_co_tim",
         "canonical_name": "nhồi máu cơ tim", "canonical_type": "disease"},
        {"chunk_id": "c1", "doc_id": "d1", "surface": "statin", "canonical_id": "statin",
         "canonical_name": "statin", "canonical_type": "treatment"},
        {"chunk_id": "c2", "doc_id": "d1", "surface": "đau ngực", "canonical_id": "dau_nguc",
         "canonical_name": "đau ngực", "canonical_type": "symptom"},
        {"chunk_id": "c3", "doc_id": "d2", "surface": "statin", "canonical_id": "statin",
         "canonical_name": "statin", "canonical_type": "treatment"},
    ]


def relations():
    return [{"edge_type": "TREATS", "source": "statin", "target": "nhoi_mau_co_tim",
             "evidence": "NMCT điều trị bằng statin", "provenance": {"chunk_id": "c1", "doc_id": "d1"}}]


def store():
    return build_graph(InMemoryGraphStore(), documents(), chunks(), entities(), relations())


def test_build_graph_nodes_edges_and_stats():
    built = store()
    assert built.stats()["chunk_count"] == 3
    assert built.stats()["entity_count"] == 3
    assert built.chunks_for_entity("nhoi_mau_co_tim") == [{"chunk_id": "c1", "doc_id": "d1"}]
    assert built.related_entities("statin") == ["nhoi_mau_co_tim"]


def test_build_graph_rejects_relations_without_evidence():
    with pytest.raises(ValueError):
        build_graph(InMemoryGraphStore(), documents(), chunks(), entities(),
                    [{"edge_type": "TREATS", "source": "statin", "target": "nhoi_mau_co_tim",
                      "provenance": {"chunk_id": "c1"}}])


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "graph.json"
    save_graph(path, store())
    reloaded = load_graph(path)
    assert reloaded.stats()["edge_count"] == store().stats()["edge_count"]
    assert reloaded.related_entities("nhoi_mau_co_tim") == ["statin"]


def test_rank_chunks_direct_and_expanded():
    built = store()
    assert [row["chunk_id"] for row in rank_chunks_by_entities(built, ["nhoi_mau_co_tim"], 10)] == ["c1"]
    expanded = rank_chunks_by_entities(built, ["nhoi_mau_co_tim"], 10, expand=True)
    assert expanded[0] == {"chunk_id": "c1", "doc_id": "d1", "score": 1.5}
    assert expanded[1] == {"chunk_id": "c3", "doc_id": "d2", "score": 0.5}


def test_graph_retriever_returns_p1_candidate_contract():
    retriever = GraphRetriever(store(), lexicon())
    candidates = retriever.retrieve("bệnh nhân NMCT", 10)
    assert candidates[0]["chunk_id"] == "c1" and candidates[0]["doc_id"] == "d1"
    assert candidates[0]["source"] == "graph" and candidates[0]["rank"] == 1
    assert retriever.retrieve("không có thực thể nào", 10) == []
    assert retriever.retrieve("NMCT", 1)[0]["rank"] == 1


def test_graph_recall_and_ablation():
    retriever = GraphRetriever(store(), lexicon())
    queries = [{"id": "q1", "text": "NMCT"}, {"id": "q2", "text": "statin"}]
    labels = [{"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]},
              {"id": "q2", "relevant_docs": ["d2"], "relevant_chunks": ["c3"]}]
    report = evaluate_graph_recall(retriever, queries, labels, ks=(5,))
    assert report["source"] == "graph" and report["expand_relations"] is False
    assert report["macro"]["chunks"]["recall@5"] == 1.0
    assert report["macro"]["documents"]["recall@5"] == 1.0
    baseline = [{"id": "q1", "candidates": []}, {"id": "q2", "candidates": []}]
    ablation = ablation_graph_vs_baseline(retriever, queries, labels, baseline_records=baseline, ks=(5,))
    assert set(ablation) == {"graph_only", "graph_plus_baseline"}


def test_union_candidate_records_dedups_by_chunk():
    first = [{"id": "q1", "candidates": [{"chunk_id": "c1", "doc_id": "d1", "score": 1.0, "rank": 1, "source": "bm25"}]}]
    second = [{"id": "q1", "candidates": [{"chunk_id": "c1", "doc_id": "d1", "score": 2.0, "rank": 1, "source": "graph"},
                                          {"chunk_id": "c2", "doc_id": "d2", "score": 1.0, "rank": 2, "source": "graph"}]}]
    merged = union_candidate_records(first, second)
    assert [candidate["chunk_id"] for candidate in merged[0]["candidates"]] == ["c1", "c2"]
    assert merged[0]["candidates"][0]["source"] == "union"


class _FakeSession:
    def __init__(self, log):
        self.log = log

    def run(self, query, **params):
        self.log.append((query, params))
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self):
        self.queries, self.closed = [], False

    def session(self, database=None):
        return _FakeSession(self.queries)

    def close(self):
        self.closed = True


def test_neo4j_backend_builds_cypher_with_injected_driver():
    driver = _FakeDriver()
    graph = Neo4jGraphStore(uri="bolt://example", driver=driver)
    graph.create_constraints()
    graph.upsert_node({"node_id": "c1", "node_type": "Chunk", "properties": {"doc_id": "d1"}})
    graph.upsert_node({"node_id": "e1", "node_type": "Disease", "properties": {}})
    graph.upsert_edge({"source": "c1", "target": "e1", "edge_type": "MENTIONS",
                       "provenance": {"chunk_id": "c1"}})
    statements = " ".join(query for query, _ in driver.queries)
    assert "CREATE CONSTRAINT" in statements
    assert "MERGE (n:Chunk {node_id: $node_id})" in statements
    assert "MERGE (n:Entity:Disease {node_id: $node_id})" in statements
    assert "MERGE (a:Chunk {node_id: $source})" in statements
    graph.close()
    assert driver.closed is True


def test_neo4j_backend_requires_driver_when_not_injected():
    try:
        import neo4j  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError):
            Neo4jGraphStore(uri="bolt://localhost:7687")
    else:
        pytest.skip("neo4j driver installed; server integration is environment-specific")
