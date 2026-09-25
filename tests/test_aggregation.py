import pytest
from src.scoring.doc_aggregation import DocumentAggregator
from src.scoring.chunk_selector import select_ids, select_results


def chunks():
    return [{"chunk_id": "d1_c1", "doc_id": "d1", "rerank_score": 0.9, "score": 0.01},
            {"chunk_id": "d1_c2", "doc_id": "d1", "rerank_score": 0.5, "score": 10},
            {"chunk_id": "d2_c1", "doc_id": "d2", "rerank_score": 0.4, "score": 20}]


@pytest.mark.parametrize("method,expected", [("max", .9), ("max_p", .9), ("mean_top_k", .7), ("top_k_mean", .7), ("weighted", .8)])
def test_aggregation(method, expected):
    result = DocumentAggregator(method, k=2, weight=.5).score_documents(chunks())
    assert result[0]["doc_id"] == "d1"
    assert result[0]["score"] == pytest.approx(expected)


def test_threshold_fallback_and_maximum():
    rows = chunks()
    assert select_ids(rows, "chunk_id", .95, 2, 2) == ["d1_c1", "d1_c2"]
    assert select_ids(rows, "chunk_id", .6, 0, 2) == ["d1_c1"]
    assert select_ids(rows, "chunk_id", None, 0, 1) == ["d1_c1"]
    assert select_ids([], "chunk_id", .5, 1, 3) == []
    with pytest.raises(ValueError):
        select_ids(rows, "chunk_id", .5, 4, 2)


def test_doc_branch_independent_of_chunk_selection():
    config = {"chunk_threshold": 1, "chunk_fallback": 0, "chunk_max": 20,
              "doc_threshold": .8, "doc_fallback": 0, "doc_max": 10}
    result = select_results("q1", chunks(), config)
    assert result["relevant_chunks"] == []
    assert result["relevant_docs"] == ["d1"]


def test_duplicate_chunks_do_not_bias_mean():
    rows = chunks()
    result = DocumentAggregator("mean_top_k").score_documents(rows + [rows[0]])
    assert result[0]["score"] == pytest.approx(.7)


def test_rechunks_are_collapsed_before_fallback():
    from src.data.adapter import official_candidates
    rows = [{"chunk_id": "internal1", "doc_id": "d1", "score": .9},
            {"chunk_id": "internal2", "doc_id": "d1", "score": .8},
            {"chunk_id": "internal3", "doc_id": "d1", "score": .7}]
    mapped = official_candidates(rows, {"internal1": "c1", "internal2": "c1", "internal3": "c2"},
                                 {"c1": "d1", "c2": "d1"})
    assert select_ids(mapped, "chunk_id", .95, 2, 2) == ["c1", "c2"]
    assert mapped[0]["score"] == .9
