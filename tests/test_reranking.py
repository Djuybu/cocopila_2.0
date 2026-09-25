import pytest
from src.reranking.bge import CrossEncoderReranker


class Model:
    def predict(self, pairs, batch_size):
        assert pairs == [("query", "one"), ("query", "two")]
        return [0.2, 0.9]


def test_reranking_scores_without_threshold_or_selection():
    rows = [{"chunk_id": "c1", "doc_id": "d1", "text": "one", "score": 100},
            {"chunk_id": "c2", "doc_id": "d2", "text": "two", "score": 1}]
    result = CrossEncoderReranker(model=Model()).rerank("query", rows)
    assert [row["chunk_id"] for row in result] == ["c2", "c1"]
    assert result[0]["rerank_score"] == .9
    assert result[0]["score"] == 1
    assert "rerank_score" not in rows[0]


def test_empty_candidates_do_not_load_model():
    assert CrossEncoderReranker().rerank("query", []) == []
