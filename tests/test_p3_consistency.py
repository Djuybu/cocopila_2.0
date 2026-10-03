"""P3-05: doc/chunk consistency analysis, policy and prediction merge tests."""
import pytest

from src.p3.consistency import (
    CONSISTENCY_COLUMNS,
    analyze_consistency,
    apply_policy,
    summarize_consistency,
)
from src.p3.predictions import build_predictions, load_chunk_selector_config

MAPPING = {"c1": "d1", "c2": "d2", "c3": "d1"}


def predictions():
    return [
        {"id": "q1", "relevant_chunks": ["c1"], "relevant_docs": ["d1"]},
        {"id": "q2", "relevant_chunks": ["c2"], "relevant_docs": ["d2", "d1"]},
        {"id": "q3", "relevant_chunks": ["c3"], "relevant_docs": ["d2"]},
    ]


def test_analyze_quantifies_both_directions():
    report = analyze_consistency(predictions(), MAPPING)
    assert list(report.columns) == list(CONSISTENCY_COLUMNS)
    assert report["consistent"].tolist() == [True, False, False]
    assert report.loc[1, "docs_without_chunk"] == ["d1"]
    assert report.loc[2, "chunks_without_doc"] == ["c3"]
    assert summarize_consistency(report) == {
        "query_count": 3, "queries_consistent": 1, "queries_inconsistent": 2,
        "total_unknown_chunks": 0, "total_chunks_without_doc": 1, "total_docs_without_chunk": 2,
    }


def test_unknown_chunk_is_flagged():
    report = analyze_consistency([{"id": "q", "relevant_chunks": ["cX"], "relevant_docs": []}], MAPPING)
    assert report.loc[0, "unknown_chunk_count"] == 1
    assert not report.loc[0, "consistent"]


def test_policy_none_is_normalized_and_others_are_configurable():
    assert apply_policy(predictions(), MAPPING, "none") == [
        {"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]},
        {"id": "q2", "relevant_docs": ["d2", "d1"], "relevant_chunks": ["c2"]},
        {"id": "q3", "relevant_docs": ["d2"], "relevant_chunks": ["c3"]},
    ]
    pruned_chunks = apply_policy(predictions(), MAPPING, "prune_chunks")
    assert pruned_chunks[1]["relevant_chunks"] == ["c2"] and pruned_chunks[2]["relevant_chunks"] == []
    added = apply_policy(predictions(), MAPPING, "add_parent_docs")
    assert added[2]["relevant_docs"] == ["d2", "d1"]
    pruned_docs = apply_policy(predictions(), MAPPING, "prune_docs")
    assert pruned_docs[1]["relevant_docs"] == ["d2"]


def test_policy_rejects_unknown_chunk_bad_policy_and_empty_mapping():
    with pytest.raises(ValueError):
        apply_policy([{"id": "q", "relevant_chunks": ["cX"], "relevant_docs": []}], MAPPING, "prune_chunks")
    with pytest.raises(ValueError):
        apply_policy(predictions(), MAPPING, "nope")
    with pytest.raises(ValueError):
        analyze_consistency(predictions(), {})


def test_build_predictions_keeps_branches_independent():
    scored = [{"id": "q1", "candidates": [
        {"chunk_id": "c1", "doc_id": "d1", "rerank_score": 0.9},
        {"chunk_id": "c2", "doc_id": "d2", "rerank_score": 0.1}]}]
    chunk_config = {"chunk_threshold": 0.5, "chunk_fallback": 0, "chunk_max": 5}
    doc_config = {"doc_threshold": None, "doc_fallback": 0, "doc_max": 1, "doc_aggregation": "max"}
    (result,) = build_predictions(scored, chunk_config, doc_config)
    assert result["relevant_chunks"] == ["c1"]
    assert result["relevant_docs"] == ["d1"]


def test_load_chunk_selector_config_reads_p2_section(tmp_path):
    path = tmp_path / "best_chunk_selector.yaml"
    path.write_text("chunk_selector:\n  chunk_threshold: 0.3\n  chunk_fallback: 2\n  chunk_max: 10\n",
                    encoding="utf-8")
    assert load_chunk_selector_config(path) == {"chunk_threshold": 0.3, "chunk_fallback": 2, "chunk_max": 10}
    path.write_text("metrics: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_chunk_selector_config(path)
