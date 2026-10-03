"""P3-02/P3-03: document aggregation and ablation contract tests."""
import pytest

from src.p3.doc_aggregation import (
    ABLATION_COLUMNS,
    AGGREGATION_METHODS,
    aggregate_documents,
    aggregation_spec,
    choose_baseline,
    evaluate_aggregation,
    prepare_official_candidates,
    validate_query_coverage,
)


def chunks():
    return [
        {"chunk_id": "d1c1", "doc_id": "d1", "rerank_score": 0.9},
        {"chunk_id": "d1c2", "doc_id": "d1", "rerank_score": 0.3},
        {"chunk_id": "d2c1", "doc_id": "d2", "rerank_score": 0.7},
        {"chunk_id": "d2c2", "doc_id": "d2", "rerank_score": 0.7},
    ]


def scored_records():
    return [
        {"id": "q1", "candidates": [
            {"chunk_id": "d1c1", "doc_id": "d1", "rerank_score": 0.9},
            {"chunk_id": "d2c1", "doc_id": "d2", "rerank_score": 0.1}]},
        {"id": "q2", "candidates": [
            {"chunk_id": "d2c2", "doc_id": "d2", "rerank_score": 0.8},
            {"chunk_id": "d1c2", "doc_id": "d1", "rerank_score": 0.2}]},
    ]


def labels():
    return [{"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": []},
            {"id": "q2", "relevant_docs": ["d2"], "relevant_chunks": []}]


def test_max_aggregation_is_the_baseline():
    rows = aggregate_documents(chunks(), "max")
    assert [(row["doc_id"], row["score"]) for row in rows] == [("d1", 0.9), ("d2", 0.7)]
    assert rows[0]["best_chunk_id"] == "d1c1"


def test_mean_top_k_and_weighted_scores():
    mean2 = {row["doc_id"]: row["score"] for row in aggregate_documents(chunks(), "mean_top_k", k=2)}
    assert mean2 == {"d1": pytest.approx(0.6), "d2": pytest.approx(0.7)}
    weighted = {row["doc_id"]: row["score"] for row in aggregate_documents(chunks(), "weighted", k=2, weight=0.5)}
    assert weighted == {"d1": pytest.approx(0.75), "d2": pytest.approx(0.7)}


def test_aggregation_is_deterministic_and_duplicate_safe():
    rows = chunks()
    forward = [row["doc_id"] for row in aggregate_documents(rows, "max")]
    backward = [row["doc_id"] for row in aggregate_documents(list(reversed(rows)), "max")]
    assert forward == backward
    mean = {row["doc_id"]: row["score"] for row in aggregate_documents(rows + [rows[0]], "mean_top_k", k=2)}
    assert mean["d1"] == pytest.approx(0.6)


def test_ablation_reports_macro_f2_per_method():
    table = evaluate_aggregation(scored_records(), labels(), doc_max=10)
    assert list(table.columns) == list(ABLATION_COLUMNS)
    assert len(table) == len(AGGREGATION_METHODS)
    assert set(table["method"]) == {spec["name"] for spec in AGGREGATION_METHODS}
    assert (table["macro_f2"] > 0).all()
    # Both docs are selected per query: F2_doc = 5*1/(4*1+2) = 5/6.
    assert table.loc[table["method"] == "max", "macro_f2"].iloc[0] == pytest.approx(5 / 6)


def test_baseline_selection_is_deterministic():
    table = evaluate_aggregation(scored_records(), labels())
    best = choose_baseline(table)
    assert best["method"] in {spec["name"] for spec in AGGREGATION_METHODS}
    assert choose_baseline(table) == best
    assert aggregation_spec(best["method"])["name"] == best["method"]


def test_aggregation_rejects_unknown_method_and_coverage_mismatch():
    with pytest.raises(ValueError):
        aggregation_spec("nope")
    with pytest.raises(ValueError):
        evaluate_aggregation(scored_records(), [{"id": "qx", "relevant_docs": []}])
    with pytest.raises(ValueError):
        validate_query_coverage(scored_records(), [{"id": "q1", "relevant_docs": []}])


def test_prepare_official_candidates_collapses_rechunks():
    records = [{"id": "q1", "candidates": [
        {"chunk_id": "i1", "doc_id": "d1", "rerank_score": 0.2},
        {"chunk_id": "i2", "doc_id": "d1", "rerank_score": 0.9}]}]
    prepared = prepare_official_candidates(records, {"i1": "c1", "i2": "c1"}, {"c1": "d1"})
    assert [candidate["chunk_id"] for candidate in prepared["q1"]] == ["c1"]
    assert prepared["q1"][0]["rerank_score"] == 0.9
