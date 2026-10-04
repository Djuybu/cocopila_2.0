"""P3-14: document pipeline holdout tuning, ablation and direct-doc fusion tests."""
import pytest

from src.p3.doc_pipeline import (
    ABLATION_COLUMNS,
    document_scores_for_query,
    load_doc_pipeline_config,
    macro_f2_doc,
    tune_doc_pipeline,
    write_doc_pipeline,
)


def scored_records():
    records = []
    for index in range(1, 6):
        records.append({"id": f"q{index}", "candidates": [
            {"chunk_id": f"c{index}", "doc_id": f"d{index}", "rerank_score": 0.9},
            {"chunk_id": f"x{index}", "doc_id": f"d{index + 10}", "rerank_score": 0.1}]})
    return records


def labels():
    return [{"id": f"q{index}", "relevant_docs": [f"d{index}"], "relevant_chunks": [f"c{index}"]}
            for index in range(1, 6)]


def direct_docs():
    return [{"id": f"q{index}", "candidates": [{"doc_id": f"d{index}", "score": 0.5},
                                               {"doc_id": f"d{index + 10}", "score": 0.5}]}
            for index in range(1, 6)]


def test_document_scores_direct_fusion_normalises_scales():
    rows = document_scores_for_query(
        [{"chunk_id": "c1", "doc_id": "d1", "rerank_score": 10.0},
         {"chunk_id": "c2", "doc_id": "d2", "rerank_score": 0.0}],
        [{"doc_id": "d1", "score": 0.0}, {"doc_id": "d2", "score": 100.0}], direct_weight=0.5)
    assert {row["doc_id"]: row["score"] for row in rows} == {
        "d1": pytest.approx(0.5), "d2": pytest.approx(0.5)}


def test_macro_f2_doc_matches_manual_calculation():
    doc_scores = {"q1": [{"doc_id": "d1", "score": 0.9}, {"doc_id": "d2", "score": 0.1}]}
    labels_by_id = {"q1": {"d1"}}
    # top-1 selects d1: F2 = 5*1/(4*1+1) = 1.0
    assert macro_f2_doc(doc_scores, labels_by_id, ["q1"], None, 1, 1) == pytest.approx(1.0)
    # selecting both docs: F2 = 5*1/(4*1+2) = 5/6
    assert macro_f2_doc(doc_scores, labels_by_id, ["q1"], None, 0, 2) == pytest.approx(5 / 6)
    with pytest.raises(ValueError):
        macro_f2_doc(doc_scores, labels_by_id, [], None, 0, 1)


def test_tune_doc_pipeline_reports_holdout_and_ablation():
    best, ablation = tune_doc_pipeline(scored_records(), labels(), n_folds=5, seed=42,
                                       fallbacks=[0, 1], maximums=[1, 2], direct_weights=(0.0,))
    assert list(ablation.columns) == list(ABLATION_COLUMNS)
    assert set(ablation["aggregation"]) == {"max", "mean_top2", "mean_top3", "weighted_max_mean_top3"}
    assert set(ablation["direct_doc_weight"]) == {0.0}
    assert best["holdout"]["n_folds"] == 5
    assert best["holdout"]["mean_f2_doc"] == pytest.approx(1.0)
    assert best["evaluation_scope"] == "kfold_out_of_fold"
    assert best["direct_doc_enabled"] is False


def test_direct_doc_ablation_only_when_records_are_provided():
    _, without = tune_doc_pipeline(scored_records(), labels(), n_folds=5, seed=42,
                                   fallbacks=[0, 1], maximums=[1, 2])
    assert set(without["direct_doc_weight"]) == {0.0}
    best, with_direct = tune_doc_pipeline(scored_records(), labels(), direct_doc_records=direct_docs(),
                                          n_folds=5, seed=42, fallbacks=[0, 1], maximums=[1, 2])
    assert set(with_direct["direct_doc_weight"]) == {0.0, 0.5}
    assert "direct_doc_enabled" in best


def test_write_and_load_doc_pipeline(tmp_path):
    best, _ = tune_doc_pipeline(scored_records(), labels(), n_folds=5, seed=42,
                                fallbacks=[0, 1], maximums=[1, 2])
    path = tmp_path / "best_doc_pipeline.yaml"
    payload = write_doc_pipeline(path, best)
    assert payload["target_stage"] == "P3-14_official_aggregation"
    loaded = load_doc_pipeline_config(path)
    assert loaded["doc_aggregation"] == best["doc_aggregation"]
    assert loaded["direct_doc_weight"] == best["direct_doc_weight"]
    with pytest.raises(FileExistsError):
        write_doc_pipeline(path, best)
