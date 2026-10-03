"""P3-04: independent document-selector tuning tests."""
import pandas as pd
import pytest

from src.p3.doc_selector import (
    DEFAULT_FALLBACKS,
    DEFAULT_MAXIMUMS,
    default_doc_thresholds,
    doc_selector_config,
    load_doc_selector_config,
    select_best_doc_config,
    sweep_doc_selector,
    write_best_doc_selector,
)


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


def test_default_thresholds_mirror_p2_grid():
    thresholds = default_doc_thresholds([0.0, 1.0], steps=20)
    assert thresholds[0] is None and len(thresholds) == 20
    assert thresholds[1] == pytest.approx(0.05)
    assert thresholds[-1] == pytest.approx(0.95)
    assert default_doc_thresholds([0.5, 0.5]) == [None, 0.5]
    assert default_doc_thresholds([]) == [None, 0.0, 0.5]


def test_grid_defaults_match_p2_shape():
    assert list(DEFAULT_FALLBACKS) == [0, 1, 2, 3, 5]
    assert list(DEFAULT_MAXIMUMS) == [3, 5, 10, 15, 20]


def test_sweep_selects_top1_doc_and_never_exposes_chunk_keys():
    best, sweep = sweep_doc_selector(scored_records(), labels(), fallbacks=[0, 1], maximums=[1, 2])
    assert best["doc_max"] == 1
    assert best["macro_f2"] == pytest.approx(1.0)
    assert best["evaluation_scope"] == "tuning_on_supplied_labels"
    assert not any(key.startswith("chunk_") for key in best)
    assert set(sweep["doc_fallback"]) <= {0, 1}


def test_sweep_and_best_config_are_deterministic():
    first, sweep = sweep_doc_selector(scored_records(), labels(), fallbacks=[0, 1], maximums=[1, 2])
    second, _ = sweep_doc_selector(scored_records(), labels(), fallbacks=[0, 1], maximums=[1, 2])
    assert first == second
    assert select_best_doc_config(sweep) == first


def test_write_and_load_best_doc_selector(tmp_path):
    best, _ = sweep_doc_selector(scored_records(), labels(), fallbacks=[0, 1], maximums=[1, 2])
    path = tmp_path / "best_doc_selector.yaml"
    payload = write_best_doc_selector(path, best, baseline="max")
    assert payload["independent_of_chunk_selector"] is True
    loaded = load_doc_selector_config(path)
    assert loaded == doc_selector_config(best)
    assert loaded["doc_max"] == 1
    with pytest.raises(FileExistsError):
        write_best_doc_selector(path, best)


def test_config_validation_rejects_incomplete_and_empty():
    with pytest.raises(ValueError):
        doc_selector_config({"doc_max": 1})
    with pytest.raises(ValueError):
        select_best_doc_config(pd.DataFrame())
