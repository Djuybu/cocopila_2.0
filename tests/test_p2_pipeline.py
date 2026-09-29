"""Unit tests for P2 joint selector sweep, false negative analysis, and P3 handoff packaging."""
from pathlib import Path
import pytest
import yaml

from src.data.download import sha256_file
from src.data.loader import load_records
from src.pipeline.handoff import export_reranking_input
from src.pipeline.p2_pipeline import run_p2_full_pipeline
from src.scoring.sweep import (
    analyze_false_negatives,
    generate_p2_chunk_predictions,
    package_p2_to_p3_handoff,
    sweep_chunk_selector,
)
from src.submission.validator import SubmissionValidator
from src.utils.io import read_json, write_json, write_jsonl
from tests.test_handoff import handoff_config


class MockReranker:
    def __init__(self):
        pass

    def rerank(self, query, candidates):
        # Score higher if query token is in text
        ranked = []
        for c in candidates:
            score = 1.0 if query.strip() in c["text"] else 0.1
            ranked.append({**c, "rerank_score": score})
        return sorted(ranked, key=lambda x: -x["rerank_score"])


@pytest.fixture
def sample_scored_data():
    scored_records = [
        {
            "id": "q1",
            "candidates": [
                {"chunk_id": "c1", "doc_id": "d1", "text": "aspirin dosage", "rerank_score": 0.95},
                {"chunk_id": "c2", "doc_id": "d1", "text": "aspirin side effect", "rerank_score": 0.80},
                {"chunk_id": "c3", "doc_id": "d2", "text": "unrelated", "rerank_score": 0.20},
            ],
        },
        {
            "id": "q2",
            "candidates": [
                {"chunk_id": "c4", "doc_id": "d3", "text": "ibuprofen dose", "rerank_score": 0.90},
                {"chunk_id": "c5", "doc_id": "d3", "text": "ibuprofen info", "rerank_score": 0.50},
                {"chunk_id": "c6", "doc_id": "d4", "text": "unrelated text", "rerank_score": 0.10},
            ],
        },
    ]
    labels = [
        {"id": "q1", "relevant_chunks": ["c1", "c2"], "relevant_docs": ["d1"]},
        {"id": "q2", "relevant_chunks": ["c4", "c7"], "relevant_docs": ["d3"]},  # c7 is missing from candidates!
    ]
    registry = {
        "expected_query_ids": ["q1", "q2"],
        "doc_ids": ["d1", "d2", "d3", "d4"],
        "chunk_to_doc": {
            "c1": "d1", "c2": "d1", "c3": "d2",
            "c4": "d3", "c5": "d3", "c6": "d4", "c7": "d3"
        },
        "internal_to_official": {
            "c1": "c1", "c2": "c2", "c3": "c3",
            "c4": "c4", "c5": "c5", "c6": "c6", "c7": "c7"
        },
    }
    return scored_records, labels, registry


def test_sweep_chunk_selector_finds_optimal_f2(sample_scored_data):
    scored_records, labels, registry = sample_scored_data
    thresholds = [0.1, 0.4, 0.7, 0.85]
    fallbacks = [0, 1, 2]
    maximums = [2, 5]

    best_config, sweep_df, plateau_info = sweep_chunk_selector(
        scored_records=scored_records,
        labels=labels,
        thresholds=thresholds,
        fallbacks=fallbacks,
        maximums=maximums,
        internal_to_official=registry["internal_to_official"],
        chunk_to_doc=registry["chunk_to_doc"],
    )

    assert "chunk_threshold" in best_config
    assert "chunk_fallback" in best_config
    assert "chunk_max" in best_config
    assert best_config["macro_f2"] > 0.0
    assert not sweep_df.empty
    assert "plateau_count" in plateau_info
    assert plateau_info["best_f2"] == best_config["macro_f2"]


def test_analyze_false_negatives_correct_attribution(sample_scored_data):
    scored_records, labels, registry = sample_scored_data
    # Use threshold 0.85, fallback 1, max 2:
    # q1: c1 (0.95) selected, c2 (0.80) pruned by threshold -> P2_THRESHOLD_PRUNED
    # q2: c4 (0.90) selected. c7 is in labels but was never in candidates -> P1_RETRIEVAL_MISS!
    config = {
        "chunk_threshold": 0.85,
        "chunk_fallback": 1,
        "chunk_max": 2,
    }

    fn_df, summary = analyze_false_negatives(
        scored_records=scored_records,
        labels=labels,
        best_config=config,
        internal_to_official=registry["internal_to_official"],
        chunk_to_doc=registry["chunk_to_doc"],
    )

    assert summary["total_false_negatives"] == 2
    assert summary["p1_retrieval_miss"] == 1
    assert summary["p2_threshold_pruned"] == 1

    cats = dict(zip(fn_df["chunk_id"], fn_df["category"]))
    assert cats["c7"] == "P1_RETRIEVAL_MISS"
    assert cats["c2"] == "P2_THRESHOLD_PRUNED"


def test_package_p2_to_p3_handoff_structure(sample_scored_data, tmp_path):
    scored_records, labels, registry = sample_scored_data
    source_dir = tmp_path / "p2_run"
    source_dir.mkdir()
    write_jsonl(source_dir / "reranked.jsonl", scored_records)
    write_json(source_dir / "labels.json", labels)
    write_json(source_dir / "registry.json", registry)
    write_json(source_dir / "queries.json", [{"id": "q1", "text": "aspirin"}, {"id": "q2", "text": "ibuprofen"}])
    write_json(source_dir / "input_manifest.json", {
        "schema_version": "medical-rag-candidates-v1",
        "source_run_id": "p1_test",
        "fingerprints": {"test": "123"},
    })

    best_config = {
        "chunk_threshold": 0.7,
        "chunk_fallback": 1,
        "chunk_max": 3,
        "macro_f2": 0.85,
        "macro_f1": 0.80,
        "macro_precision": 0.75,
        "macro_recall": 0.90,
        "avg_chunks_per_query": 1.5,
    }
    _, sweep_df, _ = sweep_chunk_selector(scored_records, labels, thresholds=[0.7], fallbacks=[1], maximums=[3])
    fn_df, _ = analyze_false_negatives(scored_records, labels, best_config)

    p3_dir = tmp_path / "p2_to_p3_handoff"
    output = package_p2_to_p3_handoff(
        output_dir=p3_dir,
        source_run_dir=source_dir,
        best_config=best_config,
        sweep_df=sweep_df,
        fn_df=fn_df,
        archive_zip=True,
    )

    assert (output / "reranked.jsonl").exists()
    assert (output / "best_chunk_selector.yaml").exists()
    assert (output / "p2_selected_chunks.json").exists()
    assert (output / "threshold_sweep.csv").exists()
    assert (output / "fn_analysis.csv").exists()
    assert (output / "manifest.json").exists()
    assert (tmp_path / "p2_to_p3_handoff.zip").exists()

    manifest = read_json(output / "manifest.json")
    assert manifest["stage"] == "p2_to_p3_handoff"
    for filename, checksum in manifest["files"].items():
        assert sha256_file(output / filename) == checksum

    # Verify selected chunks are valid official IDs
    selected_data = read_json(output / "p2_selected_chunks.json")
    val = SubmissionValidator(
        expected_query_ids=registry["expected_query_ids"],
        doc_ids=set(registry["doc_ids"]),
        chunk_to_doc=registry["chunk_to_doc"],
    )
    for row in selected_data:
        for cid in row["relevant_chunks"]:
            assert cid in val.chunk_to_doc


def test_p2_full_pipeline_end_to_end(handoff_config, tmp_path):
    # Use real handoff export from P1
    handoff_bundle = export_reranking_input(handoff_config)

    p2_output = tmp_path / "p2_output_run"
    p3_output = tmp_path / "final_p3_handoff"

    mock_reranker = MockReranker()
    summary = run_p2_full_pipeline(
        handoff_dir=handoff_bundle,
        output_dir=p2_output,
        p3_handoff_dir=p3_output,
        reranker=mock_reranker,
        sweep_thresholds=[0.0, 0.5, 0.9],
        sweep_fallbacks=[0, 1],
        sweep_maximums=[2, 5],
        archive_zip=True,
    )

    assert summary["status"] == "success"
    assert (p3_output / "reranked.jsonl").exists()
    assert (p3_output / "best_chunk_selector.yaml").exists()
    assert (p3_output / "p2_selected_chunks.json").exists()
    assert (p3_output / "manifest.json").exists()
    assert Path(summary["p3_handoff"]["archive_zip"]).exists()
