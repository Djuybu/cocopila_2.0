"""Unit and integration tests for P2-06: Training data generation & leakage check."""
from pathlib import Path
import subprocess
import sys
import pytest

from src.training.data_generator import (
    generate_reranker_pairs,
    report_data_statistics,
    validate_no_leakage,
    write_training_data,
)
from src.utils.io import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def corpus_and_queries():
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "Aspirin reduces pain and fever."},
        {"chunk_id": "c2", "doc_id": "d1", "text": "Aspirin dosage guidelines for adults."},
        {"chunk_id": "c3", "doc_id": "d2", "text": "Insulin therapy for type 1 diabetes."},
        {"chunk_id": "c4", "doc_id": "d3", "text": "Metformin side effects and indications."},
        {"chunk_id": "c5", "doc_id": "d4", "text": "Cardiovascular risks in hypertension."},
    ]
    queries = [
        {"id": "q1", "text": "aspirin dosage", "split": "train"},
        {"id": "q2", "text": "diabetes treatment", "split": "train"},
        {"id": "q3", "text": "hypertension control", "split": "val"},
    ]
    labels = [
        {"id": "q1", "relevant_chunks": ["c1", "c2"]},
        {"id": "q2", "relevant_chunks": ["c3"]},
        {"id": "q3", "relevant_chunks": ["c5"]},
    ]
    return queries, chunks, labels


def test_generate_pairs_correct_count(corpus_and_queries):
    queries, chunks, labels = corpus_and_queries
    # For train: q1 has 2 positives, q2 has 1 positive -> 3 positives total
    # With neg_ratio=2 -> 3 * 2 = 6 negatives total -> 9 pairs total
    pairs = generate_reranker_pairs(
        queries=queries,
        chunks=chunks,
        labels=labels,
        target_split="train",
        neg_ratio=2,
        seed=42,
    )

    positives = [p for p in pairs if p["label"] == 1.0]
    negatives = [p for p in pairs if p["label"] == 0.0]

    assert len(positives) == 3
    assert len(negatives) <= 6
    assert len(pairs) == len(positives) + len(negatives)


def test_positive_pairs_properties(corpus_and_queries):
    queries, chunks, labels = corpus_and_queries
    pairs = generate_reranker_pairs(queries, chunks, labels, target_split="train")
    positives = [p for p in pairs if p["label"] == 1.0]

    for p in positives:
        assert p["label"] == 1.0
        assert len(p["query"]) > 0
        assert len(p["text"]) > 0
        assert "doc_id" in p


def test_negative_pairs_are_not_ground_truth(corpus_and_queries):
    queries, chunks, labels = corpus_and_queries
    pairs = generate_reranker_pairs(queries, chunks, labels, target_split="train")
    label_map = {l["id"]: set(l["relevant_chunks"]) for l in labels}

    for p in pairs:
        if p["label"] == 0.0:
            qid = p["query_id"]
            cid = p["chunk_id"]
            # Negative must never be in ground truth
            assert cid not in label_map[qid]


def test_validate_no_leakage_clean(corpus_and_queries):
    queries, chunks, labels = corpus_and_queries
    train_pairs = generate_reranker_pairs(queries, chunks, labels, target_split="train")
    val_pairs = generate_reranker_pairs(queries, chunks, labels, target_split="val")

    report = validate_no_leakage(train_pairs, val_pairs)
    assert report["status"] == "clean"
    assert report["leakage_count"] == 0


def test_validate_no_leakage_detects_query_overlap():
    train_pairs = [{"query_id": "q1", "chunk_id": "c1", "doc_id": "d1", "label": 1.0}]
    val_pairs = [{"query_id": "q1", "chunk_id": "c2", "doc_id": "d2", "label": 1.0}]

    with pytest.raises(ValueError, match="Query ID leakage detected"):
        validate_no_leakage(train_pairs, val_pairs)


def test_validate_no_leakage_detects_document_leakage():
    train_pairs = [{"query_id": "q1", "chunk_id": "c1", "doc_id": "d_shared", "label": 1.0}]
    val_pairs = [{"query_id": "q2", "chunk_id": "c2", "doc_id": "d_shared", "label": 1.0}]

    with pytest.raises(ValueError, match="Document leakage detected"):
        validate_no_leakage(train_pairs, val_pairs)


def test_report_data_statistics(corpus_and_queries):
    queries, chunks, labels = corpus_and_queries
    pairs = generate_reranker_pairs(queries, chunks, labels, target_split="train", neg_ratio=3)
    stats = report_data_statistics(pairs)

    assert stats["total_pairs"] > 0
    assert stats["num_positives"] > 0
    assert stats["num_negatives"] > 0
    assert "pos_neg_ratio" in stats
    assert stats["avg_query_words"] > 0


def test_cli_generate_training_data_execution(corpus_and_queries, tmp_path):
    queries, chunks, labels = corpus_and_queries
    write_json(tmp_path / "queries.json", queries)
    write_json(tmp_path / "chunks.json", chunks)
    write_json(tmp_path / "labels.json", labels)

    output_file = tmp_path / "reranker_train.jsonl"
    report_file = tmp_path / "stats.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_training_data.py"),
        "--data-dir",
        str(tmp_path),
        "--target-split",
        "train",
        "--neg-ratio",
        "2",
        "--output",
        str(output_file),
        "--report",
        str(report_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed:\n{res.stderr}"

    assert output_file.exists()
    assert report_file.exists()

    stats = read_json(report_file)
    assert stats["num_positives"] == 3
