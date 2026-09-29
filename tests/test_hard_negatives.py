"""Unit and integration tests for P2-07: Hard negative mining."""
from pathlib import Path
import subprocess
import sys
import pytest

from src.training.hard_negatives import (
    is_suspected_false_negative,
    jaccard_similarity,
    merge_training_data,
    mine_hard_negatives,
    validate_hard_negatives,
    write_hard_negatives,
)
from src.utils.io import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def candidate_pool_data():
    records = [
        {
            "id": "q1",
            "text": "aspirin headache relief",
            "candidates": [
                {"chunk_id": "c1", "doc_id": "d1", "text": "Aspirin for headaches.", "score": 0.95},
                {"chunk_id": "c2", "doc_id": "d2", "text": "Ibuprofen for headaches.", "score": 0.88},
                {"chunk_id": "c3", "doc_id": "d3", "text": "Aspirin for headaches and severe pain.", "score": 0.85},
                {"chunk_id": "c4", "doc_id": "d4", "text": "Acetaminophen dosage instructions.", "score": 0.70},
                {"chunk_id": "c5", "doc_id": "d5", "text": "Antibiotics for bacterial infections.", "score": 0.20},
            ],
        },
        {
            "id": "q2",
            "text": "type 2 diabetes drugs",
            "candidates": [
                {"chunk_id": "c6", "doc_id": "d6", "text": "Metformin first line for type 2 diabetes.", "score": 0.92},
                {"chunk_id": "c7", "doc_id": "d7", "text": "Insulin resistance in diabetes.", "score": 0.80},
                {"chunk_id": "c8", "doc_id": "d8", "text": "Hypertension therapy guidelines.", "score": 0.30},
            ],
        },
    ]
    labels = [
        {"id": "q1", "relevant_chunks": ["c1"]},
        {"id": "q2", "relevant_chunks": ["c6"]},
    ]
    return records, labels


def test_jaccard_similarity():
    s1 = "aspirin headache relief medicine"
    s2 = "aspirin headache relief tablet"
    score = jaccard_similarity(s1, s2)
    assert 0.5 <= score <= 0.8

    assert jaccard_similarity("exact same text", "exact same text") == 1.0
    assert jaccard_similarity("apples oranges", "dog cat") == 0.0


def test_mine_excludes_ground_truth(candidate_pool_data):
    records, labels = candidate_pool_data
    hard_negs, summary = mine_hard_negatives(records, labels, max_negatives_per_query=5)

    label_map = {l["id"]: set(l["relevant_chunks"]) for l in labels}
    for h in hard_negs:
        qid = h["query_id"]
        cid = h["chunk_id"]
        assert cid not in label_map[qid]

    assert summary["ground_truth_excluded"] == 2


def test_mine_respects_max_per_query(candidate_pool_data):
    records, labels = candidate_pool_data
    hard_negs, _ = mine_hard_negatives(records, labels, max_negatives_per_query=2)

    q1_negs = [h for h in hard_negs if h["query_id"] == "q1"]
    q2_negs = [h for h in hard_negs if h["query_id"] == "q2"]

    assert len(q1_negs) <= 2
    assert len(q2_negs) <= 2


def test_suspected_false_negative_filtered(candidate_pool_data):
    records, labels = candidate_pool_data
    # c3 "Aspirin reduces severe headache pain." has high overlap with c1 "Aspirin for headaches."
    # If similarity threshold is low (0.35), c3 will be filtered as suspected FN
    hard_negs, summary = mine_hard_negatives(
        records,
        labels,
        max_negatives_per_query=5,
        filter_near_duplicates=True,
        similarity_threshold=0.35,
    )
    assert summary["suspected_fn_excluded"] >= 1
    chunk_ids = [h["chunk_id"] for h in hard_negs]
    assert "c3" not in chunk_ids


def test_validate_hard_negatives_clean(candidate_pool_data):
    records, labels = candidate_pool_data
    hard_negs, _ = mine_hard_negatives(records, labels)
    val = validate_hard_negatives(hard_negs, labels)

    assert val["status"] == "clean"
    assert val["ground_truth_contamination"] == 0
    assert val["avg_hard_negative_score"] > 0


def test_validate_hard_negatives_contamination_detection(candidate_pool_data):
    records, labels = candidate_pool_data
    contaminated = [
        {"query_id": "q1", "chunk_id": "c1", "score": 0.95},  # c1 is ground truth!
    ]
    with pytest.raises(ValueError, match="Contamination error"):
        validate_hard_negatives(contaminated, labels)


def test_merge_training_data():
    base_pairs = [
        {"query_id": "q1", "chunk_id": "c1", "label": 1.0},
        {"query_id": "q1", "chunk_id": "c_rand1", "label": 0.0},
        {"query_id": "q1", "chunk_id": "c_rand2", "label": 0.0},
    ]
    hard_negs = [
        {"query_id": "q1", "chunk_id": "c_hard1", "label": 0.0},
        {"query_id": "q1", "chunk_id": "c_hard2", "label": 0.0},
    ]

    merged = merge_training_data(base_pairs, hard_negs, hard_neg_ratio=0.5)
    cids = [m["chunk_id"] for m in merged]

    assert "c1" in cids  # Positive preserved
    assert "c_hard1" in cids or "c_hard2" in cids  # Hard negative included


def test_cli_mine_hard_negatives_execution(candidate_pool_data, tmp_path):
    records, labels = candidate_pool_data
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    write_jsonl(run_dir / "candidates.jsonl", records)
    write_json(run_dir / "labels.json", labels)

    output_file = run_dir / "hard_negatives.jsonl"
    report_file = run_dir / "hard_negatives_report.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "mine_hard_negatives.py"),
        "--run-dir",
        str(run_dir),
        "--max-per-query",
        "2",
        "--output",
        str(output_file),
        "--output-report",
        str(report_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed:\n{res.stderr}"

    assert output_file.exists()
    assert report_file.exists()

    report = read_json(report_file)
    assert report["mining_summary"]["total_mined_hard_negatives"] > 0
    assert report["validation_summary"]["ground_truth_contamination"] == 0
