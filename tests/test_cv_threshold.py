"""Unit and integration tests for P2-09: Cross-validation threshold tuning."""
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from src.scoring.cv_threshold import (
    bootstrap_query_split,
    cross_validate_threshold,
    kfold_query_split,
)
from src.utils.io import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_cv_records():
    # 6 queries to test K-fold (e.g. 3 folds of 2 queries each)
    records = []
    labels = []
    for i in range(1, 7):
        qid = f"q{i}"
        cands = [
            {"chunk_id": f"c_{i}_1", "doc_id": f"d_{i}", "text": f"text {i} 1", "score": 0.90},
            {"chunk_id": f"c_{i}_2", "doc_id": f"d_{i}", "text": f"text {i} 2", "score": 0.70},
            {"chunk_id": f"c_{i}_3", "doc_id": f"d_{i}_other", "text": f"text {i} 3", "score": 0.40},
            {"chunk_id": f"c_{i}_4", "doc_id": f"d_{i}_other", "text": f"text {i} 4", "score": 0.20},
        ]
        records.append({"id": qid, "candidates": cands})
        labels.append({"id": qid, "relevant_chunks": [f"c_{i}_1", f"c_{i}_2"]})

    return records, labels


def test_kfold_query_split_coverage():
    query_ids = [f"q{i}" for i in range(10)]
    splits = kfold_query_split(query_ids, n_folds=5, seed=123)

    assert len(splits) == 5
    seen_val = []
    for train_q, val_q in splits:
        assert len(set(train_q) & set(val_q)) == 0  # No leakage
        assert len(train_q) + len(val_q) == 10
        seen_val.extend(val_q)

    assert sorted(seen_val) == sorted(query_ids)


def test_bootstrap_query_split():
    query_ids = [f"q{i}" for i in range(10)]
    splits = bootstrap_query_split(query_ids, n_rounds=8, val_ratio=0.3, seed=42)

    assert len(splits) == 8
    for train_q, val_q in splits:
        assert len(set(train_q) & set(val_q)) == 0
        assert len(val_q) == 3


def test_cross_validate_threshold_kfold(synthetic_cv_records):
    records, labels = synthetic_cv_records
    thresholds = [0.1, 0.3, 0.5, 0.8]

    cv_df, report = cross_validate_threshold(
        scored_records=records,
        labels=labels,
        thresholds=thresholds,
        method="kfold",
        n_folds=3,
        seed=42,
    )

    expected_cols = [
        "threshold",
        "mean_f2",
        "std_f2",
        "min_f2",
        "max_f2",
        "mean_precision",
        "mean_recall",
        "avg_chunks_per_query",
        "stability_score",
    ]
    for col in expected_cols:
        assert col in cv_df.columns

    assert len(cv_df) == len(thresholds)
    assert "recommended_stable_threshold" in report
    assert "pure_peak_threshold" in report
    assert "out_of_fold_generalization" in report

    stable = report["recommended_stable_threshold"]
    assert stable["threshold"] in thresholds
    assert len(stable["confidence_interval_95"]) == 2
    assert stable["confidence_interval_95"][0] <= stable["confidence_interval_95"][1]


def test_cross_validate_threshold_bootstrap(synthetic_cv_records):
    records, labels = synthetic_cv_records
    cv_df, report = cross_validate_threshold(
        scored_records=records,
        labels=labels,
        thresholds=[0.3, 0.6],
        method="bootstrap",
        n_rounds=5,
        seed=100,
    )

    assert report["method"] == "bootstrap"
    assert len(cv_df) == 2
    assert report["n_splits"] == 5


def test_cli_cv_threshold_execution(synthetic_cv_records, tmp_path):
    records, labels = synthetic_cv_records
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    write_jsonl(run_dir / "candidates.jsonl", records)
    write_json(run_dir / "labels.json", labels)

    output_csv = run_dir / "cv_sweep.csv"
    output_json = run_dir / "cv_report.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "cv_threshold.py"),
        "--run-dir",
        str(run_dir),
        "--method",
        "kfold",
        "--n-folds",
        "3",
        "--steps",
        "5",
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed:\n{res.stderr}"

    assert output_csv.exists()
    assert output_json.exists()

    df = pd.read_csv(output_csv)
    assert not df.empty
    assert "std_f2" in df.columns

    report = read_json(output_json)
    assert "recommended_stable_threshold" in report
