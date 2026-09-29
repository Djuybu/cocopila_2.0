"""Unit and integration tests for P2-04: Sweep minimum fallback Top-N & ablation analysis."""
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from src.scoring.ablation import ablation_threshold_vs_fallback, sweep_fallback
from src.utils.io import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_scored_data():
    records = [
        {
            "id": "q1",
            "candidates": [
                {"chunk_id": "c1", "doc_id": "d1", "text": "chunk1", "score": 0.95},
                {"chunk_id": "c2", "doc_id": "d1", "text": "chunk2", "score": 0.70},
                {"chunk_id": "c3", "doc_id": "d2", "text": "chunk3", "score": 0.40},
                {"chunk_id": "c4", "doc_id": "d2", "text": "chunk4", "score": 0.15},
            ],
        },
        {
            "id": "q2",
            "candidates": [
                {"chunk_id": "c5", "doc_id": "d3", "text": "chunk5", "score": 0.60},
                {"chunk_id": "c6", "doc_id": "d3", "text": "chunk6", "score": 0.50},
                {"chunk_id": "c7", "doc_id": "d4", "text": "chunk7", "score": 0.30},
            ],
        },
    ]
    labels = [
        {"id": "q1", "relevant_chunks": ["c1", "c2"]},
        {"id": "q2", "relevant_chunks": ["c5"]},
    ]
    return records, labels


def test_sweep_fallback_all_metrics(sample_scored_data):
    records, labels = sample_scored_data
    fallbacks = [0, 1, 2, 3]
    threshold = 0.55

    df = sweep_fallback(
        scored_records=records,
        labels=labels,
        threshold=threshold,
        fallbacks=fallbacks,
    )

    expected_cols = [
        "fallback",
        "threshold",
        "macro_f2",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "avg_chunks_per_query",
        "min_chunks_per_query",
        "max_chunks_per_query",
        "zero_chunk_queries",
        "zero_chunk_ratio",
    ]
    for col in expected_cols:
        assert col in df.columns

    assert len(df) == len(fallbacks)
    # At least one configuration must have F2 > 0
    assert df["macro_f2"].max() > 0


def test_fallback_zero_returns_threshold_only(sample_scored_data):
    records, labels = sample_scored_data
    # With threshold 0.8, only c1 (0.95) passes for q1; q2 has max score 0.60 so 0 chunks pass
    df = sweep_fallback(records, labels, threshold=0.8, fallbacks=[0])
    row = df.iloc[0]
    assert row["fallback"] == 0
    assert row["zero_chunk_queries"] == 1
    assert row["zero_chunk_ratio"] == 0.5


def test_fallback_eliminates_zero_chunk_queries(sample_scored_data):
    records, labels = sample_scored_data
    # At high threshold 0.99, 0 chunks pass for both queries
    df_fb0 = sweep_fallback(records, labels, threshold=0.99, fallbacks=[0])
    assert df_fb0.iloc[0]["zero_chunk_queries"] == 2

    # With fallback=1, every query gets top-1 candidate
    df_fb1 = sweep_fallback(records, labels, threshold=0.99, fallbacks=[1])
    assert df_fb1.iloc[0]["zero_chunk_queries"] == 0
    assert df_fb1.iloc[0]["avg_chunks_per_query"] == 1.0


def test_ablation_threshold_vs_fallback(sample_scored_data):
    records, labels = sample_scored_data
    report = ablation_threshold_vs_fallback(
        scored_records=records,
        labels=labels,
        threshold=0.8,
        fallbacks=[0, 1, 2],
    )

    assert "baseline_threshold_only" in report
    assert "best_fallback_config" in report
    assert "delta" in report
    assert "verdict" in report
    assert "sweep_table" in report

    assert report["baseline_threshold_only"]["fallback"] == 0
    # Fallback 1 or 2 should rescue q2 and decrease zero-chunk queries
    assert report["delta"]["zero_queries_eliminated"] >= 1


def test_cli_sweep_fallback_execution(sample_scored_data, tmp_path):
    records, labels = sample_scored_data
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    write_jsonl(run_dir / "candidates.jsonl", records)
    write_json(run_dir / "labels.json", labels)

    output_csv = run_dir / "fallback_sweep.csv"
    output_json = run_dir / "fallback_ablation.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sweep_fallback.py"),
        "--run-dir",
        str(run_dir),
        "--threshold",
        "0.65",
        "--fallbacks",
        "0,1,2",
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
    assert len(df) == 3
    assert "macro_f2" in df.columns

    ablation = read_json(output_json)
    assert "baseline_threshold_only" in ablation
    assert "verdict" in ablation
