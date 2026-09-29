"""Unit and integration tests for P2-05: Sweep max-N output cap & tradeoff analysis."""
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from src.scoring.ablation import analyze_precision_recall_tradeoff, sweep_max_output
from src.utils.io import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_scored_data():
    records = [
        {
            "id": "q1",
            "candidates": [
                {"chunk_id": "c1", "doc_id": "d1", "text": "chunk1", "score": 0.95},
                {"chunk_id": "c2", "doc_id": "d1", "text": "chunk2", "score": 0.85},
                {"chunk_id": "c3", "doc_id": "d2", "text": "chunk3", "score": 0.70},
                {"chunk_id": "c4", "doc_id": "d2", "text": "chunk4", "score": 0.65},
                {"chunk_id": "c5", "doc_id": "d2", "text": "chunk5", "score": 0.60},
            ],
        },
        {
            "id": "q2",
            "candidates": [
                {"chunk_id": "c6", "doc_id": "d3", "text": "chunk6", "score": 0.90},
                {"chunk_id": "c7", "doc_id": "d3", "text": "chunk7", "score": 0.75},
                {"chunk_id": "c8", "doc_id": "d4", "text": "chunk8", "score": 0.62},
            ],
        },
    ]
    labels = [
        {"id": "q1", "relevant_chunks": ["c1", "c2"]},
        {"id": "q2", "relevant_chunks": ["c6"]},
    ]
    return records, labels


def test_sweep_max_output_metrics(sample_scored_data):
    records, labels = sample_scored_data
    maximums = [1, 2, 3, 5]
    threshold = 0.50

    df = sweep_max_output(
        scored_records=records,
        labels=labels,
        threshold=threshold,
        fallback=0,
        maximums=maximums,
    )

    expected_cols = [
        "max_chunks",
        "threshold",
        "fallback",
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

    assert len(df) == len(maximums)


def test_max_reduces_avg_chunks(sample_scored_data):
    records, labels = sample_scored_data
    # Threshold 0.50 allows 5 chunks in q1 and 3 chunks in q2 (avg = 4.0)
    df = sweep_max_output(records, labels, threshold=0.50, fallback=0, maximums=[2, 5])
    df_sorted = df.sort_values(by="max_chunks")
    cap2 = df_sorted[df_sorted["max_chunks"] == 2].iloc[0]
    cap5 = df_sorted[df_sorted["max_chunks"] == 5].iloc[0]

    assert cap2["avg_chunks_per_query"] == 2.0
    assert cap5["avg_chunks_per_query"] == 4.0


def test_precision_improves_with_cap(sample_scored_data):
    records, labels = sample_scored_data
    # For q1 truth is {c1, c2}. If threshold is 0.50, selecting 5 chunks yields P = 2/5 = 0.40.
    # Capping at max_chunks=2 selects {c1, c2}, yielding P = 2/2 = 1.0!
    df = sweep_max_output(records, labels, threshold=0.50, fallback=0, maximums=[2, 5])
    df_sorted = df.sort_values(by="max_chunks")
    cap2 = df_sorted[df_sorted["max_chunks"] == 2].iloc[0]
    cap5 = df_sorted[df_sorted["max_chunks"] == 5].iloc[0]

    assert cap2["macro_precision"] > cap5["macro_precision"]


def test_analyze_precision_recall_tradeoff(sample_scored_data):
    records, labels = sample_scored_data
    df = sweep_max_output(records, labels, threshold=0.50, fallback=0, maximums=[1, 2, 5])
    report = analyze_precision_recall_tradeoff(df)

    assert "best_max_chunks" in report
    assert "best_macro_f2" in report
    assert "unlimited_baseline" in report
    assert "tradeoff" in report
    assert "analysis" in report
    assert report["tradeoff"]["precision_gain"] >= 0


def test_cli_sweep_max_output_execution(sample_scored_data, tmp_path):
    records, labels = sample_scored_data
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    write_jsonl(run_dir / "candidates.jsonl", records)
    write_json(run_dir / "labels.json", labels)

    output_csv = run_dir / "max_output_sweep.csv"
    output_json = run_dir / "max_output_tradeoff.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sweep_max_output.py"),
        "--run-dir",
        str(run_dir),
        "--threshold",
        "0.60",
        "--fallback",
        "0",
        "--max-values",
        "1,2,5",
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
    assert "max_chunks" in df.columns

    report = read_json(output_json)
    assert "best_max_chunks" in report
    assert "tradeoff" in report
