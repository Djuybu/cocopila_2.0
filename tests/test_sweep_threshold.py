"""Unit and integration tests for P2-03: Sweep threshold reranker & plateau analysis."""
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from src.scoring.sweep import (
    detect_threshold_plateau,
    sweep_reranker_thresholds,
)
from src.utils.io import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_scored_records():
    records = [
        {
            "id": "q1",
            "candidates": [
                {"chunk_id": "c1", "doc_id": "d1", "text": "chunk1", "score": 0.95},
                {"chunk_id": "c2", "doc_id": "d1", "text": "chunk2", "score": 0.75},
                {"chunk_id": "c3", "doc_id": "d2", "text": "chunk3", "score": 0.35},
                {"chunk_id": "c4", "doc_id": "d2", "text": "chunk4", "score": 0.10},
            ],
        },
        {
            "id": "q2",
            "candidates": [
                {"chunk_id": "c5", "doc_id": "d3", "text": "chunk5", "score": 0.88},
                {"chunk_id": "c6", "doc_id": "d3", "text": "chunk6", "score": 0.60},
                {"chunk_id": "c7", "doc_id": "d4", "text": "chunk7", "score": 0.45},
            ],
        },
    ]
    labels = [
        {"id": "q1", "relevant_chunks": ["c1", "c2"]},
        {"id": "q2", "relevant_chunks": ["c5"]},
    ]
    return records, labels


def test_detect_threshold_plateau_continuous():
    """Verify that contiguous plateau is correctly identified and median threshold is recommended."""
    data = {
        "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "macro_f2":  [0.50, 0.60, 0.75, 0.85, 0.85, 0.845, 0.70, 0.40],
    }
    df = pd.DataFrame(data)
    info = detect_threshold_plateau(df, threshold_col="threshold", metric_col="macro_f2", tolerance=0.01)

    assert info["best_f2"] == 0.85
    assert info["plateau_min_threshold"] == 0.4
    assert info["plateau_max_threshold"] == 0.6
    assert info["plateau_count"] == 3
    assert info["plateau_width"] == 0.2
    # Median of 0.4 and 0.6 is 0.5:
    assert info["recommended_threshold"] == 0.5


def test_detect_threshold_plateau_handles_isolated_spikes():
    """Verify that an isolated spike separated by a trough does not contaminate the main peak plateau."""
    data = {
        "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "macro_f2":  [0.845, 0.70, 0.72, 0.848, 0.850, 0.849, 0.60],
    }
    df = pd.DataFrame(data)
    info = detect_threshold_plateau(df, threshold_col="threshold", metric_col="macro_f2", tolerance=0.01)

    assert info["best_f2"] == 0.85
    # The segment containing the peak is [0.4, 0.5, 0.6], not 0.1
    assert info["plateau_min_threshold"] == 0.4
    assert info["plateau_max_threshold"] == 0.6
    assert info["plateau_count"] == 3


def test_sweep_reranker_thresholds_computes_all_metrics(synthetic_scored_records):
    records, labels = synthetic_scored_records
    thresholds = [0.1, 0.4, 0.7, 0.9]

    sweep_df, plateau_info, best_cfg = sweep_reranker_thresholds(
        scored_records=records,
        labels=labels,
        thresholds=thresholds,
        tolerance=0.01,
    )

    # Check required columns
    expected_cols = [
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
        assert col in sweep_df.columns

    assert len(sweep_df) == len(thresholds)
    assert plateau_info["best_f2"] > 0
    assert best_cfg["threshold"] in thresholds
    assert "zero_chunk_queries" in best_cfg

    # Verify monotonicity of chunk volume vs threshold
    sorted_by_th = sweep_df.sort_values("threshold")
    chunks = sorted_by_th["avg_chunks_per_query"].tolist()
    # At lower thresholds, more chunks are selected
    assert chunks[0] >= chunks[-1]


def test_sweep_threshold_zero_queries_count(synthetic_scored_records):
    records, labels = synthetic_scored_records
    # When threshold is higher than all scores in q2 (max is 0.88), q2 will have 0 chunks
    sweep_df, _, _ = sweep_reranker_thresholds(
        scored_records=records,
        labels=labels,
        thresholds=[0.90],
    )
    row = sweep_df.iloc[0]
    assert row["zero_chunk_queries"] == 1
    assert row["zero_chunk_ratio"] == 0.5


def test_cli_sweep_threshold_execution(synthetic_scored_records, tmp_path):
    records, labels = synthetic_scored_records
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    write_jsonl(run_dir / "candidates.jsonl", records)
    write_json(run_dir / "labels.json", labels)

    output_csv = run_dir / "threshold_sweep.csv"
    output_summary = run_dir / "summary.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sweep_threshold.py"),
        "--run-dir",
        str(run_dir),
        "--output-csv",
        str(output_csv),
        "--output-summary",
        str(output_summary),
        "--steps",
        "10",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed with output:\n{res.stderr}"

    assert output_csv.exists()
    assert output_summary.exists()

    df = pd.read_csv(output_csv)
    assert not df.empty
    assert "macro_f2" in df.columns
    assert "avg_chunks_per_query" in df.columns

    summary = read_json(output_summary)
    assert "plateau_info" in summary
    assert "recommended_config" in summary
    assert summary["plateau_info"]["best_f2"] > 0
