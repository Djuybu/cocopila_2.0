"""Unit tests for P2-02 Chunk-level F2 Evaluation."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import pytest

from src.evaluation.evaluate_f2 import (
    compute_chunk_metrics,
    evaluate_candidate_selection,
    evaluate_chunk_predictions,
    export_metrics_csv,
    export_metrics_json,
    format_metrics_table,
)
from src.utils.io import read_json


TOY_REFERENCE_CSV = Path(__file__).parent / "fixtures" / "toy_chunk_f2_reference.csv"
HANDOFF_DIR = Path(__file__).resolve().parent.parent / "data" / "p1_p2_handoff_qwen3"


def test_compute_chunk_metrics_hand_calculated():
    """Verify single-query metrics on exact hand-calculated toy queries."""
    # toy_q1: T={c1, c2}, P={c1, c3} -> TP=1, |T|=2, |P|=2
    m1 = compute_chunk_metrics(["c1", "c2"], ["c1", "c3"])
    assert m1["tp"] == 1
    assert m1["fp"] == 1
    assert m1["fn"] == 1
    assert m1["precision"] == 0.5
    assert m1["recall"] == 0.5
    assert m1["f1"] == 0.5
    assert m1["f2"] == 0.5

    # toy_q2: T={c4}, P={c4, c5, c6} -> TP=1, |T|=1, |P|=3
    m2 = compute_chunk_metrics(["c4"], ["c4", "c5", "c6"])
    assert m2["tp"] == 1
    assert m2["fp"] == 2
    assert m2["fn"] == 0
    assert m2["precision"] == pytest.approx(1 / 3)
    assert m2["recall"] == 1.0
    assert m2["f1"] == 0.5
    assert m2["f2"] == pytest.approx(5 / 7)

    # toy_q3: T={c7, c8}, P={c9} -> TP=0
    m3 = compute_chunk_metrics(["c7", "c8"], ["c9"])
    assert m3["tp"] == 0
    assert m3["fp"] == 1
    assert m3["fn"] == 2
    assert m3["precision"] == 0.0
    assert m3["recall"] == 0.0
    assert m3["f1"] == 0.0
    assert m3["f2"] == 0.0

    # toy_q4: T={c10}, P={} -> TP=0
    m4 = compute_chunk_metrics(["c10"], [])
    assert m4["tp"] == 0
    assert m4["fp"] == 0
    assert m4["fn"] == 1
    assert m4["precision"] == 0.0
    assert m4["recall"] == 0.0
    assert m4["f1"] == 0.0
    assert m4["f2"] == 0.0


def test_evaluate_chunk_predictions_toy_macro():
    """Verify macro aggregation on the 4-query toy dataset."""
    ground_truth = [
        {"id": "toy_q1", "relevant_chunks": ["c1", "c2"]},
        {"id": "toy_q2", "relevant_chunks": ["c4"]},
        {"id": "toy_q3", "relevant_chunks": ["c7", "c8"]},
        {"id": "toy_q4", "relevant_chunks": ["c10"]},
    ]
    predictions = [
        {"id": "toy_q1", "relevant_chunks": ["c1", "c3"]},
        {"id": "toy_q2", "relevant_chunks": ["c4", "c5", "c6"]},
        {"id": "toy_q3", "relevant_chunks": ["c9"]},
        {"id": "toy_q4", "relevant_chunks": []},
    ]

    report = evaluate_chunk_predictions(predictions, ground_truth)
    macro = report["macro"]

    # Hand-derived exact fractions:
    # Precision: (1/2 + 1/3 + 0 + 0) / 4 = (5/6) / 4 = 5/24
    assert macro["precision"] == pytest.approx(5 / 24, abs=1e-6)
    # Recall: (1/2 + 1 + 0 + 0) / 4 = 1.5 / 4 = 3/8
    assert macro["recall"] == pytest.approx(3 / 8, abs=1e-6)
    # F1: (1/2 + 1/2 + 0 + 0) / 4 = 1/4
    assert macro["f1"] == pytest.approx(1 / 4, abs=1e-6)
    # F2: (1/2 + 5/7 + 0 + 0) / 4 = (17/14) / 4 = 17/56
    assert macro["f2"] == pytest.approx(17 / 56, abs=1e-6)


def test_f2_formula_mathematical_identity():
    """Verify equivalence between set-based formula and harmonic mean definition."""
    truth = {"a", "b", "c"}
    pred = {"a", "b", "d", "e"}
    tp = len(truth & pred)
    p = tp / len(pred)
    r = tp / len(truth)

    # Harmonic mean definition with beta=2:
    harmonic_f2 = (1 + 4) * (p * r) / (4 * p + r)

    # Set-based formula:
    set_f2 = (5 * tp) / (4 * len(truth) + len(pred))

    metrics = compute_chunk_metrics(truth, pred)
    assert metrics["f2"] == pytest.approx(harmonic_f2)
    assert metrics["f2"] == pytest.approx(set_f2)


def test_edge_cases():
    """Verify empty sets, duplicate IDs, zero-division, and query mismatch."""
    # Both sets empty
    m = compute_chunk_metrics([], [])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f2"] == 0.0

    # Custom zero division
    m_zd = compute_chunk_metrics([], [], zero_division=1.0)
    assert m_zd["precision"] == 1.0 and m_zd["f2"] == 1.0

    # Duplicates in predictions are deduplicated
    m_dup = compute_chunk_metrics(["c1"], ["c1", "c1", "c1"])
    assert m_dup["num_predicted"] == 1
    assert m_dup["tp"] == 1
    assert m_dup["precision"] == 1.0
    assert m_dup["recall"] == 1.0
    assert m_dup["f2"] == 1.0

    # Query ID mismatch raises ValueError
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_chunk_predictions([{"id": "q1", "relevant_chunks": []}],
                                   [{"id": "q2", "relevant_chunks": []}])


def test_evaluate_on_handoff_qwen3_top1():
    """Verify exact hand-calculated Top-1 metrics on data/p1_p2_handoff_qwen3."""
    labels_file = HANDOFF_DIR / "labels.json"
    candidates_file = HANDOFF_DIR / "candidates.jsonl"
    assert labels_file.exists() and candidates_file.exists()

    from src.data.loader import load_records
    candidates = load_records(candidates_file)
    labels = load_records(labels_file)

    report = evaluate_candidate_selection(candidates, labels, top_k=1)
    macro = report["macro"]

    # Queries with TP=1: q1 and q4 (2 out of 7)
    # Queries with TP=0: q2, q3, q5, q6, q7 (5 out of 7)
    # Expected macro: exactly 2/7
    assert macro["precision"] == pytest.approx(2 / 7, abs=1e-6)
    assert macro["recall"] == pytest.approx(2 / 7, abs=1e-6)
    assert macro["f1"] == pytest.approx(2 / 7, abs=1e-6)
    assert macro["f2"] == pytest.approx(2 / 7, abs=1e-6)


def test_evaluate_on_handoff_qwen3_top2():
    """Verify exact hand-calculated Top-2 metrics on data/p1_p2_handoff_qwen3."""
    from src.data.loader import load_records
    candidates = load_records(HANDOFF_DIR / "candidates.jsonl")
    labels = load_records(HANDOFF_DIR / "labels.json")

    report = evaluate_candidate_selection(candidates, labels, top_k=2)
    macro = report["macro"]

    # Queries with TP=1: q1, q2, q3, q4 (4 queries with P=0.5, R=1.0, F1=2/3, F2=5/6)
    # Queries with TP=0: q5, q6, q7 (3 queries with 0.0)
    # Expected macro:
    # Precision: 4 * 0.5 / 7 = 2/7
    # Recall: 4 * 1.0 / 7 = 4/7
    # F1: 4 * (2/3) / 7 = 8/21
    # F2: 4 * (5/6) / 7 = 20/42 = 10/21
    assert macro["precision"] == pytest.approx(2 / 7, abs=1e-6)
    assert macro["recall"] == pytest.approx(4 / 7, abs=1e-6)
    assert macro["f1"] == pytest.approx(8 / 21, abs=1e-6)
    assert macro["f2"] == pytest.approx(10 / 21, abs=1e-6)


def test_table_and_export_functions(tmp_path):
    """Verify ASCII table formatting and CSV/JSON export."""
    ground_truth = [{"id": "q1", "relevant_chunks": ["c1"]}]
    predictions = [{"id": "q1", "relevant_chunks": ["c1", "c2"]}]

    report = evaluate_chunk_predictions(predictions, ground_truth)
    table_str = format_metrics_table(report)
    assert "Query ID" in table_str
    assert "MACRO AVERAGE" in table_str
    assert "q1" in table_str

    csv_path = tmp_path / "metrics.csv"
    export_metrics_csv(report, csv_path)
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "MACRO_AVERAGE" in content
    assert "q1" in content

    json_path = tmp_path / "metrics.json"
    export_metrics_json(report, json_path)
    assert json_path.exists()
    data = read_json(json_path)
    assert "macro" in data and "per_query" in data


def test_cli_execution(tmp_path):
    """Verify execution of scripts/evaluate_f2.py CLI."""
    out_csv = tmp_path / "cli_report.csv"
    out_json = tmp_path / "cli_report.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts" / "evaluate_f2.py"),
        "--handoff-dir",
        str(HANDOFF_DIR),
        "--top-k",
        "2",
        "--output-csv",
        str(out_csv),
        "--output-json",
        str(out_json),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert out_csv.exists()
    assert out_json.exists()

    report = read_json(out_json)
    assert report["macro"]["f2"] == pytest.approx(10 / 21, abs=1e-6)


def test_root_entrypoint():
    """Verify execution of root evaluate_f2.py."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "evaluate_f2.py"),
        "--handoff-dir",
        str(HANDOFF_DIR),
        "--top-k",
        "1",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Root wrapper failed: {result.stderr}"
