"""P3-17: working-notes assembly from experiment artifacts tests."""
import json
from pathlib import Path

import pytest

from src.p3.submission_log import append_submission
from src.p3.working_notes import collect_metrics, export_working_notes, render_working_notes
from src.utils.io import write_json


def test_collect_metrics_without_artifacts_is_empty(tmp_path):
    metrics = collect_metrics(tmp_path / "missing-p2", tmp_path / "missing-p3")
    assert metrics["p2"] == {} and metrics["p3"] == {} and metrics["sources"] == []


def test_collect_metrics_reads_available_artifacts(tmp_path):
    p2 = tmp_path / "p2"
    p2.mkdir()
    write_json(p2 / "selector_cv_report.json", {"status": "complete", "n_splits": 5, "query_count": 7,
                                                "macro": {"f2": 0.62}, "confidence_interval_95": [0.5, 0.7]})
    write_json(p2 / "reranker_benchmark.json", {"run_id": "r1", "before": {"macro": {"f2": 0.4}},
                                                "after": {"macro": {"f2": 0.6}}})
    p3 = tmp_path / "p3"
    p3.mkdir()
    write_json(p3 / "doc_pipeline_report.json", {"best_doc_pipeline": {
        "ablation_method": "max", "direct_doc_weight": 0.0,
        "holdout": {"mean_f2_doc": 0.71, "std_f2_doc": 0.03}, "full_data_f2_doc": 0.75}})
    (p3 / "doc_pipeline_ablation.csv").write_text(
        "aggregation,direct_doc_weight,oof_mean_f2,oof_std_f2,doc_threshold,doc_fallback,doc_max,full_data_f2\n"
        "max,0.0,0.71,0.03,,1,5,0.75\n", encoding="utf-8")
    log = tmp_path / "submission_log.csv"
    append_submission(log, {"run_id": "r1", "split": "public", "local_score": 0.7})

    metrics = collect_metrics(p2, p3, log)
    assert metrics["p2"]["selector_cv"]["macro_f2"] == 0.62
    assert metrics["p2"]["reranker_benchmark"]["after_macro_f2"] == 0.6
    assert metrics["p3"]["doc_pipeline"]["holdout_mean_f2_doc"] == 0.71
    assert metrics["p3"]["doc_pipeline_ablation"][0]["aggregation"] == "max"
    assert metrics["submissions"]["public_count"] == 1
    assert len(metrics["sources"]) == 5


def test_render_includes_values_and_pending_markers():
    text = render_working_notes(collect_metrics())
    assert "n/a (pending official run)" in text
    assert "P1 — Mai Ngọc Duy" in text and "P2 — Mạc Duy" in text and "P3 — Quế" in text
    assert "Figure 5" in text and "Table 6" in text


def test_export_working_notes_writes_outline_and_metrics(tmp_path):
    result = export_working_notes(tmp_path / "notes")
    assert Path(result["outline"]).exists() and Path(result["metrics"]).exists()
    payload = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))
    assert "p2" in payload
    with pytest.raises(FileExistsError):
        export_working_notes(tmp_path / "notes")
