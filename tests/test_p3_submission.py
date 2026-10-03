"""P3-06/P3-07: submission generation, validation and ZIP packing tests."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.p3.submission import (
    build_submission,
    normalize_predictions,
    validator_from_registry,
    write_submission,
)
from src.utils.io import write_json

REPO = Path(__file__).resolve().parents[1]
REGISTRY = {"expected_query_ids": ["q1", "q2"], "doc_ids": ["d1", "d2"],
            "chunk_to_doc": {"c1": "d1", "c2": "d2"}, "internal_to_official": {"c1": "c1", "c2": "c2"}}
VALID_PREDICTIONS = [{"id": "q1", "relevant_docs": ["d2", "d1", "d2"], "relevant_chunks": ["c2", "c1"]},
                     {"id": "q2", "relevant_docs": [], "relevant_chunks": []}]


def test_normalize_sorts_and_dedups():
    out = normalize_predictions([{"id": "q1", "relevant_docs": ["d2", "d1", "d2"],
                                  "relevant_chunks": ["c2", "c1"]}])
    assert out == [{"id": "q1", "relevant_docs": ["d1", "d2"], "relevant_chunks": ["c1", "c2"]}]


def test_normalize_rejects_bad_rows():
    with pytest.raises(ValueError):
        normalize_predictions([])
    with pytest.raises(ValueError):
        normalize_predictions([{"id": "q1", "relevant_docs": [1], "relevant_chunks": []}])
    with pytest.raises(ValueError):
        normalize_predictions([{"id": "q1", "relevant_docs": [], "relevant_chunks": []},
                               {"id": "q1", "relevant_docs": [], "relevant_chunks": []}])


def test_build_submission_validates_against_registry():
    assert build_submission(VALID_PREDICTIONS, validator_from_registry(REGISTRY))[0]["relevant_docs"] == ["d1", "d2"]
    with pytest.raises(ValueError):
        build_submission(VALID_PREDICTIONS[:1], validator_from_registry(REGISTRY))


def test_write_submission_is_exclusive(tmp_path):
    path = tmp_path / "submission.json"
    valid = normalize_predictions(VALID_PREDICTIONS)
    write_submission(path, valid, validator_from_registry(REGISTRY))
    assert json.loads(path.read_text(encoding="utf-8")) == valid
    with pytest.raises(FileExistsError):
        write_submission(path, valid, validator_from_registry(REGISTRY))


def test_validator_from_registry_rejects_non_mapping():
    with pytest.raises(ValueError):
        validator_from_registry("nope")


def _run(*args):
    return subprocess.run([sys.executable, *[str(arg) for arg in args]],
                          cwd=REPO, capture_output=True, text=True)


@pytest.fixture
def handoff(tmp_path):
    run = tmp_path / "handoff"
    run.mkdir()
    records = [{"id": "q1", "candidates": [{"chunk_id": "c1", "doc_id": "d1", "rerank_score": 0.9}]},
               {"id": "q2", "candidates": [{"chunk_id": "c2", "doc_id": "d2", "rerank_score": 0.8}]}]
    (run / "reranked.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    write_json(run / "registry.json", REGISTRY)
    (run / "best_chunk_selector.yaml").write_text(
        "chunk_selector:\n  chunk_threshold: null\n  chunk_fallback: 1\n  chunk_max: 5\n", encoding="utf-8")
    doc_selector = tmp_path / "best_doc_selector.yaml"
    doc_selector.write_text(
        "doc_selector:\n  doc_aggregation: max\n  doc_top_k: 3\n  doc_weight: 0.5\n"
        "  doc_threshold: null\n  doc_fallback: 1\n  doc_max: 5\n", encoding="utf-8")
    return run, doc_selector


def test_submission_cli_flow_generate_validate_zip(handoff, tmp_path):
    run, doc_selector = handoff
    submission = tmp_path / "submission.json"
    generate = _run(REPO / "scripts" / "generate_submission.py", "--run-dir", run,
                    "--doc-selector", doc_selector, "--registry", run / "registry.json",
                    "--output", submission)
    assert generate.returncode == 0, generate.stderr
    data = json.loads(submission.read_text(encoding="utf-8"))
    assert [row["id"] for row in data] == ["q1", "q2"]
    assert data[0]["relevant_chunks"] == ["c1"]

    check = _run(REPO / "scripts" / "validate_submission.py", "--input", submission,
                 "--registry", run / "registry.json")
    assert check.returncode == 0, check.stdout + check.stderr

    archive = tmp_path / "submission.zip"
    zipped = _run(REPO / "scripts" / "zip_submission.py", "--input", submission,
                  "--output", archive, "--registry", run / "registry.json")
    assert zipped.returncode == 0, zipped.stderr
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["submission.json"]

    zip_check = _run(REPO / "scripts" / "validate_submission.py", "--input", archive,
                     "--registry", run / "registry.json")
    assert zip_check.returncode == 0, zip_check.stdout + zip_check.stderr


def test_validate_cli_rejects_unknown_ids(handoff, tmp_path):
    run, _ = handoff
    bad = tmp_path / "bad.json"
    write_json(bad, [{"id": "q1", "relevant_docs": ["dX"], "relevant_chunks": []},
                     {"id": "q2", "relevant_docs": [], "relevant_chunks": []}])
    result = _run(REPO / "scripts" / "validate_submission.py", "--input", bad,
                  "--registry", run / "registry.json")
    assert result.returncode == 1
