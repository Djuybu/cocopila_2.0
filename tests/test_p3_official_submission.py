"""P3-15: official submission JSON/ZIP generation and provenance tests."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from src.p3.official_submission import generate_official_submission, git_provenance, resolve_doc_config
from src.p3.submission import validator_from_registry
from src.utils.io import write_json

REPO = Path(__file__).resolve().parents[1]
REGISTRY = {"expected_query_ids": ["q1", "q2"], "doc_ids": ["d1", "d2"],
            "chunk_to_doc": {"c1": "d1", "c2": "d2"}, "internal_to_official": {"c1": "c1", "c2": "c2"}}
CHUNK_CONFIG = {"chunk_threshold": None, "chunk_fallback": 1, "chunk_max": 5}
PIPELINE = {"doc_aggregation": "max", "doc_top_k": 3, "doc_weight": 0.5, "doc_threshold": None,
            "doc_fallback": 1, "doc_max": 5, "direct_doc_weight": 0.0}


def scored_records():
    return [{"id": "q1", "candidates": [{"chunk_id": "c1", "doc_id": "d1", "rerank_score": 0.9}]},
            {"id": "q2", "candidates": [{"chunk_id": "c2", "doc_id": "d2", "rerank_score": 0.8}]}]


def test_resolve_doc_config_rejects_incomplete_pipeline():
    assert resolve_doc_config(PIPELINE)["doc_max"] == 5
    with pytest.raises(ValueError):
        resolve_doc_config({"doc_max": 5})


def test_generate_official_submission_writes_json_zip_and_manifest(tmp_path):
    validator = validator_from_registry(REGISTRY)
    result = generate_official_submission(scored_records(), CHUNK_CONFIG, PIPELINE, validator,
                                          run_id="p3test", output_dir=tmp_path / "out",
                                          submission_dir=tmp_path / "subs", project_root=REPO)
    assert Path(result["json"]).exists() and Path(result["zip"]).exists()
    with zipfile.ZipFile(result["zip"]) as archive:
        assert archive.namelist() == ["submission_p3test.json"]
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["run_id"] == "p3test" and manifest["query_count"] == 2
    assert manifest["zip_sha256"] and manifest["json_sha256"]
    assert manifest["git_commit"] and "worktree_dirty" in manifest
    with pytest.raises(FileExistsError):
        generate_official_submission(scored_records(), CHUNK_CONFIG, PIPELINE, validator,
                                     run_id="p3test", output_dir=tmp_path / "out",
                                     submission_dir=tmp_path / "subs", project_root=REPO)


def test_git_provenance_reports_commit():
    provenance = git_provenance(REPO)
    assert len(provenance["git_commit"]) == 40
    assert isinstance(provenance["worktree_dirty"], bool)


def _run(*args):
    return subprocess.run([sys.executable, *[str(arg) for arg in args]],
                          cwd=REPO, capture_output=True, text=True)


@pytest.fixture
def handoff(tmp_path):
    run = tmp_path / "handoff"
    run.mkdir()
    (run / "reranked.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scored_records()), encoding="utf-8")
    write_json(run / "registry.json", REGISTRY)
    write_json(run / "labels.json", [{"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]},
                                     {"id": "q2", "relevant_docs": ["d2"], "relevant_chunks": ["c2"]}])
    (run / "best_chunk_selector.yaml").write_text(
        "chunk_selector:\n  chunk_threshold: null\n  chunk_fallback: 1\n  chunk_max: 5\n", encoding="utf-8")
    return run


def test_doc_pipeline_and_official_submission_cli_flow(handoff, tmp_path):
    tune = _run(REPO / "scripts" / "tune_doc_pipeline.py", "--run-dir", handoff,
                "--folds", "2", "--output-dir", tmp_path / "pipeline")
    assert tune.returncode == 0, tune.stderr
    pipeline_yaml = tmp_path / "pipeline" / "best_doc_pipeline.yaml"
    assert pipeline_yaml.exists()

    generate = _run(REPO / "scripts" / "generate_official_submission.py", "--run-dir", handoff,
                    "--doc-pipeline", pipeline_yaml, "--run-id", "cli001",
                    "--output-dir", tmp_path / "submission",
                    "--submission-dir", tmp_path / "zips")
    assert generate.returncode == 0, generate.stderr + generate.stdout
    archive = tmp_path / "zips" / "submission_cli001.zip"
    assert archive.exists()
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["submission_cli001.json"]
