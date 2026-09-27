"""P1-11: fixed candidate fields, portable reranker input and unchanged identities."""
import copy
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest
import yaml

from src.pipeline.benchmark import run_retrieval_benchmark
from src.pipeline.handoff import BUNDLE_FILES, export_reranking_input, validate_reranking_input
from src.pipeline.rerank import run_reranking
from src.reranking.bge import CrossEncoderReranker
from src.retrieval.schema import standardize_candidates, validate_candidate_records
from src.utils.config import load_config
from src.utils.io import read_json, write_json


@pytest.fixture
def handoff_config(experiment, tmp_path):
    labels = tmp_path / "source_labels.json"
    write_json(labels, [{"id": "q1", "relevant_chunks": ["d1_c000"], "relevant_docs": ["d1"]},
                        {"id": "q2", "relevant_chunks": ["d2_c000"], "relevant_docs": ["d2"]}])
    experiment["benchmark"] = {"owner": "Mai Ngọc Duy", "labels_path": str(labels), "cutoffs": [1, 3],
                               "build_indexes_if_missing": True, "rrf_constants": [20, 60],
                               "label_quality": "test_fixture", "experiment_log_path": str(tmp_path / "log.csv")}
    source = run_retrieval_benchmark(experiment)
    return {"handoff": {"source_run_dir": str(source), "method": "bm25",
                        "output_dir": str(tmp_path / "handoff"), "archive_path": str(tmp_path / "handoff.zip")},
            "reranker": {"enabled": True, "model": "BAAI/bge-reranker-v2-m3", "device": "cpu", "batch_size": 1}}


def records(root):
    return [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]


def test_export_is_self_contained_and_preserves_ids_and_text(handoff_config, tmp_path):
    root = export_reranking_input(handoff_config)
    source = root.parent / "outputs/test_run/bm25"
    before, after = records(source), records(root)
    for original, exported in zip(before, after):
        for old, new in zip(original["candidates"], exported["candidates"]):
            assert (new["chunk_id"], new["doc_id"], new["text"], new["score"]) == (old["chunk_id"], old["doc_id"], old["text"], old["score"])
            assert new["query_id"] == exported["id"]
            assert new["dense_rank"] is None and new["dense_score"] is None and new["fused_score"] is None
            assert new["bm25_rank"] == old["rank"]
    assert any(c["chunk_id"] == "internal1" for c in after[0]["candidates"])
    manifest = validate_reranking_input(root)
    assert manifest["query_count"] == 2 and manifest["candidate_count"] == 6
    assert manifest["label_quality"] == "test_fixture"
    assert str(tmp_path) not in (root / "config.yaml").read_text()
    with zipfile.ZipFile(handoff_config["handoff"]["archive_path"]) as archive:
        assert set(archive.namelist()) == set(BUNDLE_FILES)
        assert all("/" not in name for name in archive.namelist())


def test_received_bundle_runs_reranking_without_original_dataset(handoff_config, tmp_path):
    root = export_reranking_input(handoff_config)
    receiver = tmp_path / "another_machine/received"
    shutil.copytree(root, receiver)
    (tmp_path / "processed").rename(tmp_path / "source_no_longer_available")
    validate_reranking_input(receiver)
    assert load_config(receiver / "config.yaml")["evaluation"]["labels_path"] == str(receiver / "labels.json")
    class Model:
        def predict(self, pairs, batch_size):
            return [float(index) for index, _ in enumerate(pairs)]
    run_reranking(receiver, reranker=CrossEncoderReranker(model=Model(), batch_size=1))
    scored = [json.loads(line) for line in (receiver / "reranked.jsonl").read_text().splitlines()]
    for before, after in zip(records(receiver), scored):
        assert before["id"] == after["id"]
        assert {(c["query_id"], c["chunk_id"], c["doc_id"], c["text"]) for c in before["candidates"]} == {
            (c["query_id"], c["chunk_id"], c["doc_id"], c["text"]) for c in after["candidates"]}
        assert all("rerank_score" in c for c in after["candidates"])


def test_export_never_overwrites_existing_bundle(handoff_config):
    root = export_reranking_input(handoff_config)
    before = (root / "candidates.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        export_reranking_input(handoff_config)
    assert (root / "candidates.jsonl").read_bytes() == before


def test_receiver_detects_tampered_artifacts(handoff_config):
    root = export_reranking_input(handoff_config)
    (root / "queries.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        validate_reranking_input(root)
    with pytest.raises(ValueError, match="checksum"):
        run_reranking(root)
    assert not (root / "reranked.jsonl").exists()


def test_export_rejects_stale_corpus_before_creating_output(handoff_config, tmp_path):
    path = tmp_path / "processed/chunks.json"
    chunks = read_json(path)
    chunks[0]["text"] = "changed"
    path.write_text(json.dumps(chunks))
    with pytest.raises(ValueError, match="Source data changed"):
        export_reranking_input(handoff_config)
    assert not (tmp_path / "handoff").exists()


@pytest.mark.parametrize("change", [{"query_id": "wrong"}, {"chunk_id": "unknown"}, {"text": "wrong text"}])
def test_export_rejects_corrupt_candidate_identity(handoff_config, tmp_path, change):
    source = tmp_path / "outputs/test_run/bm25"
    rows = records(source)
    rows[0]["candidates"][0].update(change)
    (source / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        export_reranking_input(handoff_config)
    assert not (tmp_path / "handoff").exists()


def test_contract_retains_fusion_provenance_and_accepts_empty_candidates():
    candidate = {"chunk_id": "a", "doc_id": "d", "text": "text", "score": .03, "rank": 1, "source": "rrf",
                 "source_ranks": {"bm25": 3, "dense": 1}, "source_scores": {"bm25": 400, "dense": .9}}
    original = copy.deepcopy(candidate)
    rows = standardize_candidates("q", [candidate])
    assert candidate == original
    assert rows[0]["bm25_rank"] == 3 and rows[0]["bm25_score"] == 400
    assert rows[0]["dense_rank"] == 1 and rows[0]["dense_score"] == .9
    assert rows[0]["fused_score"] == .03
    registry = {"expected_query_ids": ["q"], "internal_to_official": {"a": "a"}, "chunk_to_doc": {"a": "d"}}
    validate_candidate_records([{"id": "q", "candidates": rows}], [{"id": "q", "text": "query"}], registry)
    validate_candidate_records([{"id": "q", "candidates": []}], [{"id": "q", "text": "query"}], registry)
    with pytest.raises(ValueError, match="coverage"):
        validate_candidate_records([], [{"id": "q", "text": "query"}], registry)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.9"])
def test_contract_rejects_non_numeric_or_nonfinite_scores(value):
    with pytest.raises(ValueError, match="finite numeric"):
        standardize_candidates("q", [{"chunk_id": "a", "doc_id": "d", "text": "text", "source": "dense", "rank": 1, "score": value}])


@pytest.mark.parametrize("corruption", ["missing_field", "duplicate_chunk", "bad_parent", "bad_rank", "conflicting_provenance"])
def test_validator_rejects_malformed_contract(corruption):
    rows = [{"id": "q", "candidates": standardize_candidates("q", [
        {"chunk_id": "a", "doc_id": "d", "text": "text", "source": "bm25", "score": 3.0, "rank": 1}])}]
    candidate = rows[0]["candidates"][0]
    if corruption == "missing_field":
        del candidate["dense_score"]
    elif corruption == "duplicate_chunk":
        rows[0]["candidates"].append(copy.deepcopy(candidate))
    elif corruption == "bad_parent":
        candidate["doc_id"] = "unknown"
    elif corruption == "bad_rank":
        candidate["rank"] = True
    else:
        candidate["bm25_score"] = 99.0
    registry = {"expected_query_ids": ["q"], "internal_to_official": {"a": "a"}, "chunk_to_doc": {"a": "d"}}
    with pytest.raises(ValueError):
        validate_candidate_records(rows, [{"id": "q", "text": "query"}], registry)


def test_validator_rejects_incorrect_manifest_counts(handoff_config):
    root = export_reranking_input(handoff_config)
    manifest = read_json(root / "manifest.json")
    manifest["candidate_count"] += 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="counts"):
        validate_reranking_input(root)


def test_handoff_cli_works_outside_repository(handoff_config, tmp_path):
    project = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "handoff.yaml"
    config_path.write_text(yaml.safe_dump(handoff_config))
    receiver_cwd = tmp_path / "receiver_cwd"
    receiver_cwd.mkdir()
    commands = [("export_reranking_input.py", "--config", str(config_path)),
                ("validate_reranking_input.py", "--run-dir", handoff_config["handoff"]["output_dir"])]
    for script, option, value in commands:
        result = subprocess.run([sys.executable, str(project / "scripts" / script), option, value],
                                cwd=receiver_cwd, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    manifest = read_json(Path(handoff_config["handoff"]["output_dir"]) / "manifest.json")
    assert manifest["export_git_commit"]
