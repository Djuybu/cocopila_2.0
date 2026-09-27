"""Qwen CLI/factory integration on P1 bundles, without any model downloads."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from src.data.loader import load_records
from src.evaluation.reranking_metrics import evaluate_reranked_candidates
from src.pipeline.cli import main
from src.pipeline.handoff import BUNDLE_FILES, export_reranking_input, validate_reranking_input
from src.pipeline.rerank import run_reranking
from src.pipeline.reranker_benchmark import benchmark_reranking
from src.reranking.bge import CrossEncoderReranker
from src.reranking.qwen_reranker import QwenReranker
from src.utils.config import load_config
from src.utils.io import read_json
from tests.test_handoff import handoff_config


class Model:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs, batch_size):
        self.pairs.extend(pairs)
        return [1.0 if query in text else -1.0 for query, text in pairs]


@pytest.fixture
def received(handoff_config):
    return export_reranking_input(handoff_config)


@pytest.fixture
def qwen_config():
    return {"reranker": {"enabled": True, "type": "qwen", "model": "Qwen/Qwen3-Reranker-0.6B",
                         "device": "cpu", "batch_size": 2, "max_length": 256, "dtype": "float32",
                         "instruction": "Evaluate medical relevance."}}


def snapshot(root):
    return {name: (root / name).read_bytes() for name in BUNDLE_FILES}


def test_pipeline_factory_uses_qwen_on_old_bge_bundle(received, qwen_config, tmp_path, monkeypatch):
    original = snapshot(received)
    loaded = []
    def load_model(self):
        loaded.append(self)
        self.model = Model()
    monkeypatch.setattr(QwenReranker, "load_model", load_model)
    (tmp_path / "processed").rename(tmp_path / "unavailable_original_corpus")
    output = tmp_path / "p2_qwen"
    run_reranking(received, reranker_config=qwen_config, output_dir=output)
    assert isinstance(loaded[0], QwenReranker)
    assert loaded[0].max_length == 256 and loaded[0].dtype == "float32"
    assert loaded[0].instruction == qwen_config["reranker"]["instruction"]
    assert snapshot(received) == original
    validate_reranking_input(received)
    config = load_config(output / "config.yaml")
    assert config["reranker"] == qwen_config["reranker"]
    assert config["evaluation"]["labels_path"] == str(output / "labels.json")
    assert (output / "input_manifest.json").read_bytes() == original["manifest.json"]
    scored = load_records(output / "reranked.jsonl")
    assert sum(len(row["candidates"]) for row in scored) == 6
    assert read_json(output / "reranking_metadata.json")["adapter"] == "QwenReranker"
    assert read_json(output / "reranking_metadata.json")["backend_injected"] is False


def test_bge_default_and_instruction_still_work(received, monkeypatch):
    monkeypatch.setattr(CrossEncoderReranker, "load_model", lambda self: setattr(self, "model", Model()))
    run_reranking(received)
    assert read_json(received / "reranking_metadata.json")["adapter"] == "CrossEncoderReranker"
    validate_reranking_input(received)


def test_qwen_bundle_uses_factory_without_override(handoff_config, qwen_config, monkeypatch):
    handoff_config["reranker"] = qwen_config["reranker"]
    root = export_reranking_input(handoff_config)
    monkeypatch.setattr(QwenReranker, "load_model", lambda self: setattr(self, "model", Model()))
    run_reranking(root)
    assert read_json(root / "reranking_metadata.json")["adapter"] == "QwenReranker"


def test_cli_config_override_requires_separate_output(received, qwen_config, tmp_path, monkeypatch):
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(yaml.safe_dump(qwen_config))
    monkeypatch.setattr(sys, "argv", ["run_reranking", "--run-dir", str(received), "--config", str(config_path)])
    with pytest.raises(SystemExit) as error:
        main("run_reranking")
    assert error.value.code == 2
    assert not (received / "reranked.jsonl").exists()


@pytest.mark.parametrize("command", ["run_reranking", "benchmark_reranking"])
def test_cli_scores_received_bundle(command, received, qwen_config, tmp_path, monkeypatch):
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(yaml.safe_dump(qwen_config))
    output = tmp_path / command
    monkeypatch.setattr(QwenReranker, "load_model", lambda self: setattr(self, "model", Model()))
    monkeypatch.setattr(sys, "argv", [command, "--run-dir", str(received), "--config", str(config_path),
                                     "--output-dir", str(output)])
    main(command)
    assert sum(len(row["candidates"]) for row in load_records(output / "reranked.jsonl")) == 6
    if command == "benchmark_reranking":
        assert read_json(output / "reranking_benchmark.json")["reranker"]["type"] == "qwen"


def test_benchmark_reports_supplied_labels_without_regeneration(received, qwen_config, tmp_path):
    original = snapshot(received)
    output = tmp_path / "benchmark"
    report = benchmark_reranking(received, output, reranker_config=qwen_config, ks=[1, 2, 3],
                                 reranker=QwenReranker(model=Model()))
    assert report["query_count"] == 2 and report["candidate_count"] == 6
    assert report["backend_injected"] is True
    assert report["label_quality"] == "test_fixture"
    assert report["after"]["macro"]["chunks"]["recall@1"] == 1.0
    assert report["after"]["macro"]["chunks"]["mrr@1"] == 1.0
    assert report["before"]["stage"] == "before_reranking"
    assert report["performance"]["model_load_seconds"] >= 0
    assert len(report["performance"]["per_query"]) == 2
    assert report["performance"]["peak_cuda_allocated_bytes"] is None
    assert (output / "labels.json").read_bytes() == original["labels.json"]
    assert snapshot(received) == original
    with pytest.raises(FileExistsError):
        benchmark_reranking(received, output, reranker_config=qwen_config)


@pytest.mark.parametrize("case", ["disabled", "invalid_type", "tampered"])
def test_bad_inputs_fail_before_output_creation(case, received, qwen_config, tmp_path):
    output = tmp_path / "bad"
    if case == "disabled":
        qwen_config["reranker"]["enabled"] = False
    elif case == "invalid_type":
        qwen_config["reranker"]["type"] = "unknown"
    else:
        (received / "queries.json").write_text("[]")
    with pytest.raises(ValueError):
        benchmark_reranking(received, output, reranker_config=qwen_config)
    assert not output.exists()


def test_override_and_existing_paths_never_overwrite(received, qwen_config, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        run_reranking(received, reranker_config=qwen_config)
    with pytest.raises(ValueError, match="separate"):
        run_reranking(received, output_dir=received / "nested", reranker_config=qwen_config)
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_reranking(received, output_dir=output, reranker_config=qwen_config)


@pytest.mark.parametrize("corruption", ["drop", "duplicate", "text", "score", "in_place"])
def test_pipeline_rejects_backend_that_drops_or_rewrites_candidates(corruption, received, tmp_path):
    class BadBackend:
        def rerank(self, query, candidates):
            rows = [{**row, "rerank_score": 1.0} for row in candidates]
            if corruption == "drop":
                return rows[:-1]
            if corruption == "duplicate":
                return [rows[0]] * len(rows)
            if corruption == "in_place":
                candidates[0]["source_scores"]["bm25"] = 12345
                return [{**row, "rerank_score": 1.0} for row in candidates]
            rows[0]["text" if corruption == "text" else "score"] = "changed"
            return rows
    with pytest.raises(ValueError):
        run_reranking(received, output_dir=tmp_path / "bad_backend", reranker=BadBackend())
    validate_reranking_input(received)


def test_binary_metrics_depend_on_rank_and_map_internal_ids():
    labels = [{"id": "q", "relevant_chunks": ["official"], "relevant_docs": ["d1"]}]
    rows = [{"id": "q", "candidates": [{"chunk_id": "n", "doc_id": "d2"}, {"chunk_id": "internal", "doc_id": "d1"}]}]
    result = evaluate_reranked_candidates(rows, labels, [1, 2], {"internal": "official"})
    chunks = result["macro"]["chunks"]
    assert chunks["recall@1"] == 0 and chunks["recall@2"] == 1
    assert chunks["mrr@2"] == .5 and chunks["precision@2"] == .5
    assert chunks["ndcg@2"] == pytest.approx(1 / 1.584962500721156)
    assert result["macro"]["documents"]["recall@1"] == 0


def test_qwen_native_prompt_and_load_options(monkeypatch):
    observed = {}
    def cross_encoder(name, **kwargs):
        observed.update(name=name, **kwargs)
        return Model()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=cross_encoder))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="fp16"))
    monkeypatch.setattr("importlib.metadata.version", lambda _: "6.1.0")
    reranker = QwenReranker(device="cpu", instruction="Medical relevance", max_length=512,
                           dtype="float16", revision="fixed-model-revision")
    reranker.rerank("aspirin", [{"chunk_id": "c", "doc_id": "d", "text": "aspirin treatment"}])
    assert observed["prompts"] == {"medical": "Medical relevance"}
    assert observed["default_prompt_name"] == "medical"
    assert observed["max_length"] == 512 and observed["model_kwargs"] == {"dtype": "fp16"}
    assert observed["revision"] == "fixed-model-revision"
    assert reranker.model.pairs == [("aspirin", "aspirin treatment")]


def test_qwen_requires_supported_sentence_transformers(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "5.4.0")
    with pytest.raises(RuntimeError, match="sentence-transformers>=6.1"):
        QwenReranker().load_model()


def test_primary_notebook_consumes_handoff_without_legacy_corpus():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "notebooks/P2_01_reranker_evaluation.ipynb").read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "benchmark_reranking" in code and "validate_reranking_input" in code
    assert "data/test" not in code and "sys.path" not in code and "QdrantClient" not in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook cell>", "exec")


def test_qwen_default_configs():
    project = Path(__file__).resolve().parents[1]
    for name in ("configs/handoff/p1_to_p2.yaml", "configs/experiments/exp003_full.yaml"):
        cfg = load_config(project / name)
        assert cfg["reranker"]["type"] == "qwen"
        assert cfg["reranker"]["model"] == "Qwen/Qwen3-Reranker-0.6B"
