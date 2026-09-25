import importlib
import pkgutil
from pathlib import Path
import subprocess
import sys
import yaml
import pytest

from src.pipeline.retrieve import build_bm25_index, run_retrieval
from src.pipeline.full_pipeline import run_full_pipeline
from src.pipeline.reporting import run_evaluation, make_submission
from src.submission.zipper import validate_zip
from src.utils.io import read_json, write_json


def test_real_bm25_end_to_end(experiment, tmp_path):
    build_bm25_index(experiment)
    run_dir = run_full_pipeline(experiment)
    prediction = read_json(run_dir / "predictions.json")
    assert prediction == [
        {"id": "q1", "relevant_chunks": ["d1_c000"], "relevant_docs": ["d1"]},
        {"id": "q2", "relevant_chunks": ["d2_c000"], "relevant_docs": ["d2"]},
    ]
    labels = tmp_path / "labels.json"
    write_json(labels, prediction)
    metrics = run_evaluation(run_dir, labels)
    assert metrics["macro"]["relevant_chunks"]["f2"] == 1
    candidate_metrics = run_evaluation(run_dir, labels, "candidates", 3)
    assert candidate_metrics["macro"]["relevant_chunks"]["recall@3"] == 1
    zipped = make_submission(run_dir)
    assert validate_zip(zipped) == prediction
    with pytest.raises(FileExistsError):
        run_full_pipeline(experiment)
    with pytest.raises(FileExistsError):
        build_bm25_index(experiment)
    with pytest.raises(FileExistsError):
        run_evaluation(run_dir, labels)


def test_stale_index_rejected(experiment):
    build_bm25_index(experiment)
    experiment["retrieval"]["bm25"]["lowercase"] = True
    with pytest.raises(ValueError, match="mismatch"):
        run_retrieval(experiment)


def test_all_module_imports():
    import src
    for module in pkgutil.walk_packages(src.__path__, "src."):
        importlib.import_module(module.name)
    from src.retrieval.sparse_search import SparseRetriever
    from src.retrieval.bm25 import BM25Retriever
    from src.utils.validator import SubmissionValidator
    from src.submission.validator import SubmissionValidator as CanonicalValidator
    assert SparseRetriever is BM25Retriever
    assert SubmissionValidator is CanonicalValidator


def test_cli_from_unrelated_cwd(experiment, tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment))
    for script in ["build_bm25_index", "run_full_pipeline"]:
        subprocess.run([sys.executable, str(root / "scripts" / f"{script}.py"),
                        "--config", str(config)], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([sys.executable, str(root / "scripts" / "make_submission.py"),
                    "--run-dir", str(tmp_path / "outputs" / "test_run")],
                   cwd=tmp_path, check=True, capture_output=True)


def test_all_entrypoints_help(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for script in (root / "scripts").glob("*.py"):
        subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, check=True, capture_output=True)


def test_staged_cli_workflow(dataset, experiment, tmp_path):
    root = Path(__file__).resolve().parents[1]
    prepare_config = dataset[0]
    prepare_config["data"]["output_dir"] = str(tmp_path / "staged_data")
    prepare_config["data"]["mappings_dir"] = str(tmp_path / "staged_mappings")
    prepare_path = tmp_path / "prepare.yaml"
    prepare_path.write_text(yaml.safe_dump(prepare_config))
    experiment["data"].update({
        **{f"{kind}_path": str(tmp_path / "staged_data" / f"{kind}.json")
           for kind in ("documents", "chunks", "queries")},
        "mappings_dir": str(tmp_path / "staged_mappings"),
    })
    experiment_path = tmp_path / "staged.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment))
    run_dir = tmp_path / "outputs" / "test_run"
    labels_path = tmp_path / "labels.json"
    write_json(labels_path, [
        {"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["d1_c000"]},
        {"id": "q2", "relevant_docs": ["d2"], "relevant_chunks": ["d2_c000"]},
    ])
    commands = [
        ("prepare_data", ["--config", str(prepare_path)]),
        ("build_bm25_index", ["--config", str(experiment_path)]),
        ("run_retrieval", ["--config", str(experiment_path)]),
        ("run_reranking", ["--run-dir", str(run_dir)]),
        ("run_prediction", ["--run-dir", str(run_dir)]),
        ("run_evaluation", ["--run-dir", str(run_dir), "--ground-truth", str(labels_path)]),
        ("make_submission", ["--run", "test_run", "--output-dir", str(tmp_path / "outputs")]),
    ]
    for script, arguments in commands:
        subprocess.run([sys.executable, str(root / "scripts" / f"{script}.py"), *arguments],
                       cwd=tmp_path, check=True, capture_output=True)
    assert read_json(run_dir / "metrics_predictions.json")["macro"]["relevant_chunks"]["f2"] == 1
    assert validate_zip(tmp_path / "submissions" / "test_run.zip") == read_json(labels_path)
