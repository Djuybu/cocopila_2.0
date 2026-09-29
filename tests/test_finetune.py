"""Unit and integration tests for P2-08: Reranker fine-tuning & evaluation."""
from pathlib import Path
import subprocess
import sys
import pytest

from src.training.finetune import (
    CrossEncoderTrainer,
    create_training_config,
    evaluate_reranker_on_validation,
    prepare_cross_encoder_examples,
)
from src.utils.io import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_create_training_config():
    cfg = create_training_config(
        model_name="BAAI/bge-reranker-v2-m3",
        output_dir="checkpoints/v1",
        epochs=4,
        batch_size=32,
        seed=123,
    )
    assert "training" in cfg
    t = cfg["training"]
    assert t["model_name"] == "BAAI/bge-reranker-v2-m3"
    assert t["epochs"] == 4
    assert t["batch_size"] == 32
    assert t["seed"] == 123


def test_prepare_cross_encoder_examples():
    pairs = [
        {"query": "q1", "text": "t1", "label": 1.0},
        {"query": "q2", "text": "t2", "label": 0.0},
    ]
    examples = prepare_cross_encoder_examples(pairs)
    assert len(examples) == 2
    # Check attributes regardless of whether InputExample or dict
    first = examples[0]
    if hasattr(first, "texts"):
        assert first.texts == ["q1", "t1"]
        assert first.label == 1.0
    else:
        assert first["texts"] == ["q1", "t1"]
        assert first["label"] == 1.0


def test_evaluate_reranker_on_validation_mock():
    val_records = [
        {
            "id": "q1",
            "query": "aspirin dosage",
            "candidates": [
                {"chunk_id": "c1", "text": "Aspirin 500mg."},
                {"chunk_id": "c2", "text": "Insulin pen."},
            ],
        },
    ]
    labels = [
        {"id": "q1", "relevant_chunks": ["c1"]},
    ]

    # Mock predict function: returns high score for c1, low for c2
    def mock_predict(pairs):
        return [0.95, 0.10]

    metrics = evaluate_reranker_on_validation(
        model_or_predict_fn=mock_predict,
        val_records=val_records,
        labels=labels,
        threshold=0.5,
    )

    assert "macro_f2" in metrics
    assert "macro_recall" in metrics
    assert "macro_precision" in metrics
    # c1 is selected (score 0.95 >= 0.5), c2 is pruned -> P=1.0, R=1.0, F2=1.0
    assert metrics["macro_f2"] == 1.0
    assert metrics["macro_recall"] == 1.0


def test_cli_finetune_dry_run(tmp_path):
    train_pairs = [
        {"query": "q1", "text": "t1", "label": 1.0, "query_id": "q1", "chunk_id": "c1"},
        {"query": "q1", "text": "t2", "label": 0.0, "query_id": "q1", "chunk_id": "c2"},
    ]
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, train_pairs)

    checkpoint_dir = tmp_path / "checkpoints" / "v1"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "finetune_reranker.py"),
        "--train-data",
        str(train_file),
        "--output-dir",
        str(checkpoint_dir),
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--seed",
        "42",
        "--dry-run",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed:\n{res.stderr}"

    assert checkpoint_dir.exists()
    manifest_path = checkpoint_dir / "training_manifest.json"
    assert manifest_path.exists()

    manifest = read_json(manifest_path)
    assert manifest["seed"] == 42
    assert manifest["epochs"] == 2
    assert manifest["num_training_pairs"] == 2
