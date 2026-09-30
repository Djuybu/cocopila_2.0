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
    assert manifest["dry_run"] is True
    assert manifest["validation_metrics"] is None
    assert (checkpoint_dir / "training_config.yaml").exists()


def test_validation_uses_query_file_and_new_scores():
    records = [{"id": "q1", "candidates": [
        {"chunk_id": "i1", "doc_id": "d1", "text": "aspirin", "rerank_score": 0.01},
        {"chunk_id": "i2", "doc_id": "d2", "text": "insulin", "rerank_score": 0.99},
    ]}]
    seen = []
    def predict(pairs):
        seen.extend(pairs)
        return [0.9, 0.1]
    metrics = evaluate_reranker_on_validation(
        predict, records, [{"id": "q1", "relevant_chunks": ["c1"]}],
        queries=[{"id": "q1", "text": "aspirin dosage"}],
        internal_to_official={"i1": "c1", "i2": "c2"},
        chunk_to_doc={"c1": "d1", "c2": "d2"},
    )
    assert seen == [("aspirin dosage", "aspirin"), ("aspirin dosage", "insulin")]
    assert metrics["macro_f2"] == 1.0
    assert records[0]["candidates"][0]["rerank_score"] == 0.01


@pytest.mark.parametrize("scores", [[0.9], [float("nan"), 0.1]])
def test_validation_rejects_invalid_model_scores(scores):
    records = [{"id": "q1", "query": "query", "candidates": [
        {"chunk_id": "c1", "text": "one"}, {"chunk_id": "c2", "text": "two"},
    ]}]
    with pytest.raises(ValueError, match="scores|finite"):
        evaluate_reranker_on_validation(lambda pairs: scores, records, [{"id": "q1", "relevant_chunks": ["c1"]}])


def test_validation_rejects_missing_text_and_query_mismatch():
    records = [{"id": "q1", "candidates": []}]
    labels = [{"id": "q1", "relevant_chunks": []}]
    with pytest.raises(ValueError, match="query text"):
        evaluate_reranker_on_validation(lambda pairs: [], records, labels)
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_reranker_on_validation(lambda pairs: [], records, [{"id": "other"}], queries={"q1": "query"})


def test_empty_validation_candidates_do_not_call_model():
    def predict(pairs):
        raise AssertionError("Model must not be called on empty candidate lists")
    result = evaluate_reranker_on_validation(
        predict, [{"id": "q", "query": "query", "candidates": []}],
        [{"id": "q", "relevant_chunks": ["missing"]}], fallback=1,
    )
    assert result["macro_f2"] == 0


def test_training_forwards_lr_evaluates_and_reloads_best_checkpoint(monkeypatch, tmp_path):
    sentence_transformers = pytest.importorskip("sentence_transformers")
    import src.training.finetune as finetune

    class FakeCrossEncoder:
        def __init__(self, path=None, device="cpu"):
            self.device = device
            self.score = read_json(Path(path) / "saved.json")["score"] if path else 0.1

        def predict(self, pairs):
            assert pairs == [("validation query", "positive")]
            return [self.score]

        def save(self, path):
            write_json(Path(path) / "saved.json", {"score": self.score})

        def fit(self, **kwargs):
            assert kwargs["optimizer_params"] == {"lr": 0.003}
            assert kwargs["warmup_steps"] == 1
            assert kwargs["save_best_model"] is True
            evaluator = kwargs["evaluator"]
            assert evaluator.primary_metric == "macro_f2"
            for epoch, score in enumerate([0.9, 0.1]):
                self.score = score
                evaluator(self, epoch=epoch)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(finetune, "prepare_cross_encoder_examples", lambda pairs: pairs)
    trainer = CrossEncoderTrainer(output_dir=tmp_path / "checkpoint")
    trainer.model = FakeCrossEncoder()
    manifest = trainer.train(
        [{"query_id": "train_q", "doc_id": "train_doc", "query": "train", "text": "text", "label": 1.0}],
        val_records=[{"id": "val_q", "candidates": [{"chunk_id": "c1", "doc_id": "val_doc", "text": "positive"}]}],
        val_labels=[{"id": "val_q", "relevant_chunks": ["c1"]}],
        val_queries=[{"id": "val_q", "text": "validation query"}],
        epochs=2, batch_size=1, lr=0.003, warmup_ratio=0.5,
    )
    assert manifest["baseline_metrics"]["macro_f2"] == 0
    assert manifest["validation_metrics"]["macro_f2"] == 1
    assert manifest["best_epoch"] == 0
    assert trainer.model.score == 0.9


def test_training_rejects_leakage_before_loading_model(tmp_path):
    trainer = CrossEncoderTrainer(output_dir=tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="Document leakage"):
        trainer.train(
            [{"query_id": "train", "doc_id": "shared", "label": 0.0}],
            val_records=[{"id": "val", "query": "query", "candidates": [{"chunk_id": "c", "doc_id": "shared", "text": "text"}]}],
            val_labels=[{"id": "val", "relevant_chunks": ["c"]}],
        )
    assert not trainer.output_dir.exists()


def test_cli_dry_run_reads_validation_queries_and_respects_overrides(tmp_path):
    import yaml
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, [{"query_id": "train", "doc_id": "train_doc", "query": "train", "text": "text", "label": 1.0}])
    val_dir = tmp_path / "val"
    write_jsonl(val_dir / "candidates.jsonl", [{"id": "val", "candidates": [{"chunk_id": "c", "doc_id": "val_doc", "text": "text"}]}])
    write_json(val_dir / "labels.json", [{"id": "val", "relevant_chunks": ["c"]}])
    write_json(val_dir / "queries.json", [{"id": "val", "text": "validation query"}])
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"training": {"train_data": str(train_file), "val_data": str(val_dir / "candidates.jsonl"),
                      "output_dir": str(tmp_path / "unused"), "epochs": 7, "learning_rate": .1, "warmup_ratio": .25}}))
    output = tmp_path / "checkpoint"
    command = [sys.executable, str(REPO_ROOT / "scripts/finetune_reranker.py"), "--config", str(config),
               "--output-dir", str(output), "--epochs", "2", "--lr", ".003", "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = read_json(output / "training_manifest.json")
    assert manifest["epochs"] == 2
    assert manifest["learning_rate"] == .003
    assert manifest["validation_metrics"] is None
    settings = yaml.safe_load((output / "training_config.yaml").read_text())["training"]
    assert settings["warmup_ratio"] == .25
    assert not (tmp_path / "unused").exists()


def test_cli_rejects_missing_validation_data(tmp_path):
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, [{"query_id": "train", "doc_id": "train_doc", "query": "train", "text": "text", "label": 1.0}])
    output = tmp_path / "checkpoint"
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/finetune_reranker.py"),
                            "--train-data", str(train_file), "--val-data", str(tmp_path / "missing.jsonl"),
                            "--output-dir", str(output), "--dry-run"], capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("with_validation", [True, False])
def test_finetune_tiny_local_cross_encoder(tmp_path, monkeypatch, with_validation):
    """Exercise the real fit/evaluator/checkpoint lifecycle without downloading weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    pytest.importorskip("sentence_transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_cache"))
    monkeypatch.chdir(tmp_path)  # Library training scratch files stay outside the repo.
    torch.manual_seed(42)
    model_dir = tmp_path / "tiny_model"
    vocab = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "aspirin", "pain", "insulin", "diabetes"])}
    backend = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                       cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=32)
    tokenizer.save_pretrained(model_dir)
    model = BertForSequenceClassification(BertConfig(vocab_size=len(vocab), hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=32, num_labels=1))
    model.save_pretrained(model_dir)
    trainer = CrossEncoderTrainer(model_name=str(model_dir), output_dir=tmp_path / "checkpoint", device="cpu")
    trainer.initialize_model()
    initial = next(trainer.model.model.parameters()).detach().clone()
    train_pairs = [
        {"query_id": "train_q", "doc_id": "train_doc", "query": "aspirin pain", "text": "aspirin pain", "label": 1.0},
        {"query_id": "train_q", "doc_id": "other_train_doc", "query": "aspirin pain", "text": "insulin diabetes", "label": 0.0},
    ]
    manifest = trainer.train(train_pairs,
        val_records=[{"id": "val_q", "candidates": [{"chunk_id": "val_chunk", "doc_id": "val_doc", "text": "aspirin pain"}]}] if with_validation else None,
        val_labels=[{"id": "val_q", "relevant_chunks": ["val_chunk"]}] if with_validation else None,
        val_queries=[{"id": "val_q", "text": "aspirin pain"}] if with_validation else None,
        epochs=1, batch_size=2, lr=.001, warmup_ratio=0, threshold=None,
    )
    if with_validation:
        assert manifest["validation_metrics"]["macro_f2"] == 1.0
        assert manifest["best_epoch"] is not None
    else:
        assert manifest["validation_metrics"] == {}
    assert (trainer.output_dir / "config.json").exists()
    assert (trainer.output_dir / "model.safetensors").exists()
    assert not torch.equal(initial, next(trainer.model.model.parameters()).detach())
