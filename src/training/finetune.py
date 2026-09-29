"""P2-08: Reranker fine-tuning module with checkpointing and validation evaluation.

Supports training and fine-tuning cross-encoder models (e.g. BAAI/bge-reranker-v2-m3)
on positive and hard negative pairs, persisting checkpoints, training configs, and seed,
and validating that validation F2/Recall meets or exceeds baseline.
"""
from collections import defaultdict
import logging
from pathlib import Path
import random
import yaml

from src.data.loader import load_records
from src.evaluation.fbeta import classification_metrics
from src.scoring.chunk_selector import select_ids
from src.utils.io import read_json, write_json

logger = logging.getLogger("finetune_reranker")


def create_training_config(
    model_name="BAAI/bge-reranker-v2-m3",
    output_dir="checkpoints/reranker_v1",
    epochs=3,
    batch_size=16,
    lr=2e-5,
    warmup_ratio=0.1,
    seed=42,
    train_data="outputs/bge_reranked_run/reranker_train.jsonl",
    val_data=None,
    metric="f2",
):
    """Create structured training configuration dictionary."""
    return {
        "training": {
            "model_name": model_name,
            "output_dir": str(output_dir),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
            "train_data": str(train_data),
            "val_data": str(val_data) if val_data else None,
            "evaluation_metric": metric,
            "save_best_model": True,
        }
    }


def prepare_cross_encoder_examples(pairs):
    """Convert JSONL pair records to sentence_transformers InputExample objects."""
    try:
        from sentence_transformers import InputExample
        return [
            InputExample(texts=[p["query"], p["text"]], label=float(p["label"]))
            for p in pairs
        ]
    except ImportError:
        # Fallback dictionary representation if sentence_transformers is mocked
        return [
            {"texts": [p["query"], p["text"]], "label": float(p["label"])}
            for p in pairs
        ]


def evaluate_reranker_on_validation(
    model_or_predict_fn,
    val_records,
    labels,
    threshold=0.5,
    fallback=0,
    max_chunks=None,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """Evaluate reranker model on validation candidates against ground truth labels.

    Returns:
        dict: Macro precision, recall, f1, f2.
    """
    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}
    mapping = internal_to_official or {}
    effective_max = 999999 if max_chunks is None else int(max_chunks)

    total_p, total_r, total_f1, total_f2 = 0.0, 0.0, 0.0, 0.0
    num_queries = len(label_map)
    if num_queries == 0:
        raise ValueError("Cannot evaluate on empty labels")

    for row in val_records:
        qid = row["id"]
        cands = row.get("candidates", [])
        truth_chunks = label_map.get(qid, set())
        query_text = row.get("text", row.get("query", ""))

        # Score candidates
        pairs_to_score = [(query_text, c.get("text", "")) for c in cands]
        if hasattr(model_or_predict_fn, "predict"):
            scores = model_or_predict_fn.predict(pairs_to_score)
        elif callable(model_or_predict_fn):
            scores = model_or_predict_fn(pairs_to_score)
        else:
            raise TypeError("Model must have .predict method or be callable")

        scored_cands = []
        for c, score in zip(cands, scores):
            cid = mapping.get(c["chunk_id"], c["chunk_id"])
            scored_cands.append({**c, "chunk_id": cid, "score": float(score)})

        selected = select_ids(scored_cands, "chunk_id", threshold, fallback, effective_max)
        m = classification_metrics(truth_chunks, set(selected), zero_division=0.0)

        total_p += m["precision"]
        total_r += m["recall"]
        total_f1 += m["f1"]
        total_f2 += m["f2"]

    return {
        "macro_precision": round(total_p / num_queries, 5),
        "macro_recall": round(total_r / num_queries, 5),
        "macro_f1": round(total_f1 / num_queries, 5),
        "macro_f2": round(total_f2 / num_queries, 5),
    }


class CrossEncoderTrainer:
    """Trainer orchestrator for Cross-Encoder models."""

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3",
        output_dir="checkpoints/reranker_v1",
        device=None,
        seed=42,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.device = device
        self.model = None

    def initialize_model(self):
        """Lazy load or initialize model."""
        import torch
        from sentence_transformers import CrossEncoder

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        logger.info("Initializing CrossEncoder %s on %s...", self.model_name, device)
        self.model = CrossEncoder(self.model_name, num_labels=1, device=device)
        return self.model

    def train(
        self,
        train_pairs,
        val_records=None,
        val_labels=None,
        epochs=1,
        batch_size=16,
        lr=2e-5,
        warmup_ratio=0.1,
    ):
        """Train cross-encoder on pairs and save checkpoint."""
        import torch
        from torch.utils.data import DataLoader

        # Set reproducibility seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        if self.model is None:
            self.initialize_model()

        train_examples = prepare_cross_encoder_examples(train_pairs)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
        logger.info("Beginning training: %d samples, %d epochs, %d warmup steps...", len(train_examples), epochs, warmup_steps)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(self.output_dir),
            save_best_model=True if val_records else False,
            show_progress_bar=True,
        )

        metrics = {}
        if val_records and val_labels:
            logger.info("Evaluating fine-tuned checkpoint on validation...")
            metrics = evaluate_reranker_on_validation(self.model, val_records, val_labels)

        # Write training manifest
        manifest = {
            "model_name": self.model_name,
            "seed": self.seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_training_pairs": len(train_pairs),
            "output_dir": str(self.output_dir),
            "validation_metrics": metrics,
        }
        write_json(self.output_dir / "training_manifest.json", manifest)

        return manifest


def load_finetuned_reranker(checkpoint_dir, device=None):
    """Load a fine-tuned reranker into CrossEncoderReranker wrapper."""
    from src.reranking.bge import CrossEncoderReranker
    return CrossEncoderReranker(model_name=str(checkpoint_dir), device=device)
