"""P2-08: Reranker fine-tuning module with checkpointing and validation evaluation.

Supports training and fine-tuning cross-encoder models (e.g. BAAI/bge-reranker-v2-m3)
on positive and hard negative pairs, persisting checkpoints, training configs, and seed,
and reporting validation F2/Recall for the best checkpoint and the baseline.
"""
import logging
import math
from pathlib import Path
import random

from src.data.schema import unique_ids
from src.data.adapter import official_candidates
from src.evaluation.evaluate_f2 import evaluate_candidate_selection
from src.training.data_generator import validate_no_leakage
from src.utils.io import write_json

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
    queries=None,
):
    """Evaluate reranker model on validation candidates against ground truth labels.

    Returns:
        dict: Macro precision, recall, f1, f2.
    """
    query_map = _validation_queries(val_records, labels, queries)
    mapping = internal_to_official or {}
    scored_records = []
    for row in val_records:
        qid = row["id"]
        cands = row.get("candidates", [])
        query_text = query_map[qid]

        # Score candidates
        pairs_to_score = [(query_text, c.get("text", "")) for c in cands]
        if not pairs_to_score:
            scores = []
        elif hasattr(model_or_predict_fn, "predict"):
            scores = model_or_predict_fn.predict(pairs_to_score)
        elif callable(model_or_predict_fn):
            scores = model_or_predict_fn(pairs_to_score)
        else:
            raise TypeError("Model must have .predict method or be callable")

        if len(scores) != len(cands):
            raise ValueError("Reranker returned the wrong number of scores")
        scored_cands = []
        for c, score in zip(cands, scores):
            if not math.isfinite(float(score)):
                raise ValueError("Reranker score must be finite")
            scored_cands.append({**c, "rerank_score": float(score)})
        if chunk_to_doc is not None:
            scored_cands = official_candidates(scored_cands, mapping, chunk_to_doc)
        scored_records.append({"id": qid, "candidates": scored_cands})
    metrics = evaluate_candidate_selection(
        scored_records, labels, threshold=threshold, fallback=fallback,
        max_chunks=max_chunks, internal_to_official=mapping,
    )["macro"]
    return {
        f"macro_{name}": round(metrics[name], 5)
        for name in ("precision", "recall", "f1", "f2")
    }


def _validation_queries(records, labels, queries=None):
    record_ids = unique_ids(records, "id")
    if not labels or record_ids != unique_ids(labels, "id"):
        raise ValueError("Validation candidate and label query IDs must match exactly and be nonempty")
    if isinstance(queries, dict):
        query_map = dict(queries)
    elif queries is not None:
        unique_ids(queries, "id")
        query_map = {q["id"]: q["text"] for q in queries}
    else:
        query_map = {row["id"]: row.get("text", row.get("query")) for row in records}
    if set(query_map) != record_ids:
        raise ValueError("Validation query text IDs must match candidate IDs exactly")
    for qid, query in query_map.items():
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Missing validation query text for {qid}; supply queries.json")
    return query_map


def _validation_evaluator(records, labels, checkpoint_dir, **selection):
    # Import only for training; the CPU baseline does not require this dependency.
    try:
        from sentence_transformers.base.evaluation.evaluator import BaseEvaluator as SentenceEvaluator
    except ImportError:  # Sentence Transformers 3.x compatibility
        from sentence_transformers.evaluation import SentenceEvaluator

    class ChunkF2Evaluator(SentenceEvaluator):
        primary_metric = "macro_f2"
        greater_is_better = True

        def __init__(self):
            super().__init__()
            self.primary_metric = "macro_f2"
            self.best_metrics = None
            self.best_epoch = None

        def __call__(self, model, output_path=None, epoch=-1, steps=-1, **kwargs):
            metrics = evaluate_reranker_on_validation(model, records, labels, **selection)
            if self.best_metrics is None or metrics["macro_f2"] > self.best_metrics["macro_f2"]:
                # Persist here as well: fit callback wiring differs across library versions.
                model.save(str(checkpoint_dir))
                self.best_metrics = metrics
                self.best_epoch = epoch
            return metrics

    return ChunkF2Evaluator()


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
        val_queries=None,
        internal_to_official=None,
        chunk_to_doc=None,
        threshold=0.5,
        fallback=0,
        max_chunks=None,
    ):
        """Train cross-encoder on pairs and save checkpoint."""
        if not train_pairs:
            raise ValueError("Cannot train on empty pairs")
        if epochs < 1 or batch_size < 1 or lr <= 0 or not 0 <= warmup_ratio <= 1:
            raise ValueError("Invalid training hyperparameters")
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        if (val_records is None) != (val_labels is None):
            raise ValueError("Validation candidates and labels must be supplied together")
        selection = dict(queries=val_queries, internal_to_official=internal_to_official,
                         chunk_to_doc=chunk_to_doc, threshold=threshold,
                         fallback=fallback, max_chunks=max_chunks)
        evaluator = None
        if val_records is not None:
            _validation_queries(val_records, val_labels, val_queries)
            validation_pairs = [
                {"query_id": row["id"], "doc_id": c.get("doc_id"), "label": 0.0}
                for row in val_records for c in row.get("candidates", [])
            ]
            validation_pairs.extend({"query_id": row["id"]} for row in val_records)
            validate_no_leakage(train_pairs, validation_pairs)
            evaluator = _validation_evaluator(val_records, val_labels, self.output_dir, **selection)

        # Set reproducibility seed
        import torch
        from torch.utils.data import DataLoader

        torch.manual_seed(self.seed)
        random.seed(self.seed)

        if self.model is None:
            self.initialize_model()

        train_examples = prepare_cross_encoder_examples(train_pairs)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
        logger.info("Beginning training: %d samples, %d epochs, %d warmup steps...", len(train_examples), epochs, warmup_steps)

        baseline = evaluate_reranker_on_validation(self.model, val_records, val_labels, **selection) if evaluator else None
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr},
            evaluator=evaluator,
            output_path=str(self.output_dir),
            save_best_model=evaluator is not None,
            show_progress_bar=True,
        )

        metrics = {}
        if evaluator is not None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(str(self.output_dir), device=str(self.model.device))
            logger.info("Evaluating fine-tuned checkpoint on validation...")
            metrics = evaluate_reranker_on_validation(self.model, val_records, val_labels, **selection)
        else:
            self.model.save(str(self.output_dir))

        # Write training manifest
        manifest = {
            "model_name": self.model_name,
            "seed": self.seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "warmup_ratio": warmup_ratio,
            "best_epoch": evaluator.best_epoch if evaluator else None,
            "baseline_metrics": baseline,
            "validation_selector": {"threshold": threshold, "fallback": fallback, "max_chunks": max_chunks},
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
