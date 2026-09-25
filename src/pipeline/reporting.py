"""Run evaluation and package validated submission artifacts."""
import logging
from pathlib import Path
from src.data.loader import load_records
from src.evaluation.evaluator import evaluate_submission
from src.submission.generator import generate_submission
from src.submission.validator import SubmissionValidator
from src.submission.zipper import pack_submission
from src.utils.config import load_config
from src.utils.io import read_json, write_json
from src.utils.logging import run_logger


def run_evaluation(run_dir, labels_path=None, stage="predictions", k=None):
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    cfg = config.get("evaluation", {})
    labels_path = labels_path or cfg.get("labels_path")
    if not labels_path:
        raise ValueError("Provide --ground-truth or evaluation.labels_path")
    truth = load_records(labels_path)
    registry = read_json(run_dir / "registry.json")
    validator = SubmissionValidator(**registry)
    valid, errors = validator.validate(truth)
    if not valid:
        raise ValueError("Invalid ground truth: " + "; ".join(errors))
    if stage == "predictions":
        predictions = read_json(run_dir / "predictions.json")
        valid, errors = validator.validate(predictions)
        if not valid:
            raise ValueError("; ".join(errors))
        metrics = cfg.get("metrics")
    elif stage == "candidates":
        k = k if k is not None else cfg["candidate_k"]
        rows = load_records(run_dir / "candidates.jsonl")
        raw = [{"id": row["id"],
                "relevant_docs": list(dict.fromkeys(c["doc_id"] for c in row["candidates"])),
                "relevant_chunks": [c["chunk_id"] for c in row["candidates"]]} for row in rows]
        predictions = generate_submission(raw, validator, registry["internal_to_official"])
        metrics = [f"recall@{k}", f"precision@{k}"]
    else:
        raise ValueError("Unknown evaluation stage")
    results = evaluate_submission(predictions, truth, metrics, zero_division=cfg.get("zero_division", 0.0))
    output = run_dir / f"metrics_{stage}.json"
    write_json(output, results)
    run_logger(run_dir, config).info("metrics=%s output_path=%s", results["macro"], output)
    return results


def make_submission(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    registry = read_json(run_dir / "registry.json")
    validator = SubmissionValidator(**registry)
    output_dir = Path(output_dir or config["submission"]["output_dir"])
    source = run_dir / "predictions.json"
    target = output_dir / f"{config['run_name']}.zip"
    result = pack_submission(source, target, validator)
    run_logger(run_dir, config).info("submission output_path=%s", result)
    return result
