"""Rerank and evaluate a received P1-11 bundle without loading its original corpus."""
import math
from pathlib import Path
import statistics
import time

from src.data.loader import load_records
from src.evaluation.reranking_metrics import evaluate_reranked_candidates
from src.pipeline.handoff import validate_reranking_input
from src.pipeline.rerank import run_reranking
from src.utils.io import read_json, write_json
from src.utils.logging import run_logger
from src.utils.config import load_config


def benchmark_reranking(run_dir, output_dir, *, reranker_config=None, ks=(1, 3, 5, 10, 20, 50, 100, 200), reranker=None):
    source, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("Cutoffs must be positive integers")
    manifest = validate_reranking_input(source)
    effective = reranker_config if reranker_config is not None else load_config(source / "config.yaml")
    if not effective.get("reranker", effective).get("enabled"):
        raise ValueError("Benchmark requires an enabled reranker")
    labels = read_json(source / "labels.json")
    mapping = read_json(source / "registry.json")["internal_to_official"]
    candidates = load_records(source / "candidates.jsonl")
    before = evaluate_reranked_candidates(candidates, labels, ks, mapping)
    before["stage"] = "before_reranking"
    started = time.perf_counter()
    run_reranking(source, output_dir=output, reranker_config=reranker_config, reranker=reranker, measure=True)
    elapsed = time.perf_counter() - started
    scored = load_records(output / "reranked.jsonl")
    metadata = read_json(output / "reranking_metadata.json")
    after = evaluate_reranked_candidates(scored, labels, ks, mapping)
    seconds = [row["seconds"] for row in metadata["performance"]["per_query"]]
    total_inference = sum(seconds)
    report = {"run_id": output.name, "stage": "reranking_benchmark", "input_manifest": manifest,
              "label_quality": manifest["label_quality"], "id_namespace": manifest["id_namespace"],
              "limitations": manifest["limitations"], "backend_injected": metadata["backend_injected"],
              "reranker": metadata["config"], "query_count": len(scored), "candidate_count": manifest["candidate_count"],
              "before": before, "after": after,
              "delta_chunks": {metric: value - before["macro"]["chunks"][metric] for metric, value in after["macro"]["chunks"].items()},
              "performance": {**metadata["performance"], "end_to_end_seconds": elapsed,
                              "latency_seconds_mean": statistics.mean(seconds) if seconds else 0.0,
                              "latency_seconds_p50": statistics.median(seconds) if seconds else 0.0,
                              "latency_seconds_p95": sorted(seconds)[math.ceil(.95 * len(seconds)) - 1] if seconds else 0.0,
                              "queries_per_second": len(seconds) / total_inference if total_inference else 0.0,
                              "candidates_per_second": manifest["candidate_count"] / total_inference if total_inference else 0.0}}
    write_json(output / "reranking_benchmark.json", report)
    run_logger(output, load_config(output / "config.yaml")).info("label_quality=%s metrics=%s output_path=%s",
        manifest["label_quality"], after["macro"], output / "reranking_benchmark.json")
    return report
