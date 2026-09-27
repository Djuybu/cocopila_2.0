"""BM25/BGE/E5 retrieval benchmark and RRF sweep on one fixed labelled split."""
from contextlib import ExitStack
import csv
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import yaml

from src.data.loader import load_records
from src.evaluation.candidate_recall import evaluate_candidate_recall
from src.pipeline.retrieve import (build_bm25_index, build_dense_index, corpus_fingerprint,
                                  create_generator, load_dataset)
from src.retrieval.fusion import reciprocal_rank_fusion, union_candidates
from src.retrieval.schema import standardize_candidates, validate_candidate_records
from src.utils.config import run_directory
from src.utils.io import read_json, write_json, write_jsonl
from src.utils.logging import run_logger
from src.utils.profiling import ResourceProfile


def _commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _synchronize():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_retrieval_benchmark(config):
    bench = config["benchmark"]
    chunks, queries, registry = load_dataset(config)
    if not queries:
        raise ValueError("Benchmark split contains no queries")
    labels = load_records(bench["labels_path"])
    if any(not row["relevant_chunks"] or not row["relevant_docs"] for row in labels):
        raise ValueError("Retrieval benchmark needs positive labels for every query")
    from src.submission.validator import SubmissionValidator
    valid, errors = SubmissionValidator(**registry).validate(labels)
    if not valid:
        raise ValueError("; ".join(errors))
    run_dir = run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    write_json(run_dir / "registry.json", registry)
    logger = run_logger(run_dir, config)
    data_manifest = config["data"].get("manifest_path")
    manifest = read_json(data_manifest) if data_manifest else {"label_quality": bench.get("label_quality", "unspecified")}
    provenance = {"owner": bench["owner"], "git_commit": _commit(), "dataset_split": config["data"]["split"],
                  "label_quality": manifest["label_quality"], "corpus_fingerprint": corpus_fingerprint(chunks),
                  "queries_fingerprint": corpus_fingerprint(queries), "labels_fingerprint": corpus_fingerprint(labels),
                  "config": config}
    build_profile = None
    with ResourceProfile() as build:
        for source, cfg in config["retrieval"].items():
            if cfg.get("enabled", False) and bench.get("build_indexes_if_missing", False):
                if source == "bm25" and not Path(cfg["index_path"]).exists():
                    build_bm25_index(config)
                elif source == "dense" and not Path(cfg["index_dir"]).exists():
                    build_dense_index(config)
    build_profile = build.result
    results, timings, reports = {}, {}, {}
    with ExitStack() as stack:
        with ResourceProfile() as startup:
            generator = create_generator(config, chunks, stack)
            for wrapper, _ in generator.retrievers.values():
                retriever = getattr(wrapper, "retriever", wrapper)
                if hasattr(retriever, "_model"):
                    retriever._model()
                if bench.get("warmup", True):
                    retriever.retrieve(queries[0]["text"], 1)
        sources = list(generator.retrievers)
        method_names = list(sources)
        if len(sources) > 1:
            method_names += ["union"] + [f"rrf_k{k}" for k in bench["rrf_constants"]]
        results = {name: [] for name in method_names}
        timings = {name: [] for name in method_names}
        cache_hits, total_calls = 0, 0
        with ResourceProfile() as inference:
            for query in queries:
                rankings, source_times = {}, {}
                for source, (retriever, top_k) in generator.retrievers.items():
                    _synchronize()
                    begin = time.perf_counter()
                    # Warm-cache latency must not stand in for model inference.
                    measured = getattr(retriever, "retriever", retriever) if bench.get("measure_uncached", True) else retriever
                    rankings[source] = measured.retrieve(query["text"], top_k)
                    _synchronize()
                    source_times[source] = time.perf_counter() - begin
                    events = getattr(measured, "events", [])
                    cache_hits += int(bool(events) and events[-1]["cache_hit"])
                    total_calls += 1
                    results[source].append({"id": query["id"], "candidates": rankings[source]})
                    timings[source].append(source_times[source])
                    if measured is not retriever:
                        retriever.store(query["text"], top_k, rankings[source], source_times[source])
                if len(sources) > 1:
                    begin = time.perf_counter()
                    union = union_candidates(rankings)
                    results["union"].append({"id": query["id"], "candidates": union})
                    timings["union"].append(sum(source_times.values()) + time.perf_counter() - begin)
                    for k in bench["rrf_constants"]:
                        begin = time.perf_counter()
                        fused = reciprocal_rank_fusion(rankings, k, config["fusion"]["top_k"])
                        name = f"rrf_k{k}"
                        results[name].append({"id": query["id"], "candidates": fused})
                        timings[name].append(sum(source_times.values()) + time.perf_counter() - begin)
        profiles = {"index_build": build_profile, "model_startup_and_warmup": startup.result,
                    "retrieval": inference.result, "cache_hits": cache_hits, "retriever_calls": total_calls}
        for name, records in results.items():
            for record in records:
                record["candidates"] = standardize_candidates(record["id"], record["candidates"], method=name)
            validate_candidate_records(records, queries, registry)
            write_jsonl(run_dir / name / "candidates.jsonl", records)
            report = evaluate_candidate_recall(records, labels, bench["cutoffs"], registry["internal_to_official"])
            durations = timings[name]
            report["performance"] = {"latency_seconds_mean": statistics.mean(durations),
                                     "latency_seconds_p50": statistics.median(durations),
                                     "latency_seconds_p95": sorted(durations)[math.ceil(.95 * len(durations)) - 1],
                                     "includes_model_startup": False,
                                     "cache_enabled": config.get("retrieval_cache", {}).get("enabled", False),
                                     "timing_mode": "uncached_compute" if bench.get("measure_uncached", True) else "cache_lookup_or_compute"}
            report["provenance"] = provenance
            reports[name] = report
        for name, report in reports.items():
            if name.startswith("rrf_"):
                report["comparison_to_union"] = {
                    branch: {metric: value - reports["union"]["macro"][branch][metric]
                             for metric, value in report["macro"][branch].items()}
                    for branch in ("chunks", "documents")
                }
            write_json(run_dir / name / "recall.json", report)
        summary = {"provenance": provenance, "profiles": profiles,
                   "methods": {name: {"macro": report["macro"], "performance": report["performance"],
                                      "miss_all_chunk_queries": report["miss_all_chunk_queries"],
                                      "comparison_to_union": report.get("comparison_to_union")}
                               for name, report in reports.items()}}
        write_json(run_dir / "benchmark.json", summary)
        rows = []
        for name, report in reports.items():
            for k in bench["cutoffs"]:
                rows.append({"method": name, "k": k, "chunk_recall": report["macro"]["chunks"][f"recall@{k}"],
                             "doc_recall": report["macro"]["documents"][f"recall@{k}"],
                             "latency_seconds_mean": report["performance"]["latency_seconds_mean"]})
        with (run_dir / "comparison.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _append_log(config, run_dir, summary)
        logger.info("candidate recall=%s profiles=%s", summary["methods"], profiles)
    generator = None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def _append_log(config, run_dir, summary):
    log_path = Path(config["benchmark"]["experiment_log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    dense = config["retrieval"].get("dense", {})
    entries = []
    for method, values in summary["methods"].items():
        entries.append({
            "run_id": config["run_name"], "owner": config["benchmark"]["owner"],
            "git_commit": summary["provenance"]["git_commit"], "dataset_split": config["data"]["split"],
            "model": dense.get("model") if dense.get("enabled") else "BM25",
            "revision": dense.get("revision"), "method": method,
            "label_quality": summary["provenance"]["label_quality"],
            "chunk_recall_at_100": values["macro"]["chunks"].get("recall@100"),
            "doc_recall_at_100": values["macro"]["documents"].get("recall@100"),
            "latency_seconds_mean": values["performance"]["latency_seconds_mean"],
            "peak_ram_bytes": summary["profiles"]["retrieval"]["peak_process_rss_bytes"],
            "peak_vram_bytes": summary["profiles"]["retrieval"]["peak_cuda_allocated_bytes"],
            "output_path": str(run_dir),
        })
    existed = log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        if not existed:
            writer.writeheader()
        writer.writerows(entries)


def evaluate_retrieval_files(candidates_path, labels_path, output_dir, cutoffs, internal_to_official=None):
    report = evaluate_candidate_recall(load_records(candidates_path), load_records(labels_path), cutoffs,
                                       internal_to_official)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "candidate_recall.json", report)
    write_json(output_dir / "missed_queries.json", {
        "chunks": report["miss_all_chunk_queries"], "documents": report["miss_all_doc_queries"]})
    return report
