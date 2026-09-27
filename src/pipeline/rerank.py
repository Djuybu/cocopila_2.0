"""Score all retrieved candidates; disabled reranking is recorded explicitly."""
import copy
import shutil
import time
from pathlib import Path
import yaml
from src.data.loader import load_records
from src.data.schema import unique_ids
from src.reranking.factory import create_reranker
from src.reranking.base import validate_reranked_candidates
from src.utils.config import load_config
from src.utils.io import read_json, write_json, write_jsonl
from src.utils.logging import run_logger


def run_reranking(run_dir, *, reranker=None, reranker_config=None, output_dir=None, measure=False):
    """Score all input candidates; optionally snapshot a separate, immutable-input run.

    A config override requires a new output directory, so received handoff files
    and their checksums stay untouched. Old in-place stage calls still work.
    """
    source = Path(run_dir).resolve()
    target = Path(output_dir).resolve() if output_dir is not None else source
    if reranker_config is not None and target == source:
        raise ValueError("A reranker config override requires a separate output_dir")
    if target != source and (target.is_relative_to(source) or source.is_relative_to(target)):
        raise ValueError("Reranking output must be separate from the input run")
    if (target != source and target.exists()) or (target / "reranked.jsonl").exists():
        raise FileExistsError(target)
    if (source / "manifest.json").exists():
        from src.pipeline.handoff import validate_reranking_input
        validate_reranking_input(source)
    config = load_config(source / "config.yaml")
    if reranker_config is not None:
        # Replace, don't merge: BGE-specific values must not leak into Qwen.
        config["reranker"] = copy.deepcopy(reranker_config.get("reranker", reranker_config))
    query_rows = read_json(source / "queries.json")
    unique_ids(query_rows, "id")
    queries = {row["id"]: row["text"] for row in query_rows}
    records = load_records(source / "candidates.jsonl")
    if unique_ids(records, "id") != set(queries):
        raise ValueError("Candidate queries do not match the run")
    cfg = config["reranker"]
    backend_injected = reranker is not None
    if cfg["enabled"] and reranker is None:
        reranker = create_reranker(cfg)
    if target != source:
        target.mkdir(parents=True, exist_ok=False)
        for name in ("queries.json", "candidates.jsonl", "registry.json", "labels.json"):
            if (source / name).exists():
                shutil.copyfile(source / name, target / name)
        if (source / "manifest.json").exists():
            shutil.copyfile(source / "manifest.json", target / "input_manifest.json")
        config.update(run_name=target.name, output_dir=str(target.parent))
        # The copied labels are authoritative; don't retain sender machine paths.
        if (target / "labels.json").exists():
            config.setdefault("evaluation", {})["labels_path"] = "labels.json"
        config["reranking_input"] = {"run_dir": str(source), "manifest_path": str(source / "manifest.json")
                                     if (source / "manifest.json").exists() else None}
        with (target / "config.yaml").open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        config = load_config(target / "config.yaml")
    logger = run_logger(target, config)
    torch_backend, cuda_device = None, None
    if measure and cfg["enabled"] and not backend_injected and str(cfg.get("device", "")).startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            torch_backend, cuda_device = torch, torch.device(cfg["device"])
    def synchronize():
        if torch_backend is not None:
            torch_backend.cuda.synchronize(cuda_device)
    model_load_seconds, timings = None, []
    if measure and cfg["enabled"]:
        started = time.perf_counter()
        if hasattr(reranker, "load_model"):
            reranker.load_model()
        synchronize()
        model_load_seconds = time.perf_counter() - started
        if torch_backend is not None:
            torch_backend.cuda.reset_peak_memory_stats(cuda_device)
    def scored():
        for row in records:
            candidates = [{**candidate, "query_id": row["id"]} for candidate in row["candidates"]]
            original = copy.deepcopy(candidates) if cfg["enabled"] else None
            if measure:
                synchronize()
                started = time.perf_counter()
            ranked = reranker.rerank(queries[row["id"]], candidates) if cfg["enabled"] else candidates
            if measure:
                synchronize()
                timings.append({"query_id": row["id"], "candidate_count": len(candidates),
                                "seconds": time.perf_counter() - started})
            if cfg["enabled"]:
                validate_reranked_candidates(original, ranked)
            yield {"id": row["id"], "reranking_enabled": cfg["enabled"],
                   "candidates": ranked}
    write_jsonl(target / "reranked.jsonl", scored())
    write_json(target / "reranking_metadata.json", {
        "enabled": cfg["enabled"], "adapter": type(reranker).__name__ if cfg["enabled"] else None,
        "backend_injected": backend_injected,
        "config": cfg, "input_run_dir": str(source),
        "query_count": len(records), "candidate_count": sum(len(row["candidates"]) for row in records),
        "label_quality": config.get("label_quality", "unspecified"),
        "performance": {"per_query": timings, "model_load_seconds": model_load_seconds,
                        "timing_includes_model_load": False, "warmup": False,
                        "peak_cuda_allocated_bytes": torch_backend.cuda.max_memory_allocated(cuda_device)
                        if torch_backend is not None else None} if measure else None,
    })
    logger.info("reranking enabled=%s adapter=%s output_path=%s", cfg["enabled"],
                type(reranker).__name__ if reranker else None, target / "reranked.jsonl")
    return target / "reranked.jsonl"
