"""Score all retrieved candidates; disabled reranking is recorded explicitly."""
import logging
from pathlib import Path
from src.data.loader import load_records
from src.data.schema import unique_ids
from src.reranking.bge import CrossEncoderReranker
from src.utils.config import load_config
from src.utils.io import read_json, write_jsonl
from src.utils.logging import run_logger


def run_reranking(run_dir, *, reranker=None):
    run_dir = Path(run_dir)
    if (run_dir / "reranked.jsonl").exists():
        raise FileExistsError(run_dir / "reranked.jsonl")
    config = load_config(run_dir / "config.yaml")
    logger = run_logger(run_dir, config)
    queries = {row["id"]: row["text"] for row in read_json(run_dir / "queries.json")}
    records = load_records(run_dir / "candidates.jsonl")
    if unique_ids(records, "id") != set(queries):
        raise ValueError("Candidate queries do not match the run")
    cfg = config["reranker"]
    if cfg["enabled"] and reranker is None:
        reranker = CrossEncoderReranker(cfg["model"], cfg["device"], batch_size=cfg["batch_size"])
    def scored():
        for row in records:
            candidates = [{**candidate, "query_id": row["id"]} for candidate in row["candidates"]]
            yield {"id": row["id"], "reranking_enabled": cfg["enabled"],
                   "candidates": reranker.rerank(queries[row["id"]], candidates) if cfg["enabled"] else candidates}
    write_jsonl(run_dir / "reranked.jsonl", scored())
    logger.info("reranking enabled=%s output_path=%s", cfg["enabled"], run_dir / "reranked.jsonl")
    return run_dir / "reranked.jsonl"
