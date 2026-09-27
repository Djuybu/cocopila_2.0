"""Config-driven index building and candidate generation."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import yaml

from src.data.adapter import build_mappings
from src.data.loader import load_records
from src.data.schema import validate_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.candidate_generator import CandidateGenerator
from src.retrieval.cache import CachedRetriever
from src.retrieval.schema import standardize_candidates, validate_candidate_records
from src.submission.validator import SubmissionValidator
from src.utils.config import run_directory
from src.utils.io import read_json, write_json, write_jsonl
from src.utils.logging import run_logger


def corpus_fingerprint(chunks):
    return hashlib.sha256(json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_dataset(config):
    data = config["data"]
    documents = load_records(data["documents_path"])
    chunks = load_records(data["chunks_path"])
    queries = load_records(data["queries_path"])
    validate_corpus(documents, chunks, queries)
    chunk_to_doc, internal = build_mappings(chunks)
    mapping_dir = Path(data["mappings_dir"])
    if read_json(mapping_dir / "chunk_to_doc.json") != chunk_to_doc:
        raise ValueError("chunk_to_doc mapping does not match prepared corpus")
    if read_json(mapping_dir / "internal_to_official_id.json") != internal:
        raise ValueError("internal_to_official mapping does not match prepared corpus")
    registry = {"expected_query_ids": [q["id"] for q in queries],
                "doc_ids": [doc["doc_id"] for doc in documents],
                "chunk_to_doc": chunk_to_doc, "internal_to_official": internal}
    SubmissionValidator(**registry)
    return chunks, queries, registry


def build_bm25_index(config):
    chunks, _, _ = load_dataset(config)
    cfg = config["retrieval"]["bm25"]
    retriever = BM25Retriever(k1=cfg["k1"], b=cfg["b"], epsilon=cfg["epsilon"],
                              lowercase=cfg["lowercase"], text_key=cfg["text_key"],
                              tokenizer=cfg.get("tokenizer", "whitespace"))
    retriever.build_index(chunks)
    retriever.save_index(cfg["index_path"])
    return cfg["index_path"]


def build_dense_index(config):
    import gc
    import logging
    import sys
    from src.data.indexer import QdrantIndexer
    from src.retrieval.dense import DenseRetriever
    chunks, _, _ = load_dataset(config)
    cfg = config["retrieval"]["dense"]
    path = Path(cfg["index_dir"])
    path.mkdir(parents=True, exist_ok=False)
    retriever = configured_dense(cfg)
    indexer = QdrantIndexer(path, cfg["collection"], cfg["dimension"], distance=cfg.get("distance", "cosine"))
    try:
        indexer.create_collection()
        batch_size = cfg["batch_size"]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = retriever.encode_documents([row[cfg["text_key"]] for row in batch], batch_size,
                                                show_progress=cfg.get("show_progress", False))
            indexer.index_documents(batch, vectors, batch_size)
            if start // batch_size % 100 == 0:
                logging.info("dense index: %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))
        write_json(path / "metadata.json", dense_metadata(cfg, chunks))
    finally:
        indexer.client.close()
        del retriever
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return str(path)


def configured_dense(cfg, client=None):
    from src.retrieval.dense import DenseRetriever
    return DenseRetriever(
        cfg["model"], cfg["device"], client=client, collection_name=cfg["collection"],
        query_prefix=cfg["query_prefix"], document_prefix=cfg.get("document_prefix", ""),
        normalize_embeddings=cfg.get("normalize_embeddings", True),
        max_seq_length=cfg["max_seq_length"], batch_size=cfg["batch_size"],
        revision=cfg.get("revision"), model_cache_dir=cfg.get("model_cache_dir"),
        model_dtype=cfg.get("model_dtype"))


def dense_metadata(cfg, chunks):
    return {
        "model": cfg["model"], "dimension": cfg["dimension"], "text_key": cfg["text_key"],
        "max_seq_length": cfg["max_seq_length"], "corpus_fingerprint": corpus_fingerprint(chunks),
        "distance": cfg.get("distance", "cosine"), "document_prefix": cfg.get("document_prefix", ""),
        "normalize_embeddings": cfg.get("normalize_embeddings", True), "revision": cfg.get("revision"),
        "model_dtype": cfg.get("model_dtype"),
    }


def create_generator(config, chunks, stack):
    retrievers = {}
    for name, cfg in config["retrieval"].items():
        if not cfg.get("enabled", False):
            continue
        if name == "bm25":
            retriever = BM25Retriever(cfg["index_path"])
            if corpus_fingerprint(retriever.corpus) != corpus_fingerprint(chunks):
                raise ValueError("BM25 index is stale or belongs to a different corpus")
            for parameter in ("k1", "b", "epsilon", "lowercase", "text_key"):
                if getattr(retriever, parameter) != cfg[parameter]:
                    raise ValueError(f"BM25 index/config mismatch: {parameter}; rebuild into a new index")
            if retriever.tokenizer != cfg.get("tokenizer", "whitespace"):
                raise ValueError("BM25 tokenizer mismatch; rebuild into a new index")
        elif name == "dense":
            from qdrant_client import QdrantClient
            metadata = read_json(Path(cfg["index_dir"]) / "metadata.json")
            expected = dense_metadata(cfg, chunks)
            legacy_defaults = {"distance": "cosine", "document_prefix": "", "normalize_embeddings": True,
                               "revision": None, "model_dtype": None}
            if {**legacy_defaults, **metadata} != expected:
                raise ValueError("Dense index/config or corpus mismatch; rebuild into a new index")
            client = QdrantClient(path=cfg["index_dir"])
            stack.callback(client.close)
            retriever = configured_dense(cfg, client)
        else:
            raise NotImplementedError(f"Retriever {name} is not implemented")
        cache = config.get("retrieval_cache", {})
        if cache.get("enabled", False):
            retriever = CachedRetriever(retriever, cache["cache_dir"],
                                         {"source": name, "config": cfg, "corpus": corpus_fingerprint(chunks)})
        retrievers[name] = (retriever, cfg["top_k"])
    fusion = config["fusion"]
    return CandidateGenerator(retrievers, method=fusion["method"],
                              rrf_k=fusion.get("rrf_k", 60), alpha=fusion.get("alpha", 0.7))


def run_retrieval(config):
    chunks, queries, registry = load_dataset(config)
    run_dir = run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    write_json(run_dir / "registry.json", registry)
    write_json(run_dir / "queries.json", queries)
    logger = run_logger(run_dir, config)
    logger.info("dataset_split=%s output_path=%s", config["data"]["split"], run_dir)
    try:
        with ExitStack() as stack:
            generator = create_generator(config, chunks, stack)
            def rows():
                for query in queries:
                    row = {"id": query["id"], "candidates": standardize_candidates(
                        query["id"], generator.generate(query["text"], config["fusion"]["top_k"]))}
                    validate_candidate_records([row], [query], {**registry, "expected_query_ids": [query["id"]]})
                    yield row
            write_jsonl(run_dir / "candidates.jsonl", rows())
        logger.info("retrieval completed")
    except Exception:
        logger.exception("retrieval failed; keep run artifacts for diagnosis, use a new run_name to retry")
        raise
    return run_dir
