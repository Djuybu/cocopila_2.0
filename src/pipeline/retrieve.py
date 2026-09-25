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
                              lowercase=cfg["lowercase"], text_key=cfg["text_key"])
    retriever.build_index(chunks)
    retriever.save_index(cfg["index_path"])
    return cfg["index_path"]


def build_dense_index(config):
    from src.data.indexer import QdrantIndexer
    from src.retrieval.dense import DenseRetriever
    chunks, _, _ = load_dataset(config)
    cfg = config["retrieval"]["dense"]
    path = Path(cfg["index_dir"])
    path.mkdir(parents=True, exist_ok=False)
    retriever = DenseRetriever(cfg["model"], cfg["device"], query_prefix=cfg["query_prefix"],
                               max_seq_length=cfg["max_seq_length"])
    indexer = QdrantIndexer(path, cfg["collection"], cfg["dimension"])
    try:
        indexer.create_collection()
        batch_size = cfg["batch_size"]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = retriever.encode_documents([row[cfg["text_key"]] for row in batch], batch_size)
            indexer.index_documents(batch, vectors, batch_size)
        write_json(path / "metadata.json", {
            "model": cfg["model"], "dimension": cfg["dimension"], "text_key": cfg["text_key"],
            "max_seq_length": cfg["max_seq_length"], "corpus_fingerprint": corpus_fingerprint(chunks),
        })
    finally:
        indexer.client.close()
    return str(path)


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
        elif name == "dense":
            from qdrant_client import QdrantClient
            from src.retrieval.dense import DenseRetriever
            metadata = read_json(Path(cfg["index_dir"]) / "metadata.json")
            expected = {"model": cfg["model"], "dimension": cfg["dimension"], "text_key": cfg["text_key"],
                        "max_seq_length": cfg["max_seq_length"], "corpus_fingerprint": corpus_fingerprint(chunks)}
            if metadata != expected:
                raise ValueError("Dense index/config or corpus mismatch; rebuild into a new index")
            client = QdrantClient(path=cfg["index_dir"])
            stack.callback(client.close)
            retriever = DenseRetriever(cfg["model"], cfg["device"], client=client,
                                       collection_name=cfg["collection"], query_prefix=cfg["query_prefix"],
                                       max_seq_length=cfg["max_seq_length"])
        else:
            raise NotImplementedError(f"Retriever {name} is not implemented")
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
            rows = ({"id": query["id"], "candidates": generator.generate(query["text"], config["fusion"]["top_k"])}
                    for query in queries)
            write_jsonl(run_dir / "candidates.jsonl", rows)
        logger.info("retrieval completed")
    except Exception:
        logger.exception("retrieval failed; keep run artifacts for diagnosis, use a new run_name to retry")
        raise
    return run_dir
