"""Merge independent chunk (P2) and document (P3) selections into predictions.

The chunk branch uses the P2 ``best_chunk_selector.yaml``; the document branch
uses the P3-04 ``best_doc_selector.yaml``. P1/P2 modules are imported read-only.
"""
from pathlib import Path

import yaml

from src.p3.doc_aggregation import aggregate_documents, prepare_official_candidates
from src.scoring.chunk_selector import select_ids

CHUNK_KEYS = ("chunk_threshold", "chunk_fallback", "chunk_max")
DOC_KEYS = ("doc_threshold", "doc_fallback", "doc_max")


def load_chunk_selector_config(path):
    """Read the P2 ``chunk_selector`` section without modifying it."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    section = payload.get("chunk_selector") if isinstance(payload, dict) else None
    if not isinstance(section, dict) or any(key not in section for key in CHUNK_KEYS):
        raise ValueError(f"Invalid chunk selector YAML: {path}")
    return {key: section[key] for key in CHUNK_KEYS}


def build_predictions(scored_records, chunk_config, doc_config, *,
                      internal_to_official=None, chunk_to_doc=None):
    """Return ``[{id, relevant_docs, relevant_chunks}]`` for every scored query.

    Chunk and document branches are computed independently: the chunk selector
    parameters never influence the document list and vice versa.
    """
    missing_chunk = [key for key in CHUNK_KEYS if key not in chunk_config]
    missing_doc = [key for key in DOC_KEYS if key not in doc_config]
    if missing_chunk or missing_doc:
        raise ValueError(f"Incomplete selector config: chunk={missing_chunk}, doc={missing_doc}")
    prepared = prepare_official_candidates(scored_records, internal_to_official, chunk_to_doc)
    method = doc_config.get("doc_aggregation", "max")
    top_k = doc_config.get("doc_top_k", 3)
    weight = doc_config.get("doc_weight", 0.5)
    predictions = []
    for row in scored_records:
        candidates = prepared[row["id"]]
        documents = aggregate_documents(candidates, method, k=top_k, weight=weight)
        predictions.append({
            "id": row["id"],
            "relevant_chunks": select_ids(candidates, "chunk_id", chunk_config["chunk_threshold"],
                                          chunk_config["chunk_fallback"], chunk_config["chunk_max"]),
            "relevant_docs": select_ids(documents, "doc_id", doc_config["doc_threshold"],
                                        doc_config["doc_fallback"], doc_config["doc_max"]),
        })
    return predictions
