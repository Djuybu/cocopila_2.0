"""Threshold -> minimum fallback -> maximum count, for both score branches."""
from src.scoring.doc_aggregation import DocumentAggregator, candidate_score


def select_ids(rows, id_key, threshold, fallback, maximum):
    if fallback < 0 or maximum < 0 or fallback > maximum:
        raise ValueError("Require 0 <= fallback <= maximum")
    ordered, seen = [], set()
    for row in sorted(rows, key=lambda item: (-candidate_score(item), item[id_key])):
        if row[id_key] not in seen:
            ordered.append(row)
            seen.add(row[id_key])
    selected = [row for row in ordered if threshold is None or candidate_score(row) >= threshold]
    if len(selected) < fallback:
        selected = ordered[:fallback]
    return [row[id_key] for row in selected[:maximum]]


def select_results(query_id, chunks, config):
    aggregator = DocumentAggregator(
        config.get("doc_aggregation", "max"), config["doc_max"],
        k=config.get("doc_top_k", 3), weight=config.get("doc_weight", 0.5))
    # Both branches see all scored chunks; chunk thresholds do not prune docs.
    documents = aggregator.score_documents(chunks)
    return {
        "id": query_id,
        "relevant_chunks": select_ids(chunks, "chunk_id", config["chunk_threshold"], config["chunk_fallback"], config["chunk_max"]),
        "relevant_docs": select_ids(documents, "doc_id", config["doc_threshold"], config["doc_fallback"], config["doc_max"]),
    }
