"""Candidate recall before reranking, at a shared chunk budget for both branches."""
from src.data.schema import unique_ids


def evaluate_candidate_recall(records, labels, ks=(20, 50, 100, 200), internal_to_official=None):
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("Cutoffs must be positive integers")
    ks = sorted(set(ks))
    if unique_ids(records, "id") != unique_ids(labels, "id"):
        raise ValueError("Candidates and labels must cover exactly the same query IDs")
    truth = {row["id"]: row for row in labels}
    mapping = internal_to_official or {}
    per_query, misses_chunks, misses_docs = {}, [], []
    for record in records:
        target = truth[record["id"]]
        gt_chunks, gt_docs = set(target["relevant_chunks"]), set(target["relevant_docs"])
        ordered, seen = [], set()
        for candidate in record["candidates"]:
            chunk_id = mapping.get(candidate["chunk_id"], candidate["chunk_id"])
            if chunk_id not in seen:
                ordered.append({**candidate, "chunk_id": chunk_id})
                seen.add(chunk_id)
        result = {"candidate_count": len(ordered), "chunks": {}, "documents": {}}
        for k in ks:
            prefix = ordered[:k]
            found_chunks = {row["chunk_id"] for row in prefix}
            found_docs = {row["doc_id"] for row in prefix}
            result["chunks"][f"recall@{k}"] = len(found_chunks & gt_chunks) / len(gt_chunks) if gt_chunks else 0.0
            result["documents"][f"recall@{k}"] = len(found_docs & gt_docs) / len(gt_docs) if gt_docs else 0.0
        top_chunks = {row["chunk_id"] for row in ordered[:max(ks)]}
        top_docs = {row["doc_id"] for row in ordered[:max(ks)]}
        result["missing_chunks_at_max_k"] = sorted(gt_chunks - top_chunks)
        result["missing_docs_at_max_k"] = sorted(gt_docs - top_docs)
        result["miss_all_chunks"] = bool(gt_chunks) and not bool(top_chunks & gt_chunks)
        result["miss_all_docs"] = bool(gt_docs) and not bool(top_docs & gt_docs)
        if result["miss_all_chunks"]:
            misses_chunks.append(record["id"])
        if result["miss_all_docs"]:
            misses_docs.append(record["id"])
        per_query[record["id"]] = result
    count = len(per_query)
    macro = {branch: {f"recall@{k}": sum(row[branch][f"recall@{k}"] for row in per_query.values()) / count if count else 0.0
                      for k in ks} for branch in ("chunks", "documents")}
    return {"stage": "before_reranking", "aggregation": "macro_by_query", "cutoffs": ks,
            "document_budget": "parents_of_first_k_unique_official_chunks", "query_count": count,
            "macro": macro, "per_query": per_query,
            "miss_all_chunk_queries": misses_chunks, "miss_all_doc_queries": misses_docs}
