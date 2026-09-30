"""P3-05: quantify and (optionally) reconcile chunk/document selection inconsistencies.

Default behaviour is report-only because the ground-truth hierarchy rule is not
known yet (P3-13). An explicit policy can be applied when a rule is confirmed.
"""
import pandas as pd

CONSISTENCY_COLUMNS = ("query_id", "selected_chunk_count", "selected_doc_count", "unknown_chunk_count",
                       "chunks_without_doc_count", "docs_without_chunk_count",
                       "chunks_without_doc", "docs_without_chunk", "consistent")
POLICIES = ("none", "prune_chunks", "add_parent_docs", "prune_docs")


def _unique(values):
    return list(dict.fromkeys(values))


def analyze_consistency(predictions, chunk_to_doc):
    """One row per query with every doc/chunk inconsistency quantified."""
    if not isinstance(chunk_to_doc, dict) or not chunk_to_doc:
        raise ValueError("A non-empty chunk_to_doc mapping is required")
    rows = []
    for row in predictions:
        query_id = row.get("id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"Prediction needs a nonempty id: {row!r}")
        chunks = _unique(row.get("relevant_chunks", []))
        docs = _unique(row.get("relevant_docs", []))
        docs_set = set(docs)
        unknown = [chunk for chunk in chunks if chunk not in chunk_to_doc]
        chunks_without_doc = [chunk for chunk in chunks
                              if chunk in chunk_to_doc and chunk_to_doc[chunk] not in docs_set]
        parents = {chunk_to_doc[chunk] for chunk in chunks if chunk in chunk_to_doc}
        docs_without_chunk = [doc_id for doc_id in docs if doc_id not in parents]
        rows.append({
            "query_id": query_id,
            "selected_chunk_count": len(chunks),
            "selected_doc_count": len(docs),
            "unknown_chunk_count": len(unknown),
            "chunks_without_doc_count": len(chunks_without_doc),
            "docs_without_chunk_count": len(docs_without_chunk),
            "chunks_without_doc": chunks_without_doc,
            "docs_without_chunk": docs_without_chunk,
            "consistent": not unknown and not chunks_without_doc and not docs_without_chunk,
        })
    return pd.DataFrame(rows, columns=list(CONSISTENCY_COLUMNS))


def summarize_consistency(report_df):
    """Aggregate every inconsistency count over all queries."""
    if report_df.empty:
        raise ValueError("Empty consistency report")
    return {
        "query_count": int(len(report_df)),
        "queries_consistent": int(report_df["consistent"].sum()),
        "queries_inconsistent": int((~report_df["consistent"]).sum()),
        "total_unknown_chunks": int(report_df["unknown_chunk_count"].sum()),
        "total_chunks_without_doc": int(report_df["chunks_without_doc_count"].sum()),
        "total_docs_without_chunk": int(report_df["docs_without_chunk_count"].sum()),
    }


def apply_policy(predictions, chunk_to_doc, policy="none"):
    """Apply a configured hierarchy policy; ``none`` returns the input unchanged."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown consistency policy: {policy}")
    if policy == "none":
        return [{"id": row["id"], "relevant_docs": _unique(row.get("relevant_docs", [])),
                 "relevant_chunks": _unique(row.get("relevant_chunks", []))} for row in predictions]
    result = []
    for row in predictions:
        chunks = _unique(row.get("relevant_chunks", []))
        docs = _unique(row.get("relevant_docs", []))
        unknown = [chunk for chunk in chunks if chunk not in chunk_to_doc]
        if unknown:
            raise ValueError(f"Cannot apply policy {policy}: unmapped chunks {unknown}")
        if policy == "prune_chunks":
            docs_set = set(docs)
            chunks = [chunk for chunk in chunks if chunk_to_doc[chunk] in docs_set]
        elif policy == "add_parent_docs":
            docs = _unique([*docs, *(chunk_to_doc[chunk] for chunk in chunks)])
        else:  # prune_docs
            parents = {chunk_to_doc[chunk] for chunk in chunks}
            docs = [doc_id for doc_id in docs if doc_id in parents]
        result.append({"id": row["id"], "relevant_docs": docs, "relevant_chunks": chunks})
    return result
