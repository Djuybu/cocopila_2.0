"""P2-06: Training data generator for reranker fine-tuning.

Converts queries, corpus chunks, and labels into high-quality positive and negative pairs
(reranker_train.jsonl) while strictly enforcing document-disjoint split boundaries and zero leakage.
"""
from pathlib import Path
import random

from src.utils.io import write_jsonl


def generate_reranker_pairs(
    queries,
    chunks,
    labels,
    split_info=None,
    target_split="train",
    neg_ratio=5,
    seed=42,
    include_metadata=True,
):
    """Generate labeled query-chunk pairs for cross-encoder reranker training.

    Parameters:
        queries: List of query dicts (id, text, optional split, optional negative_chunk_ids).
        chunks: List of chunk dicts (chunk_id, doc_id, text, optional metadata).
        labels: List of label dicts (id, relevant_chunks, optional relevant_docs).
        split_info: Optional dictionary mapping document/query IDs to splits (from split_report.json).
        target_split: Split name to filter for ('train', 'val', or None for all).
        neg_ratio: Number of negative chunks per positive chunk (default: 5).
        seed: Random seed for sampling negatives.
        include_metadata: Whether to preserve chunk metadata in output pairs.

    Returns:
        list of dicts: Labeled pairs with fields:
            query_id, chunk_id, doc_id, query, text, label (1.0/0.0), split, metadata.
    """
    rng = random.Random(seed)
    if type(neg_ratio) is not int or neg_ratio < 0:
        raise ValueError("neg_ratio must be a nonnegative integer")

    chunk_map = {row["chunk_id"]: row for row in chunks}
    label_map = {row["id"]: row for row in labels}

    # Split filtering if available
    query_split_map = {}
    doc_split_map = {}
    if split_info:
        query_split_map = dict(split_info.get("queries", {}))
        doc_split_map = dict(split_info.get("documents", {}))

    def assign(mapping, key, split):
        if split is None:
            return
        if key in mapping and mapping[key] != split:
            raise ValueError(f"Conflicting split assignments for {key}")
        mapping[key] = split

    for c in chunks:
        assign(doc_split_map, c["doc_id"], c.get("split"))
    # When only queries have split tags, infer document membership from positives.
    for q in queries:
        qid = q.get("id", q.get("query_id"))
        assign(query_split_map, qid, q.get("split"))
        for cid in label_map.get(qid, {}).get("relevant_chunks", []):
            if cid in chunk_map:
                assign(doc_split_map, chunk_map[cid]["doc_id"], query_split_map.get(qid))
    if target_split is not None and not query_split_map:
        raise ValueError("Split assignments required; supply split_report.json or explicitly use target_split=None")

    active_queries = []
    for q in queries:
        qid = q.get("id", q.get("query_id"))
        q_split = query_split_map.get(qid)
        if target_split is not None and q_split is None:
            raise ValueError(f"Missing split assignment for query {qid}")
        if target_split is not None and q_split != target_split:
            continue
        active_queries.append(q)

    # Determine corpus pool of candidate negatives within the target split
    if target_split is not None:
        pool_chunk_ids = [
            c["chunk_id"] for c in chunks
            if doc_split_map.get(c["doc_id"]) == target_split
        ]
    else:
        pool_chunk_ids = [c["chunk_id"] for c in chunks]

    pairs = []
    for q in active_queries:
        qid = q.get("id", q.get("query_id"))
        qtext = q.get("text", q.get("query", ""))
        lbl = label_map.get(qid, {})
        pos_chunk_ids = [cid for cid in lbl.get("relevant_chunks", []) if cid in chunk_map]

        if not pos_chunk_ids:
            continue

        pos_set = set(pos_chunk_ids)
        pool_set = set(pool_chunk_ids)
        if target_split is not None and not pos_set <= pool_set:
            raise ValueError(f"Positive chunks cross split boundary for query {qid}")
        # 1. Create Positive Pairs
        for cid in pos_chunk_ids:
            c = chunk_map[cid]
            pair = {
                "query_id": qid,
                "chunk_id": cid,
                "doc_id": c.get("doc_id"),
                "query": qtext,
                "text": c.get("text", ""),
                "label": 1.0,
                "split": target_split or "all",
            }
            if include_metadata:
                pair["metadata"] = c.get("metadata", {})
            pairs.append(pair)

        # 2. Select Negatives
        # Check if query already has pre-identified negative chunks
        explicit_neg_ids = [
            cid for cid in q.get("negative_chunk_ids", [])
            if cid in pool_set and cid not in pos_set
        ]

        needed_negs = len(pos_chunk_ids) * neg_ratio
        selected_neg_ids = []

        if explicit_neg_ids:
            shuffled_explicit = list(explicit_neg_ids)
            rng.shuffle(shuffled_explicit)
            selected_neg_ids.extend(shuffled_explicit[:needed_negs])

        # If more negatives needed, sample from pool chunks (excluding positive docs if known)
        pos_docs = set(c.get("doc_id") for c in [chunk_map[c_id] for c_id in pos_chunk_ids])
        remaining = needed_negs - len(selected_neg_ids)
        if remaining > 0:
            candidate_pool = [
                cid for cid in pool_chunk_ids
                if cid not in pos_set and chunk_map[cid].get("doc_id") not in pos_docs and cid not in selected_neg_ids
            ]
            if candidate_pool:
                sampled = rng.sample(candidate_pool, min(remaining, len(candidate_pool)))
                selected_neg_ids.extend(sampled)

        # 3. Create Negative Pairs
        for cid in selected_neg_ids:
            c = chunk_map[cid]
            pair = {
                "query_id": qid,
                "chunk_id": cid,
                "doc_id": c.get("doc_id"),
                "query": qtext,
                "text": c.get("text", ""),
                "label": 0.0,
                "split": target_split or "all",
            }
            if include_metadata:
                pair["metadata"] = c.get("metadata", {})
            pairs.append(pair)

    return pairs


def validate_no_leakage(train_pairs, val_pairs, split_info=None):
    """Validate that there is zero data leakage between train and val pairs.

    Checks:
    - Query ID leakage: Train and val sets must have completely disjoint query IDs.
    - Document leakage: No document in train may appear in val, including negatives.

    Raises:
        ValueError: If any leakage is detected.
    """
    train_qids = {p["query_id"] for p in train_pairs}
    val_qids = {p["query_id"] for p in val_pairs}
    q_overlap = train_qids & val_qids
    if q_overlap:
        raise ValueError(f"Query ID leakage detected between train and val: {sorted(list(q_overlap))[:5]}")

    # Every document must stay in one split, regardless of its pair label.
    train_docs = {p["doc_id"] for p in train_pairs if p.get("doc_id")}
    val_docs = {p["doc_id"] for p in val_pairs if p.get("doc_id")}
    doc_overlap = train_docs & val_docs
    if doc_overlap:
        raise ValueError(f"Document leakage detected between train and val: {sorted(list(doc_overlap))[:5]}")
    if split_info:
        for expected, pairs in (("train", train_pairs), ("val", val_pairs)):
            for pair in pairs:
                for field, assignments in (("query_id", split_info.get("queries", {})),
                                           ("doc_id", split_info.get("documents", {}))):
                    actual = assignments.get(pair.get(field))
                    if actual is not None and actual != expected:
                        raise ValueError(f"Split leakage detected: {pair.get(field)} belongs to {actual}, not {expected}")

    return {
        "status": "clean",
        "train_queries": len(train_qids),
        "val_queries": len(val_qids),
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "train_pos_docs": len({p["doc_id"] for p in train_pairs if p.get("label") == 1.0 and p.get("doc_id")}),
        "val_pos_docs": len({p["doc_id"] for p in val_pairs if p.get("label") == 1.0 and p.get("doc_id")}),
        "leakage_count": 0,
    }


def report_data_statistics(pairs):
    """Compute comprehensive dataset statistics for reranker training pairs.

    Returns:
        dict: Detailed statistics including pair counts, pos/neg ratio, length metrics.
    """
    total = len(pairs)
    if total == 0:
        return {
            "total_pairs": 0,
            "num_queries": 0,
            "num_positives": 0,
            "num_negatives": 0,
            "pos_neg_ratio": "0:0",
        }

    qids = set()
    pos_count = 0
    neg_count = 0
    query_lens_char = []
    query_lens_words = []
    chunk_lens_char = []
    chunk_lens_words = []

    for p in pairs:
        qids.add(p["query_id"])
        if p["label"] == 1.0:
            pos_count += 1
        else:
            neg_count += 1

        q_words = len(p["query"].split())
        c_words = len(p["text"].split())
        query_lens_words.append(q_words)
        chunk_lens_words.append(c_words)
        query_lens_char.append(len(p["query"]))
        chunk_lens_char.append(len(p["text"]))

    pos_neg_ratio = f"1:{neg_count / max(1, pos_count):.2f}"

    return {
        "total_pairs": total,
        "num_queries": len(qids),
        "num_positives": pos_count,
        "num_negatives": neg_count,
        "pos_neg_ratio": pos_neg_ratio,
        "negative_per_positive": round(neg_count / max(1, pos_count), 2),
        "avg_query_words": round(sum(query_lens_words) / total, 1),
        "avg_chunk_words": round(sum(chunk_lens_words) / total, 1),
        "avg_query_chars": round(sum(query_lens_char) / total, 1),
        "avg_chunk_chars": round(sum(chunk_lens_char) / total, 1),
    }


def write_training_data(pairs, output_path):
    """Save training pairs to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, pairs)
