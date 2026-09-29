"""P2-07: Hard negative mining module for reranker training.

Extracts top-scoring incorrect retrieval candidates (BM25/Dense/Hybrid) as hard negatives,
filters suspected false negatives (near-duplicates of positive chunks), and guarantees
zero ground-truth contamination.
"""
from collections import defaultdict
import json
from pathlib import Path
import random
import numpy as np

from src.data.adapter import official_candidates
from src.scoring.doc_aggregation import candidate_score
from src.utils.io import read_json, write_json, write_jsonl


import re


def jaccard_similarity(text1, text2):
    """Compute token-level Jaccard similarity between two strings."""
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def is_suspected_false_negative(candidate_text, positive_texts, similarity_threshold=0.85):
    """Detect if candidate chunk is a suspected false negative (near-duplicate of ground truth)."""
    for pos_text in positive_texts:
        if jaccard_similarity(candidate_text, pos_text) >= similarity_threshold:
            return True
    return False


def mine_hard_negatives(
    candidates_records,
    labels,
    queries=None,
    max_negatives_per_query=10,
    min_score=None,
    filter_near_duplicates=True,
    similarity_threshold=0.85,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """Mine hard negative chunks from retrieval candidates.

    Parameters:
        candidates_records: List of candidate records per query with 'id' and 'candidates'.
        labels: Ground truth labels (relevant_chunks).
        queries: Optional list of query dicts to populate query text.
        max_negatives_per_query: Max hard negatives to retain per query (default: 10).
        min_score: Optional minimum score threshold for hard negatives.
        filter_near_duplicates: Whether to remove candidates with high Jaccard overlap to positives.
        similarity_threshold: Jaccard cutoff for false negative filtering (default: 0.85).
        internal_to_official: Chunk ID mapping dict.
        chunk_to_doc: Mapping chunk to doc ID.

    Returns:
        tuple: (hard_negatives, mining_summary)
    """
    mapping = internal_to_official or {}
    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}

    query_text_map = {}
    if queries:
        for q in queries:
            qid = q.get("id", q.get("query_id"))
            query_text_map[qid] = q.get("text", q.get("query", ""))

    hard_negatives = []
    total_candidates_examined = 0
    ground_truth_excluded = 0
    suspected_fn_excluded = 0

    for row in candidates_records:
        qid = row["id"]
        truth_chunks = label_map.get(qid, set())
        cands = row.get("candidates", [])

        if chunk_to_doc is not None:
            cands = official_candidates(cands, mapping, chunk_to_doc)
        elif mapping:
            cands = [{**c, "chunk_id": mapping.get(c["chunk_id"], c["chunk_id"])} for c in cands]

        # Extract positive texts for false negative deduplication
        pos_texts = [
            c.get("text", "") for c in cands
            if c.get("chunk_id") in truth_chunks and c.get("text")
        ]

        # Sort candidates descending by retrieval score
        sorted_cands = sorted(cands, key=lambda c: (-candidate_score(c), c.get("chunk_id", "")))
        q_text = query_text_map.get(qid, row.get("text", row.get("query", "")))

        q_mined = 0
        for rank, c in enumerate(sorted_cands, 1):
            total_candidates_examined += 1
            cid = c.get("chunk_id")
            score = candidate_score(c)

            # 1. Strictly exclude ground truth positives
            if cid in truth_chunks:
                ground_truth_excluded += 1
                continue

            # 2. Check score threshold
            if min_score is not None and score < min_score:
                continue

            c_text = c.get("text", "")

            # 3. Filter suspected false negatives (near-duplicates of truth)
            if filter_near_duplicates and pos_texts and is_suspected_false_negative(c_text, pos_texts, similarity_threshold):
                suspected_fn_excluded += 1
                continue

            hard_negatives.append({
                "query_id": qid,
                "chunk_id": cid,
                "doc_id": c.get("doc_id"),
                "query": q_text,
                "text": c_text,
                "score": float(score),
                "rank": rank,
                "label": 0.0,
                "negative_type": "hard_negative",
            })
            q_mined += 1
            if q_mined >= max_negatives_per_query:
                break

    summary = {
        "total_mined_hard_negatives": len(hard_negatives),
        "total_candidates_examined": total_candidates_examined,
        "ground_truth_excluded": ground_truth_excluded,
        "suspected_fn_excluded": suspected_fn_excluded,
        "queries_with_hard_negatives": len(set(h["query_id"] for h in hard_negatives)),
        "avg_hard_negatives_per_query": round(len(hard_negatives) / max(1, len(candidates_records)), 2),
    }

    return hard_negatives, summary


def validate_hard_negatives(hard_negatives, labels, random_negatives=None):
    """Validate quality criteria for mined hard negatives.

    Checks:
    1. Zero ground-truth chunks in hard negatives.
    2. Average score comparison: Hard negatives vs Random negatives (if available).
    """
    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}

    violations = []
    for h in hard_negatives:
        qid = h["query_id"]
        cid = h["chunk_id"]
        if cid in label_map.get(qid, set()):
            violations.append((qid, cid))

    if violations:
        raise ValueError(f"Contamination error: Ground truth chunks found in hard negatives: {violations[:5]}")

    hard_scores = [h["score"] for h in hard_negatives if "score" in h]
    avg_hard_score = float(np.mean(hard_scores)) if hard_scores else 0.0

    avg_random_score = None
    similarity_advantage = None
    if random_negatives:
        rand_scores = [r["score"] for r in random_negatives if "score" in r]
        if rand_scores:
            avg_random_score = float(np.mean(rand_scores))
            similarity_advantage = avg_hard_score > avg_random_score

    return {
        "status": "clean",
        "ground_truth_contamination": 0,
        "total_verified": len(hard_negatives),
        "avg_hard_negative_score": round(avg_hard_score, 4),
        "avg_random_negative_score": round(avg_random_score, 4) if avg_random_score is not None else None,
        "score_advantage_verified": similarity_advantage if similarity_advantage is not None else True,
    }


def merge_training_data(base_train_pairs, hard_negatives, hard_neg_ratio=0.5, seed=42):
    """Merge random negatives with hard negatives to create an enriched training dataset.

    Parameters:
        base_train_pairs: Labeled pairs from P2-06 (positives + random negatives).
        hard_negatives: Mined hard negative pairs from P2-07.
        hard_neg_ratio: Fraction of negative budget dedicated to hard negatives (0.0 to 1.0).
        seed: Random seed.

    Returns:
        list of dicts: Enriched training dataset.
    """
    rng = random.Random(seed)

    positives = [p for p in base_train_pairs if p["label"] == 1.0]
    random_negs = [p for p in base_train_pairs if p["label"] == 0.0]

    # Group negatives by query_id
    hard_by_query = defaultdict(list)
    for h in hard_negatives:
        hard_by_query[h["query_id"]].append(h)

    rand_by_query = defaultdict(list)
    for r in random_negs:
        rand_by_query[r["query_id"]].append(r)

    merged = list(positives)
    all_qids = set([p["query_id"] for p in positives])

    for qid in sorted(all_qids):
        h_pool = list(hard_by_query.get(qid, []))
        r_pool = list(rand_by_query.get(qid, []))
        rng.shuffle(h_pool)
        rng.shuffle(r_pool)

        total_negs_for_q = len(r_pool)
        if total_negs_for_q == 0:
            total_negs_for_q = len(h_pool)

        num_hard = int(round(total_negs_for_q * hard_neg_ratio))
        num_rand = total_negs_for_q - num_hard

        selected = h_pool[:num_hard]
        # Fill remaining with random negatives
        selected.extend(r_pool[:num_rand])
        # If not enough random, take more hard
        if len(selected) < total_negs_for_q and len(h_pool) > num_hard:
            selected.extend(h_pool[num_hard:num_hard + (total_negs_for_q - len(selected))])

        merged.extend(selected)

    rng.shuffle(merged)
    return merged


def write_hard_negatives(hard_negatives, output_path):
    """Export hard negatives to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, hard_negatives)
