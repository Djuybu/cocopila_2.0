"""Document-disjoint split checks; no random split is silently introduced."""
from collections import defaultdict
import hashlib
import math
import random


def group_split(documents, chunks, queries, labels, *, ratios, seed):
    """Keep shared-positive queries, titles and duplicate texts in one component."""
    from src.data.prototype import text_key
    from src.data.schema import unique_ids
    if not ratios or any(value < 0 or not math.isfinite(value) for value in ratios.values()) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Split ratios must be finite, nonnegative and sum to one")
    doc_ids = unique_ids(documents, "doc_id")
    query_ids = unique_ids(queries, "id")
    if unique_ids(labels, "id") != query_ids:
        raise ValueError("Labels must cover every query")
    parent = {doc_id: doc_id for doc_id in doc_ids}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def connect(keys):
        keys = sorted(set(keys))
        if any(key not in doc_ids for key in keys):
            raise ValueError("Unknown document in split labels")
        for key in keys[1:]:
            left, right = root(keys[0]), root(key)
            parent[max(left, right)] = min(left, right)

    titles, content = defaultdict(list), defaultdict(list)
    for row in documents:
        if row.get("title") and not row.get("metadata", {}).get("title_is_source_filename", False):
            titles[text_key(row["title"])].append(row["doc_id"])
    for row in chunks:
        content[text_key(row["text"])].append(row["doc_id"])
    for keys in [*titles.values(), *content.values()]:
        connect(keys)
    chunk_parent = {row["chunk_id"]: row["doc_id"] for row in chunks}
    query_texts = {row["id"]: text_key(row["text"]) for row in queries}
    shared_queries = defaultdict(list)
    for row in labels:
        if not row["relevant_docs"] or not row["relevant_chunks"]:
            raise ValueError("Prototype split requires positive labels")
        derived = {chunk_parent[c] for c in row["relevant_chunks"]}
        if derived != set(row["relevant_docs"]):
            raise ValueError("Chunk/doc labels disagree")
        connect(row["relevant_docs"])
        shared_queries[query_texts[row["id"]]].extend(row["relevant_docs"])
    for keys in shared_queries.values():
        connect(keys)
    groups = defaultdict(list)
    for doc_id in sorted(doc_ids):
        groups[root(doc_id)].append(doc_id)
    labelled_roots = {root(row["relevant_docs"][0]) for row in labels}
    ordered = sorted(labelled_roots)
    active = sorted(name for name, ratio in ratios.items() if ratio > 0)
    if len(ordered) < len(active):
        raise ValueError("Too few independent document groups for nonempty labelled splits")
    rng = random.Random(seed)
    rng.shuffle(ordered)
    targets = {name: len(ordered) * ratios[name] for name in active}
    counts = {name: max(1, math.floor(targets[name])) for name in active}
    while sum(counts.values()) > len(ordered):
        name = max((n for n in active if counts[n] > 1), key=lambda n: (counts[n] - targets[n], n))
        counts[name] -= 1
    while sum(counts.values()) < len(ordered):
        name = max(active, key=lambda n: (targets[n] - counts[n], n))
        counts[name] += 1
    assigned, offset = {}, 0
    for name in active:
        for key in ordered[offset:offset + counts[name]]:
            assigned[key] = name
        offset += counts[name]
    # Corpus-only negative documents still belong to exactly one split.
    for key in sorted(set(groups) - labelled_roots):
        value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
        cumulative = 0
        for name in sorted(ratios):
            cumulative += ratios[name]
            if value < cumulative:
                assigned[key] = name
                break
    document_splits = {key: assigned[root(key)] for key in sorted(doc_ids)}
    query_splits = {row["id"]: assigned[root(row["relevant_docs"][0])] for row in labels}
    return {"seed": seed, "ratios": ratios, "documents": document_splits, "queries": query_splits,
            "group_count": len(groups), "labelled_group_count": len(labelled_roots),
            "counts": {name: {"documents": sum(v == name for v in document_splits.values()),
                              "queries": sum(v == name for v in query_splits.values())}
                       for name in ratios}}


def validate_split_leakage(splits):
    owners = {}
    for split, rows in splits.items():
        for row in rows:
            doc_id = row["doc_id"]
            if doc_id in owners and owners[doc_id] != split:
                raise ValueError(f"Document leakage: {doc_id} in {owners[doc_id]} and {split}")
            owners[doc_id] = split
