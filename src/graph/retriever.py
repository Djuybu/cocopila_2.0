"""P3-12: graph retrieval (GraphRetriever) over the P3-11 medical graph.

Parses entities from the query, normalizes them with the lexicon, retrieves chunks
through MENTIONS/HAS_CHUNK (optionally one clinical-relation hop), and returns
candidates in the P1 ``Candidate`` contract so the graph joins
``CandidateGenerator`` as an independent source. Also provides an independent
Recall@K harness and a graph-vs-baseline ablation.
"""
from collections import defaultdict

from src.evaluation.candidate_recall import evaluate_candidate_recall
from src.graph.ner import build_matcher, extract_entities
from src.retrieval.base import BaseRetriever, normalize_candidates

SOURCE = "graph"


def rank_chunks_by_entities(store, entity_ids, limit, expand=False):
    """Rank chunks by matched entities; optional one relation hop adds half weight."""
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    weights, docs = defaultdict(float), {}
    for entity_id in entity_ids:
        for row in store.chunks_for_entity(entity_id):
            weights[row["chunk_id"]] += 1.0
            docs[row["chunk_id"]] = row["doc_id"]
    if expand:
        for entity_id in entity_ids:
            for related in store.related_entities(entity_id):
                for row in store.chunks_for_entity(related):
                    weights[row["chunk_id"]] += 0.5
                    docs.setdefault(row["chunk_id"], row["doc_id"])
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"chunk_id": chunk_id, "doc_id": docs[chunk_id], "score": score}
            for chunk_id, score in ranked]


class GraphRetriever(BaseRetriever):
    """Entity -> chunk retrieval returning unique candidates in rank order."""

    def __init__(self, store, lexicon, *, expand=False):
        self.store, self.lexicon, self.expand = store, lexicon, expand
        self.matcher = build_matcher(lexicon)

    def parse_query_entities(self, query):
        """Normalized entity mentions (canonical_id/type) found in the query."""
        return extract_entities(query, self.matcher)

    def retrieve(self, query, top_k):
        entity_ids = list(dict.fromkeys(span["canonical_id"] for span in self.parse_query_entities(query)))
        if not entity_ids:
            return []
        rows = rank_chunks_by_entities(self.store, entity_ids, top_k, self.expand)
        return normalize_candidates(rows, SOURCE, top_k)


def retrieve_candidate_records(retriever, queries, top_k):
    """``[{"id", "candidates": [...]}]`` in the P1 candidate-record shape."""
    return [{"id": query["id"], "candidates": retriever.retrieve(query["text"], top_k)} for query in queries]


def union_candidate_records(*record_sets, top_k=None):
    """Union candidate records by query id + chunk_id, keeping first occurrence."""
    grouped = {}
    for records in record_sets:
        for record in records:
            grouped.setdefault(record["id"], []).append(record.get("candidates", []))
    merged = []
    for query_id, groups in grouped.items():
        ordered, seen = [], set()
        for rows in groups:
            for row in rows:
                if row["chunk_id"] in seen:
                    continue
                seen.add(row["chunk_id"])
                ordered.append(row)
        limit = top_k if top_k is not None else len(ordered)
        merged.append({"id": query_id, "candidates": normalize_candidates(ordered, "union", limit)})
    return merged


def evaluate_graph_recall(retriever, queries, labels, ks=(20, 50, 100, 200), top_k=None,
                          internal_to_official=None):
    """Independent chunk/document Recall@K for the graph source (P3-12)."""
    top_k = max(ks) if top_k is None else top_k
    records = retrieve_candidate_records(retriever, queries, top_k)
    report = evaluate_candidate_recall(records, labels, ks=ks, internal_to_official=internal_to_official)
    report["source"] = SOURCE
    report["expand_relations"] = bool(retriever.expand)
    return report


def ablation_graph_vs_baseline(retriever, queries, labels, baseline_records=None,
                               ks=(20, 50, 100, 200), top_k=None, internal_to_official=None):
    """Keep/drop ablation: graph alone versus graph unioned with a baseline source."""
    top_k = max(ks) if top_k is None else top_k
    graph_records = retrieve_candidate_records(retriever, queries, top_k)
    result = {"graph_only": evaluate_candidate_recall(graph_records, labels, ks=ks,
                                                      internal_to_official=internal_to_official)}
    if baseline_records:
        combined = union_candidate_records(baseline_records, graph_records, top_k=top_k)
        result["graph_plus_baseline"] = evaluate_candidate_recall(combined, labels, ks=ks,
                                                                  internal_to_official=internal_to_official)
    return result
