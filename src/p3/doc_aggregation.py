"""P3-02/P3-03: document aggregation baseline and ablation.

Consumes scored chunks (P2 ``reranked.jsonl``) and ground-truth labels. The
baseline is max-pooling over the chunk score (P3-02); the ablation compares it
with top-k mean and a weighted max+mean (P3-03). P1/P2 modules are imported
read-only: aggregation reuses ``src.scoring.doc_aggregation.DocumentAggregator``
and selection semantics reuse ``src.scoring.chunk_selector.select_ids``.
"""
import pandas as pd

from src.data.adapter import official_candidates
from src.evaluation.fbeta import classification_metrics
from src.scoring.chunk_selector import select_ids
from src.scoring.doc_aggregation import DocumentAggregator

AGGREGATION_METHODS = (
    {"name": "max", "method": "max", "k": 3, "weight": 0.5},
    {"name": "mean_top2", "method": "mean_top_k", "k": 2, "weight": 0.5},
    {"name": "mean_top3", "method": "mean_top_k", "k": 3, "weight": 0.5},
    {"name": "weighted_max_mean_top3", "method": "weighted", "k": 3, "weight": 0.5},
)
ABLATION_COLUMNS = ("method", "macro_precision", "macro_recall", "macro_f1", "macro_f2",
                    "avg_docs_per_query", "zero_doc_queries")


def aggregate_documents(chunks, method="max", *, k=3, weight=0.5):
    """Score documents from scored chunks; deterministic and aggregation-method aware."""
    return DocumentAggregator(method, k=k, weight=weight).score_documents(chunks)


def prepare_official_candidates(scored_records, internal_to_official=None, chunk_to_doc=None):
    """Collapse internal rechunks to official IDs before aggregation (P1 contract)."""
    mapping = internal_to_official or {}
    prepared = {}
    for row in scored_records:
        candidates = row.get("candidates", [])
        if chunk_to_doc is not None:
            candidates = official_candidates(candidates, mapping, chunk_to_doc)
        elif mapping:
            candidates = [{**c, "chunk_id": mapping.get(c["chunk_id"], c["chunk_id"])} for c in candidates]
        prepared[row["id"]] = candidates
    return prepared


def validate_query_coverage(scored_records, labels):
    query_ids = [row.get("id") for row in scored_records]
    if any(not isinstance(qid, str) or not qid for qid in query_ids):
        raise ValueError("Every scored record needs a nonempty query id")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query id in scored records")
    if set(query_ids) != {row.get("id") for row in labels}:
        raise ValueError("Scored candidates and labels must cover exactly the same queries")


def _ablation_row(name, docs_per_query, truth, threshold, fallback, maximum):
    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}
    counts, zero_queries = [], 0
    for query_id, truth_docs in truth.items():
        selected = select_ids(docs_per_query.get(query_id, []), "doc_id", threshold, fallback, maximum)
        counts.append(len(selected))
        zero_queries += int(not selected)
        metrics = classification_metrics(truth_docs, set(selected), zero_division=0.0)
        for key in totals:
            totals[key] += metrics[key]
    query_count = len(truth)
    return {
        "method": name,
        "macro_precision": totals["precision"] / query_count,
        "macro_recall": totals["recall"] / query_count,
        "macro_f1": totals["f1"] / query_count,
        "macro_f2": totals["f2"] / query_count,
        "avg_docs_per_query": sum(counts) / query_count,
        "zero_doc_queries": zero_queries,
    }


def evaluate_aggregation(scored_records, labels, internal_to_official=None, chunk_to_doc=None,
                         methods=AGGREGATION_METHODS, doc_threshold=None, doc_fallback=0, doc_max=10):
    """Compare aggregation methods under one fixed document-selection policy.

    Only the aggregation method varies, so the ablation isolates its effect.
    Returns a DataFrame with one row per method and macro F2_doc (P3-03).
    """
    validate_query_coverage(scored_records, labels)
    if doc_fallback < 0 or doc_max < 0 or doc_fallback > doc_max:
        raise ValueError("Require 0 <= doc_fallback <= doc_max")
    truth = {row["id"]: set(row.get("relevant_docs", [])) for row in labels}
    if not truth:
        raise ValueError("Cannot evaluate aggregation with empty labels")
    prepared = prepare_official_candidates(scored_records, internal_to_official, chunk_to_doc)
    rows = []
    for spec in methods:
        docs_per_query = {
            query_id: aggregate_documents(candidates, spec["method"], k=spec["k"], weight=spec["weight"])
            for query_id, candidates in prepared.items()
        }
        rows.append(_ablation_row(spec["name"], docs_per_query, truth, doc_threshold, doc_fallback, doc_max))
    return pd.DataFrame(rows, columns=list(ABLATION_COLUMNS))


def choose_baseline(ablation_df, prefer=("macro_f2", "macro_recall", "macro_precision")):
    """Pick the baseline aggregation deterministically (highest metric, then method name)."""
    if ablation_df.empty:
        raise ValueError("Empty aggregation ablation table")
    ordered = ablation_df.sort_values(
        by=[*prefer, "avg_docs_per_query", "method"],
        ascending=[False] * len(prefer) + [True, True],
    ).reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def aggregation_spec(name):
    """Resolve an ablation method name to its aggregation parameters."""
    for spec in AGGREGATION_METHODS:
        if spec["name"] == name:
            return dict(spec)
    raise ValueError(f"Unknown aggregation method: {name}")
