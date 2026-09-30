"""P3-04: independent document-selector tuning (threshold/fallback/max) on macro F2_doc.

The document branch is tuned independently of the P2 chunk selector: only
``doc_threshold``/``doc_fallback``/``doc_max`` are searched, over the same grid
shape P2 uses for chunks. P1/P2 modules are imported read-only.
"""
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.fbeta import classification_metrics
from src.p3.doc_aggregation import (
    aggregate_documents,
    prepare_official_candidates,
    validate_query_coverage,
)
from src.scoring.chunk_selector import select_ids

DEFAULT_FALLBACKS = (0, 1, 2, 3, 5)
DEFAULT_MAXIMUMS = (3, 5, 10, 15, 20)
SWEEP_COLUMNS = ("doc_threshold", "doc_fallback", "doc_max", "macro_precision", "macro_recall",
                 "macro_f1", "macro_f2", "avg_docs_per_query")


def default_doc_thresholds(doc_scores, steps=20):
    """Mirror the P2 chunk-selector grid: ``None`` plus marks across the score range."""
    scores = [float(score) for score in doc_scores]
    if not scores:
        return [None, 0.0, 0.5]
    low, high = min(scores), max(scores)
    if low == high:
        return [None, low]
    step = (high - low) / float(steps)
    return [None] + [round(low + index * step, 4) for index in range(1, steps)]


def _threshold_grid(all_scores, thresholds, steps, min_th, max_th):
    if thresholds is not None:
        return list(thresholds)
    if min_th is None and max_th is None:
        return default_doc_thresholds(all_scores, steps)
    low = min(all_scores) if min_th is None else float(min_th)
    high = max(all_scores) if max_th is None else float(max_th)
    if low == high:
        return [None, low]
    step = (high - low) / float(steps)
    return [None] + [round(low + index * step, 4) for index in range(1, steps)]


def sweep_doc_selector(scored_records, labels, *, aggregation="max", k=3, weight=0.5,
                       thresholds=None, fallbacks=None, maximums=None, steps=20,
                       min_th=None, max_th=None, internal_to_official=None, chunk_to_doc=None):
    """Grid-search doc_threshold x doc_fallback x doc_max maximizing Macro F2_doc.

    Returns ``(best_config, sweep_df)``. The chunk selector parameters are never
    read or reused, so a chunk threshold cannot leak into the document branch.
    """
    validate_query_coverage(scored_records, labels)
    truth = {row["id"]: set(row.get("relevant_docs", [])) for row in labels}
    if not truth:
        raise ValueError("Cannot tune the doc selector with empty labels")
    prepared = prepare_official_candidates(scored_records, internal_to_official, chunk_to_doc)
    docs_per_query = {
        query_id: aggregate_documents(candidates, aggregation, k=k, weight=weight)
        for query_id, candidates in prepared.items()
    }
    all_scores = [doc["score"] for docs in docs_per_query.values() for doc in docs]
    threshold_grid = _threshold_grid(all_scores, thresholds, steps, min_th, max_th)
    fallback_grid = list(fallbacks) if fallbacks else list(DEFAULT_FALLBACKS)
    maximum_grid = list(maximums) if maximums else list(DEFAULT_MAXIMUMS)
    records, query_count = [], len(truth)
    for threshold in threshold_grid:
        for fallback in fallback_grid:
            for maximum in maximum_grid:
                if fallback > maximum:
                    continue
                totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}
                selected_docs = 0
                for query_id, truth_docs in truth.items():
                    selected = select_ids(docs_per_query.get(query_id, []), "doc_id",
                                          threshold, fallback, maximum)
                    selected_docs += len(selected)
                    metrics = classification_metrics(truth_docs, set(selected), zero_division=0.0)
                    for key in totals:
                        totals[key] += metrics[key]
                records.append({
                    "doc_threshold": threshold, "doc_fallback": fallback, "doc_max": maximum,
                    "macro_precision": totals["precision"] / query_count,
                    "macro_recall": totals["recall"] / query_count,
                    "macro_f1": totals["f1"] / query_count,
                    "macro_f2": totals["f2"] / query_count,
                    "avg_docs_per_query": selected_docs / query_count,
                })
    sweep_df = pd.DataFrame(records, columns=list(SWEEP_COLUMNS))
    best = select_best_doc_config(sweep_df, aggregation=aggregation, k=k, weight=weight)
    return best, sweep_df


def select_best_doc_config(sweep_df, *, aggregation="max", k=3, weight=0.5):
    """Choose the best document-selector row deterministically."""
    if sweep_df.empty:
        raise ValueError("Empty doc selector sweep")
    ordered = sweep_df.sort_values(
        by=["macro_f2", "macro_f1", "macro_precision", "avg_docs_per_query", "doc_threshold"],
        ascending=[False, False, False, True, True],
        na_position="first",
    ).reset_index(drop=True)
    row = ordered.iloc[0]
    threshold = row["doc_threshold"]
    return {
        "doc_aggregation": aggregation,
        "doc_top_k": int(k),
        "doc_weight": float(weight),
        "doc_threshold": None if pd.isna(threshold) else float(threshold),
        "doc_fallback": int(row["doc_fallback"]),
        "doc_max": int(row["doc_max"]),
        "macro_precision": float(row["macro_precision"]),
        "macro_recall": float(row["macro_recall"]),
        "macro_f1": float(row["macro_f1"]),
        "macro_f2": float(row["macro_f2"]),
        "avg_docs_per_query": float(row["avg_docs_per_query"]),
        "evaluation_scope": "tuning_on_supplied_labels",
    }


SELECTOR_KEYS = ("doc_aggregation", "doc_top_k", "doc_weight", "doc_threshold", "doc_fallback", "doc_max")


def doc_selector_config(best_config):
    """Extract the selector parameters in the shape consumed by scoring."""
    missing = [key for key in SELECTOR_KEYS if key not in best_config]
    if missing:
        raise ValueError(f"Doc selector config is missing keys: {missing}")
    return {key: best_config[key] for key in SELECTOR_KEYS}


def write_best_doc_selector(path, best_config, *, baseline=None, ablation_csv=None):
    """Write ``best_doc_selector.yaml`` (exclusive creation, chunk selector untouched)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Doc selector already exists: {path}")
    payload = {
        "doc_selector": doc_selector_config(best_config),
        "metrics": {key: best_config[key] for key in
                    ("macro_precision", "macro_recall", "macro_f1", "macro_f2", "avg_docs_per_query")},
        "baseline_aggregation": baseline,
        "aggregation_ablation_csv": str(ablation_csv) if ablation_csv else None,
        "evaluation_scope": best_config.get("evaluation_scope"),
        "target_stage": "P3-04_doc_selection",
        "independent_of_chunk_selector": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return payload


def load_doc_selector_config(path):
    """Load the ``doc_selector`` section for downstream P3 stages."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("doc_selector"), dict):
        raise ValueError(f"Invalid doc selector YAML: {path}")
    return doc_selector_config(payload["doc_selector"])
