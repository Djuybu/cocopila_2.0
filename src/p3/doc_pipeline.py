"""P3-14: official document aggregation pipeline — out-of-fold tuning + ablation.

Tunes document aggregation (max / mean_top_k / weighted) and an optional
``direct_doc`` branch (a separate document retriever fused by weight) with
K-fold query holdout, maximizing Macro F2_doc. P1/P2 modules are reused read-only
(``kfold_query_split``, ``select_ids``, ``classification_metrics``).
"""
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.fbeta import classification_metrics
from src.p3.doc_aggregation import (
    AGGREGATION_METHODS,
    aggregate_documents,
    prepare_official_candidates,
    validate_query_coverage,
)
from src.p3.doc_selector import DEFAULT_FALLBACKS, DEFAULT_MAXIMUMS, default_doc_thresholds
from src.scoring.chunk_selector import select_ids
from src.scoring.cv_threshold import kfold_query_split

DIRECT_WEIGHTS = (0.0, 0.5)
DEFAULT_DIRECT_WEIGHTS = DIRECT_WEIGHTS
ABLATION_COLUMNS = ("aggregation", "direct_doc_weight", "oof_mean_f2", "oof_std_f2",
                    "doc_threshold", "doc_fallback", "doc_max", "full_data_f2")


def _normalise(rows):
    if not rows:
        return {}
    scores = [float(row["score"]) for row in rows]
    low, high = min(scores), max(scores)
    if low == high:
        return {row["doc_id"]: 1.0 for row in rows}
    return {row["doc_id"]: (float(row["score"]) - low) / (high - low) for row in rows}


def direct_docs_from_record(record):
    """Accept ``candidates`` or ``docs`` lists with ``doc_id`` and ``score``."""
    rows = record.get("candidates", record.get("docs", []))
    docs = []
    for row in rows:
        doc_id, score = row.get("doc_id"), row.get("score")
        if not isinstance(doc_id, str) or not doc_id or score is None:
            raise ValueError(f"Direct-doc entry needs doc_id and score: {row!r}")
        docs.append({"doc_id": doc_id, "score": float(score)})
    return docs


def document_scores_for_query(chunk_candidates, direct_docs=None, *, aggregation="max",
                              k=3, weight=0.5, direct_weight=0.0):
    """Aggregate chunks; optionally fuse a direct-doc ranking after min-max scaling."""
    aggregated = aggregate_documents(chunk_candidates, aggregation, k=k, weight=weight)
    if not direct_docs or direct_weight <= 0:
        return [{"doc_id": row["doc_id"], "score": row["score"]} for row in aggregated]
    aggregated_norm, direct_norm = _normalise(aggregated), _normalise(direct_docs)
    doc_ids = sorted(set(aggregated_norm) | set(direct_norm))
    return [{"doc_id": doc_id,
             "score": (1 - direct_weight) * aggregated_norm.get(doc_id, 0.0)
                      + direct_weight * direct_norm.get(doc_id, 0.0)}
            for doc_id in doc_ids]


def build_doc_scores(scored_records, direct_doc_records=None, *, aggregation="max",
                     k=3, weight=0.5, direct_weight=0.0, internal_to_official=None, chunk_to_doc=None):
    """``{query_id: [{doc_id, score}]}`` for one scoring configuration."""
    prepared = prepare_official_candidates(scored_records, internal_to_official, chunk_to_doc)
    direct_index = {row["id"]: direct_docs_from_record(row) for row in (direct_doc_records or [])}
    return {row["id"]: document_scores_for_query(
        prepared[row["id"]], direct_index.get(row["id"]), aggregation=aggregation,
        k=k, weight=weight, direct_weight=direct_weight) for row in scored_records}


def macro_f2_doc(doc_scores, labels_by_id, query_ids, threshold, fallback, maximum):
    """Macro F2_doc over the given queries for one (threshold, fallback, max) config."""
    if not query_ids:
        raise ValueError("At least one query is required")
    total = 0.0
    for query_id in query_ids:
        selected = select_ids(doc_scores.get(query_id, []), "doc_id", threshold, fallback, maximum)
        total += classification_metrics(labels_by_id[query_id], set(selected), zero_division=0.0)["f2"]
    return total / len(query_ids)


def _tune_on_train(doc_scores, labels_by_id, train_ids, thresholds, fallbacks, maximums):
    best, best_key = None, None
    for threshold in thresholds:
        for fallback in fallbacks:
            for maximum in maximums:
                if fallback > maximum:
                    continue
                score = macro_f2_doc(doc_scores, labels_by_id, train_ids, threshold, fallback, maximum)
                key = (-score, threshold is not None,
                       threshold if threshold is not None else -1.0, fallback, maximum)
                if best_key is None or key < best_key:
                    best_key, best = key, {"doc_threshold": threshold, "doc_fallback": fallback,
                                           "doc_max": maximum, "train_f2": score}
    return best


def tune_doc_pipeline(scored_records, labels, *, direct_doc_records=None, aggregations=None,
                      direct_weights=DIRECT_WEIGHTS, thresholds=None, fallbacks=None, maximums=None,
                      n_folds=5, seed=42, internal_to_official=None, chunk_to_doc=None):
    """K-fold query holdout tuning with an aggregation x direct-doc ablation.

    Returns ``(best_pipeline, ablation_df)``. ``best_pipeline`` is tuned on all
    labels for deployment; ``holdout`` reports the out-of-fold score of that
    scoring configuration (the number that was never tuned on validation).
    """
    validate_query_coverage(scored_records, labels)
    labels_by_id = {row["id"]: set(row.get("relevant_docs", [])) for row in labels}
    if not labels_by_id:
        raise ValueError("Cannot tune the doc pipeline with empty labels")
    query_ids = [row["id"] for row in scored_records]
    splits = kfold_query_split(query_ids, n_folds, seed)
    specs = list(aggregations or AGGREGATION_METHODS)
    fallback_grid = list(fallbacks) if fallbacks else list(DEFAULT_FALLBACKS)
    maximum_grid = list(maximums) if maximums else list(DEFAULT_MAXIMUMS)
    has_direct = bool(direct_doc_records)
    rows = []
    for spec in specs:
        for direct_weight in direct_weights:
            if direct_weight > 0 and not has_direct:
                continue
            doc_scores = build_doc_scores(
                scored_records, direct_doc_records, aggregation=spec["method"], k=spec["k"],
                weight=spec["weight"], direct_weight=direct_weight,
                internal_to_official=internal_to_official, chunk_to_doc=chunk_to_doc)
            grid = thresholds if thresholds is not None else default_doc_thresholds(
                [row["score"] for scores in doc_scores.values() for row in scores])
            fold_scores = []
            for train_ids, val_ids in splits:
                chosen = _tune_on_train(doc_scores, labels_by_id, train_ids,
                                        grid, fallback_grid, maximum_grid)
                fold_scores.append(macro_f2_doc(doc_scores, labels_by_id, val_ids,
                                                chosen["doc_threshold"], chosen["doc_fallback"],
                                                chosen["doc_max"]))
            full = _tune_on_train(doc_scores, labels_by_id, query_ids, grid, fallback_grid, maximum_grid)
            mean = sum(fold_scores) / len(fold_scores)
            variance = sum((value - mean) ** 2 for value in fold_scores) / len(fold_scores)
            rows.append({"aggregation": spec["name"], "direct_doc_weight": direct_weight,
                         "oof_mean_f2": mean, "oof_std_f2": variance ** 0.5,
                         "doc_threshold": full["doc_threshold"], "doc_fallback": full["doc_fallback"],
                         "doc_max": full["doc_max"], "full_data_f2": full["train_f2"],
                         "_spec": spec, "_fold_scores": fold_scores})
    ablation = pd.DataFrame([{key: row[key] for key in ABLATION_COLUMNS} for row in rows],
                            columns=list(ABLATION_COLUMNS))
    ranked = sorted(rows, key=lambda row: (-row["oof_mean_f2"], row["oof_std_f2"],
                                           row["direct_doc_weight"], row["aggregation"]))
    winner = ranked[0]
    spec = winner["_spec"]
    best_pipeline = {
        "doc_aggregation": spec["method"], "doc_top_k": spec["k"], "doc_weight": spec["weight"],
        "doc_threshold": winner["doc_threshold"], "doc_fallback": winner["doc_fallback"],
        "doc_max": winner["doc_max"], "direct_doc_weight": winner["direct_doc_weight"],
        "ablation_method": spec["name"],
        "holdout": {"n_folds": len(splits), "seed": seed,
                    "mean_f2_doc": winner["oof_mean_f2"], "std_f2_doc": winner["oof_std_f2"],
                    "fold_f2_doc": winner["_fold_scores"]},
        "full_data_f2_doc": winner["full_data_f2"],
        "evaluation_scope": "kfold_out_of_fold",
        "direct_doc_enabled": winner["direct_doc_weight"] > 0,
    }
    return best_pipeline, ablation


PIPELINE_KEYS = ("doc_aggregation", "doc_top_k", "doc_weight", "doc_threshold",
                 "doc_fallback", "doc_max", "direct_doc_weight")


def doc_pipeline_config(best_pipeline):
    """Extract the deployable selector parameters from the tuned pipeline."""
    missing = [key for key in PIPELINE_KEYS if key not in best_pipeline]
    if missing:
        raise ValueError(f"Doc pipeline config is missing keys: {missing}")
    return {key: best_pipeline[key] for key in PIPELINE_KEYS}


def write_doc_pipeline(path, best_pipeline, *, ablation_csv=None):
    """Write ``best_doc_pipeline.yaml`` (exclusive creation)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Doc pipeline already exists: {path}")
    payload = {
        "doc_pipeline": doc_pipeline_config(best_pipeline),
        "holdout": best_pipeline["holdout"],
        "full_data_f2_doc": best_pipeline["full_data_f2_doc"],
        "ablation_method": best_pipeline["ablation_method"],
        "aggregation_ablation_csv": str(ablation_csv) if ablation_csv else None,
        "evaluation_scope": best_pipeline["evaluation_scope"],
        "target_stage": "P3-14_official_aggregation",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return payload


def load_doc_pipeline_config(path):
    """Load the ``doc_pipeline`` section for submission generation."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("doc_pipeline"), dict):
        raise ValueError(f"Invalid doc pipeline YAML: {path}")
    return doc_pipeline_config(payload["doc_pipeline"])
