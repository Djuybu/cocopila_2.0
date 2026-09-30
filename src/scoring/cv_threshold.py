"""P2-09: Cross-validation and bootstrap threshold tuning for reranker candidates.

Prevents overfitting to a single validation split by evaluating thresholds across multiple
query folds or bootstrap iterations, selecting a stable threshold with mean/std F2 reporting.
"""
from collections import defaultdict
import math
import random
import numpy as np
import pandas as pd

from src.data.adapter import official_candidates
from src.data.schema import unique_ids
from src.evaluation.fbeta import classification_metrics
from src.scoring.chunk_selector import select_ids
from src.scoring.doc_aggregation import candidate_score


def kfold_query_split(query_ids, n_folds=5, seed=42):
    """Partition query IDs into deterministic K folds.

    Returns:
        list of tuples: [(train_qids, val_qids), ...]
    """
    sorted_qids = sorted(list(set(query_ids)))
    if len(sorted_qids) < 2:
        raise ValueError("At least 2 queries required for K-fold split")
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    effective_folds = min(n_folds, len(sorted_qids))
    rng = random.Random(seed)
    shuffled = list(sorted_qids)
    rng.shuffle(shuffled)

    folds = [[] for _ in range(effective_folds)]
    for idx, qid in enumerate(shuffled):
        folds[idx % effective_folds].append(qid)

    splits = []
    for f_idx in range(effective_folds):
        val_set = set(folds[f_idx])
        train_set = [qid for qid in shuffled if qid not in val_set]
        splits.append((train_set, sorted(val_set)))

    return splits


def bootstrap_query_split(query_ids, n_rounds=20, val_ratio=None, seed=42):
    """Sample training queries with replacement; validate on out-of-bag queries.

    By default draw N queries. val_ratio optionally controls the expected OOB
    fraction through the draw count, rather than forcing a validation size.

    Returns:
        list of tuples: [(train_qids, val_qids), ...]
    """
    sorted_qids = sorted(list(set(query_ids)))
    if len(sorted_qids) < 2:
        raise ValueError("At least 2 queries required for bootstrap split")

    if type(n_rounds) is not int or n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    sample_size = len(sorted_qids)
    if val_ratio is not None:
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        sample_size = max(1, round(math.log(val_ratio) / math.log(1 - 1 / len(sorted_qids))))

    rng = random.Random(seed)
    splits = []
    for _ in range(n_rounds):
        for attempt in range(1000):
            train_sample = rng.choices(sorted_qids, k=sample_size)
            val_set = set(sorted_qids) - set(train_sample)
            if val_set:
                splits.append((train_sample, sorted(val_set)))
                break
        else:
            raise ValueError("Unable to sample a nonempty out-of-bag validation set")

    return splits


def _bootstrap_mean_interval(values, seed=42):
    """Percentile interval for a mean, resampling query observations."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(1000)]
    return [round(float(v), 5) for v in np.quantile(means, [0.025, 0.975])]


def cross_validate_threshold(
    scored_records,
    labels,
    thresholds=None,
    steps=50,
    min_th=None,
    max_th=None,
    fallback=0,
    max_chunks=None,
    method="kfold",
    n_folds=5,
    n_rounds=20,
    seed=42,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """Cross-validate threshold selection across multiple folds or bootstrap iterations.

    Returns:
        tuple: (cv_df, report)
    """
    mapping = internal_to_official or {}
    if unique_ids(scored_records, "id") != unique_ids(labels, "id"):
        raise ValueError("Candidate and label query IDs must match exactly")
    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}
    all_query_ids = list(label_map.keys())
    num_queries = len(all_query_ids)
    if num_queries < 2:
        raise ValueError("At least 2 queries required for cross-validation")

    # Pre-process candidates per query
    prepared_queries = {}
    all_scores = []
    for row in scored_records:
        qid = row["id"]
        cands = row.get("candidates", [])
        if chunk_to_doc is not None:
            cands = official_candidates(cands, mapping, chunk_to_doc)
        elif mapping:
            cands = [{**c, "chunk_id": mapping.get(c["chunk_id"], c["chunk_id"])} for c in cands]
        prepared_queries[qid] = cands
        for c in cands:
            all_scores.append(candidate_score(c))

    # Grid of thresholds
    if thresholds is None:
        if steps < 1:
            raise ValueError("steps must be positive")
        if all_scores:
            s_min = min(all_scores) if min_th is None else float(min_th)
            s_max = max(all_scores) if max_th is None else float(max_th)
            if s_min == s_max:
                thresholds = [s_min]
            else:
                step = (s_max - s_min) / max(1, steps)
                thresholds = [round(s_min + i * step, 5) for i in range(steps + 1)]
        else:
            thresholds = [0.0, 0.5]

    if not thresholds:
        raise ValueError("Threshold grid must not be empty")
    effective_max = 999999 if max_chunks is None else int(max_chunks)

    # Generate splits
    if method == "kfold":
        splits = kfold_query_split(all_query_ids, n_folds=n_folds, seed=seed)
    elif method == "bootstrap":
        splits = bootstrap_query_split(all_query_ids, n_rounds=n_rounds, seed=seed)
    else:
        raise ValueError(f"Unknown cross-validation method '{method}'. Choose 'kfold' or 'bootstrap'.")

    # Precalculate per query, per threshold metric to speed up fold aggregations
    query_th_metrics = defaultdict(dict)
    for th in thresholds:
        for qid in all_query_ids:
            cands = prepared_queries.get(qid, [])
            selected = select_ids(cands, "chunk_id", th, fallback, effective_max)
            m = classification_metrics(label_map[qid], set(selected), zero_division=0.0)
            m["chunks_count"] = len(selected)
            query_th_metrics[th][qid] = m

    # Measure performance across validation folds for each threshold
    th_records = []
    for th in thresholds:
        fold_f2s = []
        fold_f1s = []
        fold_precs = []
        fold_recs = []
        fold_chunks = []

        for train_qids, val_qids in splits:
            f2_sum = sum(query_th_metrics[th][qid]["f2"] for qid in val_qids)
            f1_sum = sum(query_th_metrics[th][qid]["f1"] for qid in val_qids)
            p_sum = sum(query_th_metrics[th][qid]["precision"] for qid in val_qids)
            r_sum = sum(query_th_metrics[th][qid]["recall"] for qid in val_qids)
            c_sum = sum(query_th_metrics[th][qid]["chunks_count"] for qid in val_qids)
            n_val = len(val_qids)

            fold_f2s.append(f2_sum / n_val)
            fold_f1s.append(f1_sum / n_val)
            fold_precs.append(p_sum / n_val)
            fold_recs.append(r_sum / n_val)
            fold_chunks.append(c_sum / n_val)

        mean_f2 = float(np.mean(fold_f2s))
        std_f2 = float(np.std(fold_f2s))
        mean_p = float(np.mean(fold_precs))
        mean_r = float(np.mean(fold_recs))
        mean_chunks = float(np.mean(fold_chunks))

        # Stability score: penalizes high variance across splits
        stability_score = round(mean_f2 - 0.25 * std_f2, 5)

        th_records.append({
            "threshold": th,
            "mean_f2": round(mean_f2, 5),
            "std_f2": round(std_f2, 5),
            "min_f2": round(float(np.min(fold_f2s)), 5),
            "max_f2": round(float(np.max(fold_f2s)), 5),
            "mean_precision": round(mean_p, 5),
            "mean_recall": round(mean_r, 5),
            "avg_chunks_per_query": round(mean_chunks, 2),
            "stability_score": stability_score,
        })

    cv_df = pd.DataFrame(th_records)
    cv_df.sort_values(
        by=["stability_score", "mean_f2", "mean_recall", "avg_chunks_per_query"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    cv_df.reset_index(drop=True, inplace=True)

    # Out-of-fold generalization test: tune on train, test on val
    oof_val_f2s = []
    oof_chosen_ths = []
    oof_query_f2s = defaultdict(list)
    for train_qids, val_qids in splits:
        # Find best th on train
        best_train_th = None
        best_train_f2 = -1.0
        for th in thresholds:
            t_f2 = sum(query_th_metrics[th][qid]["f2"] for qid in train_qids) / len(train_qids)
            if t_f2 > best_train_f2:
                best_train_f2 = t_f2
                best_train_th = th
        oof_chosen_ths.append(best_train_th)
        # Score on val
        v_f2 = sum(query_th_metrics[best_train_th][qid]["f2"] for qid in val_qids) / len(val_qids)
        oof_val_f2s.append(v_f2)
        for qid in val_qids:
            oof_query_f2s[qid].append(query_th_metrics[best_train_th][qid]["f2"])

    best_stable_row = cv_df.iloc[0].to_dict()
    pure_f2_row = cv_df.sort_values(by="mean_f2", ascending=False).iloc[0].to_dict()
    oof_values = [float(np.mean(values)) for values in oof_query_f2s.values()]

    report = {
        "method": method,
        "n_splits": len(splits),
        "total_queries": num_queries,
        "fallback": fallback,
        "max_chunks": max_chunks,
        "recommended_stable_threshold": {
            "threshold": best_stable_row["threshold"],
            "mean_f2": best_stable_row["mean_f2"],
            "std_f2": best_stable_row["std_f2"],
            "confidence_interval_95": _bootstrap_mean_interval(
                [query_th_metrics[best_stable_row["threshold"]][qid]["f2"] for qid in all_query_ids], seed),
            "confidence_interval_method": "query_bootstrap_percentile_fixed_threshold",
            "evaluation_scope": "threshold_selection_on_supplied_labels",
            "stability_score": best_stable_row["stability_score"],
            "mean_precision": best_stable_row["mean_precision"],
            "mean_recall": best_stable_row["mean_recall"],
            "avg_chunks_per_query": best_stable_row["avg_chunks_per_query"],
        },
        "pure_peak_threshold": {
            "threshold": pure_f2_row["threshold"],
            "mean_f2": pure_f2_row["mean_f2"],
            "std_f2": pure_f2_row["std_f2"],
        },
        "out_of_fold_generalization": {
            "mean_oof_f2": round(float(np.mean(oof_values)), 5),
            "std_oof_f2": round(float(np.std(oof_val_f2s)), 5),
            "chosen_thresholds": oof_chosen_ths,
            "confidence_interval_95": _bootstrap_mean_interval(oof_values, seed),
            "evaluated_query_count": len(oof_values),
        },
        "cv_table": cv_df.to_dict(orient="records"),
    }

    return cv_df, report


def cross_validate_selector(scored_records, labels, *, thresholds=None,
                            fallbacks=None, maximums=None, n_folds=5, seed=42,
                            internal_to_official=None, chunk_to_doc=None):
    """Tune all selector parameters on training folds and score held-out queries."""
    from src.scoring.sweep import sweep_chunk_selector
    from src.evaluation.evaluate_f2 import evaluate_candidate_selection

    query_ids = unique_ids(labels, "id")
    if unique_ids(scored_records, "id") != query_ids:
        raise ValueError("Candidate and label query IDs must match exactly")
    if len(query_ids) < 2:
        return {"status": "insufficient_queries", "query_count": len(query_ids)}
    by_query = {row["id"]: row for row in scored_records}
    by_label = {row["id"]: row for row in labels}
    folds, per_query = [], {}
    for train_ids, val_ids in kfold_query_split(query_ids, n_folds, seed):
        best, _, _ = sweep_chunk_selector(
            [by_query[qid] for qid in train_ids], [by_label[qid] for qid in train_ids],
            thresholds=thresholds, fallbacks=fallbacks, maximums=maximums,
            internal_to_official=internal_to_official, chunk_to_doc=chunk_to_doc,
        )
        val_records = [by_query[qid] for qid in val_ids]
        if chunk_to_doc is not None:
            val_records = [{**row, "candidates": official_candidates(
                row["candidates"], internal_to_official or {}, chunk_to_doc)} for row in val_records]
        evaluation = evaluate_candidate_selection(
            val_records, [by_label[qid] for qid in val_ids],
            threshold=best["chunk_threshold"], fallback=best["chunk_fallback"],
            max_chunks=best["chunk_max"], internal_to_official=internal_to_official,
        )
        per_query.update(evaluation["per_query"])
        folds.append({"train_query_ids": train_ids, "val_query_ids": val_ids,
                      "selector": best, "validation_macro": evaluation["macro"]})
    return {
        "status": "complete", "method": "kfold", "n_splits": len(folds),
        "query_count": len(query_ids), "seed": seed,
        "evaluation_scope": "out_of_fold_selector_tuning",
        "macro": {name: sum(row[name] for row in per_query.values()) / len(per_query)
                  for name in ("precision", "recall", "f1", "f2")},
        "confidence_interval_95": _bootstrap_mean_interval([row["f2"] for row in per_query.values()], seed),
        "per_query": per_query, "folds": folds,
    }
