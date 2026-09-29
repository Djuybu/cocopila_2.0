"""P2-09: Cross-validation and bootstrap threshold tuning for reranker candidates.

Prevents overfitting to a single validation split by evaluating thresholds across multiple
query folds or bootstrap iterations, selecting a stable threshold with mean/std F2 reporting.
"""
from collections import defaultdict
import math
from pathlib import Path
import random
import numpy as np
import pandas as pd

from src.data.adapter import official_candidates
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
        splits.append((train_set, list(val_set)))

    return splits


def bootstrap_query_split(query_ids, n_rounds=20, val_ratio=0.3, seed=42):
    """Generate bootstrap resampled splits.

    Returns:
        list of tuples: [(train_qids, val_qids), ...]
    """
    sorted_qids = sorted(list(set(query_ids)))
    if len(sorted_qids) < 2:
        raise ValueError("At least 2 queries required for bootstrap split")

    val_size = max(1, int(math.ceil(len(sorted_qids) * val_ratio)))
    if val_size >= len(sorted_qids):
        val_size = len(sorted_qids) - 1

    rng = random.Random(seed)
    splits = []
    for _ in range(n_rounds):
        shuffled = list(sorted_qids)
        rng.shuffle(shuffled)
        val_set = set(shuffled[:val_size])
        train_set = [qid for qid in shuffled if qid not in val_set]
        splits.append((train_set, list(val_set)))

    return splits


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

    best_stable_row = cv_df.iloc[0].to_dict()
    pure_f2_row = cv_df.sort_values(by="mean_f2", ascending=False).iloc[0].to_dict()

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
            "confidence_interval_95": [
                round(max(0.0, best_stable_row["mean_f2"] - 1.96 * best_stable_row["std_f2"]), 5),
                round(min(1.0, best_stable_row["mean_f2"] + 1.96 * best_stable_row["std_f2"]), 5),
            ],
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
            "mean_oof_f2": round(float(np.mean(oof_val_f2s)), 5),
            "std_oof_f2": round(float(np.std(oof_val_f2s)), 5),
            "chosen_thresholds": oof_chosen_ths,
        },
        "cv_table": cv_df.to_dict(orient="records"),
    }

    return cv_df, report
