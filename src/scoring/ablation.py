"""P2-04 & P2-05: Fallback Top-N and Output Cap Max-N sweep & ablation analysis.

Provides ablation comparison between threshold-only selection, minimum fallback (Top-N),
and maximum output cap (Max-N) to optimize chunk-level F2.
"""
from collections import defaultdict
from pathlib import Path
import pandas as pd

from src.data.adapter import official_candidates
from src.evaluation.fbeta import classification_metrics
from src.scoring.chunk_selector import select_ids
from src.scoring.doc_aggregation import candidate_score


def _prepare_query_candidates(scored_records, internal_to_official=None, chunk_to_doc=None):
    """Normalize candidate records per query with official mapping."""
    mapping = internal_to_official or {}
    prepared = {}
    for row in scored_records:
        qid = row["id"]
        cands = row.get("candidates", [])
        if chunk_to_doc is not None:
            cands = official_candidates(cands, mapping, chunk_to_doc)
        elif mapping:
            cands = [{**c, "chunk_id": mapping.get(c["chunk_id"], c["chunk_id"])} for c in cands]
        prepared[qid] = cands
    return prepared


def sweep_fallback(
    scored_records,
    labels,
    threshold,
    fallbacks=None,
    max_chunks=999999,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """P2-04: Sweep minimum fallback Top-N at a fixed threshold.

    Parameters:
        scored_records: List of scored candidate records per query.
        labels: Ground truth labels (relevant_chunks).
        threshold: Fixed threshold score (e.g. recommended from P2-03).
        fallbacks: List of fallback values to evaluate (default: [0, 1, 2, 3, 5, 10]).
        max_chunks: Upper cap on chunks per query (default: 999999).
        internal_to_official: Chunk ID mapping dictionary.
        chunk_to_doc: Mapping from chunk to document ID.

    Returns:
        pd.DataFrame: Sweep results sorted by macro_f2 descending.
    """
    if fallbacks is None:
        fallbacks = [0, 1, 2, 3, 5, 10]

    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}
    num_queries = len(label_map)
    if num_queries == 0:
        raise ValueError("Cannot sweep fallback with empty labels")

    prepared_queries = _prepare_query_candidates(scored_records, internal_to_official, chunk_to_doc)
    effective_max = int(max_chunks) if max_chunks is not None else 999999

    rows = []
    for fb in fallbacks:
        if fb > effective_max:
            continue

        total_p, total_r, total_f1, total_f2 = 0.0, 0.0, 0.0, 0.0
        chunk_counts = []
        zero_queries = 0

        for qid, truth_chunks in label_map.items():
            cands = prepared_queries.get(qid, [])
            selected = select_ids(cands, "chunk_id", threshold, fb, effective_max)
            n_sel = len(selected)
            chunk_counts.append(n_sel)
            if n_sel == 0:
                zero_queries += 1

            metrics = classification_metrics(truth_chunks, set(selected), zero_division=0.0)
            total_p += metrics["precision"]
            total_r += metrics["recall"]
            total_f1 += metrics["f1"]
            total_f2 += metrics["f2"]

        macro_p = total_p / num_queries
        macro_r = total_r / num_queries
        macro_f1 = total_f1 / num_queries
        macro_f2 = total_f2 / num_queries
        avg_chunks = sum(chunk_counts) / num_queries
        min_chunks = min(chunk_counts) if chunk_counts else 0
        max_c = max(chunk_counts) if chunk_counts else 0

        rows.append({
            "fallback": fb,
            "threshold": threshold,
            "macro_f2": round(macro_f2, 5),
            "macro_f1": round(macro_f1, 5),
            "macro_precision": round(macro_p, 5),
            "macro_recall": round(macro_r, 5),
            "avg_chunks_per_query": round(avg_chunks, 2),
            "min_chunks_per_query": min_chunks,
            "max_chunks_per_query": max_c,
            "zero_chunk_queries": zero_queries,
            "zero_chunk_ratio": round(zero_queries / num_queries, 4),
        })

    df = pd.DataFrame(rows)
    df.sort_values(
        by=["macro_f2", "macro_recall", "macro_precision", "avg_chunks_per_query"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def ablation_threshold_vs_fallback(
    scored_records,
    labels,
    threshold,
    fallbacks=None,
    max_chunks=999999,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """P2-04: Ablation comparison between threshold-only selection vs threshold + fallback.

    Returns:
        dict: Detailed comparison report containing baseline, best fallback, delta metrics,
              and textual analysis.
    """
    if fallbacks is None:
        fallbacks = [0, 1, 2, 3, 5, 10]
    if 0 not in fallbacks:
        fallbacks = [0] + list(fallbacks)

    sweep_df = sweep_fallback(
        scored_records=scored_records,
        labels=labels,
        threshold=threshold,
        fallbacks=fallbacks,
        max_chunks=max_chunks,
        internal_to_official=internal_to_official,
        chunk_to_doc=chunk_to_doc,
    )

    baseline_match = sweep_df[sweep_df["fallback"] == 0]
    if baseline_match.empty:
        raise ValueError("Baseline fallback=0 missing from sweep results")
    baseline = baseline_match.iloc[0].to_dict()

    best_match = sweep_df.iloc[0].to_dict()
    best_fb = int(best_match["fallback"])

    f2_diff = round(best_match["macro_f2"] - baseline["macro_f2"], 5)
    recall_diff = round(best_match["macro_recall"] - baseline["macro_recall"], 5)
    precision_diff = round(best_match["macro_precision"] - baseline["macro_precision"], 5)
    zero_chunk_reduced = int(baseline["zero_chunk_queries"] - best_match["zero_chunk_queries"])

    if f2_diff > 0:
        verdict = f"Fallback Top-{best_fb} cải thiện Macro F2 (+{f2_diff:+.4f}) và giảm {zero_chunk_reduced} queries không có chunk."
    elif f2_diff == 0:
        verdict = f"Fallback Top-{best_fb} duy trì Macro F2 không đổi (+0.0000), tỷ lệ zero-chunk giảm từ {baseline['zero_chunk_queries']} -> {best_match['zero_chunk_queries']}."
    else:
        verdict = f"Threshold-only tốt hơn fallback (F2 giảm {f2_diff:.4f} khi bật fallback {best_fb})."

    return {
        "threshold": threshold,
        "baseline_threshold_only": {
            "fallback": 0,
            "macro_f2": baseline["macro_f2"],
            "macro_precision": baseline["macro_precision"],
            "macro_recall": baseline["macro_recall"],
            "avg_chunks_per_query": baseline["avg_chunks_per_query"],
            "zero_chunk_queries": int(baseline["zero_chunk_queries"]),
        },
        "best_fallback_config": {
            "fallback": best_fb,
            "macro_f2": best_match["macro_f2"],
            "macro_precision": best_match["macro_precision"],
            "macro_recall": best_match["macro_recall"],
            "avg_chunks_per_query": best_match["avg_chunks_per_query"],
            "zero_chunk_queries": int(best_match["zero_chunk_queries"]),
        },
        "delta": {
            "delta_f2": f2_diff,
            "delta_recall": recall_diff,
            "delta_precision": precision_diff,
            "zero_queries_eliminated": zero_chunk_reduced,
        },
        "verdict": verdict,
        "sweep_table": sweep_df.to_dict(orient="records"),
    }


def sweep_max_output(
    scored_records,
    labels,
    threshold,
    fallback=0,
    maximums=None,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """P2-05: Sweep max-N output cap at a fixed threshold and fallback.

    Parameters:
        scored_records: List of scored candidate records per query.
        labels: Ground truth labels (relevant_chunks).
        threshold: Score cutoff threshold.
        fallback: Minimum fallback Top-N (default 0).
        maximums: List of maximum chunk caps (default: [1, 2, 3, 5, 8, 10, 15, 20, 50, 100]).
        internal_to_official: ID mapping dict.
        chunk_to_doc: Mapping chunk to doc ID.

    Returns:
        pd.DataFrame: Sweep results sorted by macro_f2 descending.
    """
    if maximums is None:
        maximums = [1, 2, 3, 5, 8, 10, 15, 20, 50, 100]

    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}
    num_queries = len(label_map)
    if num_queries == 0:
        raise ValueError("Cannot sweep max output with empty labels")

    prepared_queries = _prepare_query_candidates(scored_records, internal_to_official, chunk_to_doc)

    rows = []
    for mx in maximums:
        if mx < fallback:
            continue

        total_p, total_r, total_f1, total_f2 = 0.0, 0.0, 0.0, 0.0
        chunk_counts = []
        zero_queries = 0

        for qid, truth_chunks in label_map.items():
            cands = prepared_queries.get(qid, [])
            selected = select_ids(cands, "chunk_id", threshold, fallback, mx)
            n_sel = len(selected)
            chunk_counts.append(n_sel)
            if n_sel == 0:
                zero_queries += 1

            metrics = classification_metrics(truth_chunks, set(selected), zero_division=0.0)
            total_p += metrics["precision"]
            total_r += metrics["recall"]
            total_f1 += metrics["f1"]
            total_f2 += metrics["f2"]

        macro_p = total_p / num_queries
        macro_r = total_r / num_queries
        macro_f1 = total_f1 / num_queries
        macro_f2 = total_f2 / num_queries
        avg_chunks = sum(chunk_counts) / num_queries
        min_chunks = min(chunk_counts) if chunk_counts else 0
        max_c = max(chunk_counts) if chunk_counts else 0

        rows.append({
            "max_chunks": mx,
            "threshold": threshold,
            "fallback": fallback,
            "macro_f2": round(macro_f2, 5),
            "macro_f1": round(macro_f1, 5),
            "macro_precision": round(macro_p, 5),
            "macro_recall": round(macro_r, 5),
            "avg_chunks_per_query": round(avg_chunks, 2),
            "min_chunks_per_query": min_chunks,
            "max_chunks_per_query": max_c,
            "zero_chunk_queries": zero_queries,
            "zero_chunk_ratio": round(zero_queries / num_queries, 4),
        })

    df = pd.DataFrame(rows)
    df.sort_values(
        by=["macro_f2", "macro_recall", "macro_precision", "avg_chunks_per_query"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def analyze_precision_recall_tradeoff(sweep_df):
    """P2-05: Analyze Precision vs Recall vs F2 tradeoff across max-N values.

    Returns:
        dict: Tradeoff summary including best max cap, precision gain vs unlimited,
              and recall retention.
    """
    if sweep_df.empty:
        raise ValueError("Cannot analyze empty sweep DataFrame")

    sorted_by_max = sweep_df.sort_values(by="max_chunks", ascending=True).reset_index(drop=True)
    largest_cap_row = sorted_by_max.iloc[-1].to_dict()
    best_row = sweep_df.iloc[0].to_dict()

    best_max = int(best_row["max_chunks"])
    p_gain = round(best_row["macro_precision"] - largest_cap_row["macro_precision"], 5)
    r_loss = round(largest_cap_row["macro_recall"] - best_row["macro_recall"], 5)
    f2_delta = round(best_row["macro_f2"] - largest_cap_row["macro_f2"], 5)

    if p_gain > 0 and r_loss <= 0.02:
        analysis = (
            f"Cap max_chunks={best_max} tăng Precision (+{p_gain:+.4f}) "
            f"mà không làm suy giảm Recall đáng kể (-{r_loss:.4f}), nâng F2 (+{f2_delta:+.4f})."
        )
    elif best_max == largest_cap_row["max_chunks"]:
        analysis = "Giới hạn max_chunks không mang lại lợi thế hơn so với mức tối đa hiện tại."
    else:
        analysis = (
            f"Cap max_chunks={best_max} đạt điểm cân bằng F2 tối ưu: "
            f"Precision={best_row['macro_precision']:.4f}, Recall={best_row['macro_recall']:.4f}."
        )

    return {
        "best_max_chunks": best_max,
        "best_macro_f2": best_row["macro_f2"],
        "best_macro_precision": best_row["macro_precision"],
        "best_macro_recall": best_row["macro_recall"],
        "unlimited_baseline": {
            "max_chunks": int(largest_cap_row["max_chunks"]),
            "macro_f2": largest_cap_row["macro_f2"],
            "macro_precision": largest_cap_row["macro_precision"],
            "macro_recall": largest_cap_row["macro_recall"],
        },
        "tradeoff": {
            "precision_gain": p_gain,
            "recall_loss": r_loss,
            "f2_delta": f2_delta,
        },
        "analysis": analysis,
    }
