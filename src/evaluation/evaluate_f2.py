"""P2-02: Chunk-level F2 evaluation module.

Calculates precision, recall, F1, and F2 per query and macro-aggregates
across queries for relevant_chunks.
"""
from collections import OrderedDict
import csv
import json
from pathlib import Path

from src.data.schema import unique_ids
from src.scoring.chunk_selector import select_ids
from src.scoring.doc_aggregation import candidate_score
from src.utils.io import write_json


def compute_chunk_metrics(y_true, y_pred, zero_division=0.0):
    """Compute set-based classification metrics for a single query.

    Formula:
        TP = |T ∩ P|
        FP = |P \\ T|
        FN = |T \\ P|
        Precision = TP / |P|
        Recall = TP / |T|
        F1 = (2 * TP) / (|T| + |P|)
        F2 = (5 * TP) / (4 * |T| + |P|)
    """
    truth = set(y_true)
    predicted = set(y_pred)
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / len(predicted) if predicted else float(zero_division)
    recall = tp / len(truth) if truth else float(zero_division)

    f1_denom = len(truth) + len(predicted)
    f1 = (2.0 * tp) / f1_denom if f1_denom > 0 else float(zero_division)

    f2_denom = 4.0 * len(truth) + len(predicted)
    f2 = (5.0 * tp) / f2_denom if f2_denom > 0 else float(zero_division)

    return {
        "num_truth": len(truth),
        "num_predicted": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


def evaluate_chunk_predictions(
    predictions,
    ground_truths,
    internal_to_official=None,
    zero_division=0.0,
):
    """Evaluate chunk-level predictions against ground truth labels.

    Args:
        predictions: List of dicts (with 'id' and 'relevant_chunks') or dict {qid: [chunk_ids]}.
        ground_truths: List of dicts (with 'id' and 'relevant_chunks') or dict {qid: [chunk_ids]}.
        internal_to_official: Optional dict mapping internal chunk IDs to official chunk IDs.
        zero_division: Value to use when denominator is zero (default 0.0).

    Returns:
        dict: Complete evaluation report with per-query and macro metrics.
    """
    mapping = internal_to_official or {}

    # Normalize predictions to dict {qid: list of official chunk_ids}
    if isinstance(predictions, dict):
        pred_map = {
            qid: [mapping.get(cid, cid) for cid in cids]
            for qid, cids in predictions.items()
        }
    else:
        unique_ids(predictions, "id")
        pred_map = {
            row["id"]: [mapping.get(cid, cid) for cid in row.get("relevant_chunks", [])]
            for row in predictions
        }

    # Normalize ground_truths to dict {qid: list of official chunk_ids}
    if isinstance(ground_truths, dict):
        truth_map = {
            qid: [mapping.get(cid, cid) for cid in cids]
            for qid, cids in ground_truths.items()
        }
    else:
        unique_ids(ground_truths, "id")
        truth_map = {
            row["id"]: [mapping.get(cid, cid) for cid in row.get("relevant_chunks", [])]
            for row in ground_truths
        }

    if set(pred_map.keys()) != set(truth_map.keys()):
        diff = set(pred_map.keys()) ^ set(truth_map.keys())
        raise ValueError(f"Prediction and ground-truth query IDs must match exactly. Mismatch on: {sorted(diff)}")

    per_query = OrderedDict()
    for qid in sorted(truth_map.keys()):
        metrics = compute_chunk_metrics(
            truth_map[qid],
            pred_map[qid],
            zero_division=zero_division,
        )
        metrics["truth_chunks"] = list(dict.fromkeys(truth_map[qid]))
        metrics["predicted_chunks"] = list(dict.fromkeys(pred_map[qid]))
        per_query[qid] = metrics

    query_count = len(per_query)
    if query_count > 0:
        macro = {
            "precision": sum(m["precision"] for m in per_query.values()) / query_count,
            "recall": sum(m["recall"] for m in per_query.values()) / query_count,
            "f1": sum(m["f1"] for m in per_query.values()) / query_count,
            "f2": sum(m["f2"] for m in per_query.values()) / query_count,
            "avg_tp": sum(m["tp"] for m in per_query.values()) / query_count,
            "avg_fp": sum(m["fp"] for m in per_query.values()) / query_count,
            "avg_fn": sum(m["fn"] for m in per_query.values()) / query_count,
            "avg_truth": sum(m["num_truth"] for m in per_query.values()) / query_count,
            "avg_predicted": sum(m["num_predicted"] for m in per_query.values()) / query_count,
        }
    else:
        macro = {
            "precision": float(zero_division),
            "recall": float(zero_division),
            "f1": float(zero_division),
            "f2": float(zero_division),
            "avg_tp": 0.0,
            "avg_fp": 0.0,
            "avg_fn": 0.0,
            "avg_truth": 0.0,
            "avg_predicted": 0.0,
        }

    return {
        "stage": "chunk_level_f2",
        "query_count": query_count,
        "zero_division": zero_division,
        "macro": macro,
        "per_query": per_query,
    }


def evaluate_candidate_selection(
    records,
    ground_truths,
    top_k=None,
    threshold=None,
    fallback=None,
    max_chunks=None,
    internal_to_official=None,
    zero_division=0.0,
):
    """Evaluate candidates by applying top-k ranking or threshold/fallback/max selection.

    Args:
        records: List of candidate records per query, each with 'id' and 'candidates'.
        ground_truths: List of ground truth labels or dict.
        top_k: If provided, select the top-k highest scoring chunks per query.
        threshold: Score cutoff for chunk selection.
        fallback: Minimum chunks to select if fewer pass the threshold.
        max_chunks: Maximum chunk cap per query.
        internal_to_official: Optional ID mapping dict.
        zero_division: Policy for zero division.
    """
    mapping = internal_to_official or {}
    predictions = []
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("top_k must be a nonnegative integer")

    for row in records:
        qid = row["id"]
        cands = row.get("candidates", [])

        # Apply internal to official chunk mapping
        mapped_cands = []
        for c in cands:
            cid = mapping.get(c["chunk_id"], c["chunk_id"])
            mapped_cands.append({**c, "chunk_id": cid})

        if top_k is not None:
            # Sort by candidate score descending and deduplicate by chunk_id
            ordered, seen = [], set()
            for c in sorted(mapped_cands, key=lambda item: (-candidate_score(item), item["chunk_id"])):
                if c["chunk_id"] not in seen:
                    ordered.append(c["chunk_id"])
                    seen.add(c["chunk_id"])
            selected = ordered[:top_k]
        else:
            fb = fallback if fallback is not None else 0
            mx = max_chunks if max_chunks is not None else max(len(mapped_cands), fb)
            selected = select_ids(mapped_cands, "chunk_id", threshold, fb, mx)

        predictions.append({
            "id": qid,
            "relevant_chunks": selected,
        })

    return evaluate_chunk_predictions(
        predictions,
        ground_truths,
        internal_to_official=mapping,
        zero_division=zero_division,
    )


def format_metrics_table(report):
    """Format report into a human-readable ASCII table."""
    headers = ["Query ID", "|Truth|", "|Pred|", "TP", "FP", "FN", "Precision", "Recall", "F1", "F2"]
    rows = []
    for qid, m in report["per_query"].items():
        rows.append([
            qid,
            str(m["num_truth"]),
            str(m["num_predicted"]),
            str(m["tp"]),
            str(m["fp"]),
            str(m["fn"]),
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['f2']:.4f}",
        ])

    macro = report["macro"]
    summary_row = [
        "MACRO AVERAGE",
        f"{macro['avg_truth']:.2f}",
        f"{macro['avg_predicted']:.2f}",
        f"{macro['avg_tp']:.2f}",
        f"{macro['avg_fp']:.2f}",
        f"{macro['avg_fn']:.2f}",
        f"{macro['precision']:.4f}",
        f"{macro['recall']:.4f}",
        f"{macro['f1']:.4f}",
        f"{macro['f2']:.4f}",
    ]

    col_widths = [len(h) for h in headers]
    for row in rows + [summary_row]:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def make_line(vals):
        return "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(vals)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    lines = [
        sep,
        make_line(headers),
        sep,
    ]
    for row in rows:
        lines.append(make_line(row))
    lines.append(sep)
    lines.append(make_line(summary_row))
    lines.append(sep)
    return "\n".join(lines)


def export_metrics_csv(report, output_path):
    """Export evaluation report to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id", "num_truth", "num_predicted", "tp", "fp", "fn",
        "precision", "recall", "f1", "f2", "truth_chunks", "predicted_chunks",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for qid, m in report["per_query"].items():
            writer.writerow({
                "query_id": qid,
                "num_truth": m["num_truth"],
                "num_predicted": m["num_predicted"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": f"{m['precision']:.6f}",
                "recall": f"{m['recall']:.6f}",
                "f1": f"{m['f1']:.6f}",
                "f2": f"{m['f2']:.6f}",
                "truth_chunks": ";".join(m.get("truth_chunks", [])),
                "predicted_chunks": ";".join(m.get("predicted_chunks", [])),
            })

        macro = report["macro"]
        writer.writerow({
            "query_id": "MACRO_AVERAGE",
            "num_truth": f"{macro['avg_truth']:.4f}",
            "num_predicted": f"{macro['avg_predicted']:.4f}",
            "tp": f"{macro['avg_tp']:.4f}",
            "fp": f"{macro['avg_fp']:.4f}",
            "fn": f"{macro['avg_fn']:.4f}",
            "precision": f"{macro['precision']:.6f}",
            "recall": f"{macro['recall']:.6f}",
            "f1": f"{macro['f1']:.6f}",
            "f2": f"{macro['f2']:.6f}",
            "truth_chunks": "",
            "predicted_chunks": "",
        })


def export_metrics_json(report, output_path):
    """Export evaluation report to a JSON file."""
    write_json(output_path, report)
