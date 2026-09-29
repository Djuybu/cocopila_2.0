"""Joint selector parameter sweep, False Negative error analysis, and P2->P3 handoff packaging."""
from collections import defaultdict
import copy
from pathlib import Path
import shutil
import zipfile
import pandas as pd
import yaml

from src.data.adapter import official_candidates
from src.data.download import sha256_file
from src.data.loader import load_records
from src.evaluation.fbeta import classification_metrics
from src.retrieval.schema import SCHEMA_VERSION
from src.scoring.chunk_selector import select_ids
from src.scoring.doc_aggregation import candidate_score
from src.submission.validator import SubmissionValidator
from src.utils.io import read_json, write_json, write_jsonl


def detect_threshold_plateau(
    df,
    threshold_col="threshold",
    metric_col="macro_f2",
    tolerance=0.01,
):
    """Detect the contiguous threshold plateau interval around the maximum metric.

    Parameters:
        df: pd.DataFrame containing threshold and metric columns.
        threshold_col: Name of column containing numerical threshold values.
        metric_col: Name of metric column to optimize (e.g. macro_f2).
        tolerance: Relative degradation tolerance below peak (default: 0.01 = 1%).

    Returns:
        dict: Plateau metadata including min/max thresholds, width, count, and recommended threshold.
    """
    col_th = threshold_col if threshold_col in df.columns else (
        "chunk_threshold" if "chunk_threshold" in df.columns else None
    )
    col_m = metric_col if metric_col in df.columns else None

    if col_th is None or col_m is None or df.empty:
        return {
            "plateau_min_threshold": None,
            "plateau_max_threshold": None,
            "plateau_width": 0.0,
            "plateau_count": 0,
            "best_f2": 0.0,
            "recommended_threshold": None,
            "plateau_mean_f2": 0.0,
            "tolerance": tolerance,
        }

    # Filter out null thresholds and sort by threshold ascending
    valid_df = df.dropna(subset=[col_th]).copy()
    if valid_df.empty:
        return {
            "plateau_min_threshold": None,
            "plateau_max_threshold": None,
            "plateau_width": 0.0,
            "plateau_count": 0,
            "best_f2": 0.0,
            "recommended_threshold": None,
            "plateau_mean_f2": 0.0,
            "tolerance": tolerance,
        }

    valid_df[col_th] = valid_df[col_th].astype(float)
    valid_df = valid_df.sort_values(by=col_th).reset_index(drop=True)

    best_metric = float(valid_df[col_m].max())
    if best_metric <= 0:
        first_th = float(valid_df[col_th].iloc[0])
        return {
            "plateau_min_threshold": first_th,
            "plateau_max_threshold": first_th,
            "plateau_width": 0.0,
            "plateau_count": 1,
            "best_f2": 0.0,
            "recommended_threshold": first_th,
            "plateau_mean_f2": 0.0,
            "tolerance": tolerance,
        }

    delta = tolerance * (best_metric if best_metric > 0 else 1.0)
    cutoff = best_metric - delta
    eligible_mask = valid_df[col_m] >= cutoff

    # Find contiguous segments of eligible indices
    segments = []
    current_seg = []
    for idx, is_eligible in enumerate(eligible_mask):
        if is_eligible:
            current_seg.append(idx)
        else:
            if current_seg:
                segments.append(current_seg)
                current_seg = []
    if current_seg:
        segments.append(current_seg)

    if not segments:
        return {
            "plateau_min_threshold": None,
            "plateau_max_threshold": None,
            "plateau_width": 0.0,
            "plateau_count": 0,
            "best_f2": best_metric,
            "recommended_threshold": None,
            "plateau_mean_f2": 0.0,
            "tolerance": tolerance,
        }

    # Find the segment containing the global peak (or best mean score)
    best_indices = set(valid_df[valid_df[col_m] == best_metric].index)
    target_seg = None
    for seg in segments:
        if any(idx in best_indices for idx in seg):
            target_seg = seg
            break
    if target_seg is None:
        target_seg = max(segments, key=len)

    seg_df = valid_df.iloc[target_seg]
    min_th = float(seg_df[col_th].min())
    max_th = float(seg_df[col_th].max())
    width = round(max_th - min_th, 5)
    count = len(seg_df)
    mean_f2 = round(float(seg_df[col_m].mean()), 5)

    # Pick the recommended threshold: median/center of the plateau interval
    center_val = (min_th + max_th) / 2.0
    # Find the evaluated threshold closest to center_val within this segment
    closest_idx = (seg_df[col_th] - center_val).abs().idxmin()
    rec_th = float(seg_df.loc[closest_idx, col_th])

    return {
        "plateau_min_threshold": min_th,
        "plateau_max_threshold": max_th,
        "plateau_width": width,
        "plateau_count": count,
        "best_f2": round(best_metric, 5),
        "recommended_threshold": rec_th,
        "plateau_mean_f2": mean_f2,
        "tolerance": tolerance,
    }


def sweep_reranker_thresholds(
    scored_records,
    labels,
    thresholds=None,
    steps=50,
    min_th=None,
    max_th=None,
    fallback=0,
    max_chunks=None,
    internal_to_official=None,
    chunk_to_doc=None,
    tolerance=0.01,
):
    """Perform dedicated 1D threshold sweep for reranker candidates (P2-03).

    Calculates Macro Precision, Recall, F1, F2, chunk retrieval statistics,
    and identifies the threshold plateau.

    Returns:
        tuple: (sweep_df, plateau_info, recommended_config)
    """
    mapping = internal_to_official or {}
    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}
    num_queries = len(label_map)
    if num_queries == 0:
        raise ValueError("Cannot sweep with empty labels")

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

    # Determine thresholds grid
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

    sweep_records = []
    for th in thresholds:
        total_p, total_r, total_f1, total_f2 = 0.0, 0.0, 0.0, 0.0
        chunk_counts = []
        zero_queries = 0

        for qid, truth_chunks in label_map.items():
            cands = prepared_queries.get(qid, [])
            selected = select_ids(cands, "chunk_id", th, fallback, effective_max)
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

        sweep_records.append({
            "threshold": th,
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

    sweep_df = pd.DataFrame(sweep_records)
    # Sort: highest F2, highest Recall, highest Precision, then smaller chunk output
    sweep_df.sort_values(
        by=["macro_f2", "macro_recall", "macro_precision", "avg_chunks_per_query"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    sweep_df.reset_index(drop=True, inplace=True)

    plateau_info = detect_threshold_plateau(
        sweep_df,
        threshold_col="threshold",
        metric_col="macro_f2",
        tolerance=tolerance,
    )

    rec_th = plateau_info.get("recommended_threshold")
    if rec_th is not None:
        rec_match = sweep_df[sweep_df["threshold"] == rec_th]
        rec_row = rec_match.iloc[0] if not rec_match.empty else sweep_df.iloc[0]
    else:
        rec_row = sweep_df.iloc[0]

    recommended_config = {
        "threshold": float(rec_row["threshold"]) if rec_row["threshold"] is not None else None,
        "fallback": fallback,
        "max_chunks": max_chunks,
        "macro_f2": float(rec_row["macro_f2"]),
        "macro_f1": float(rec_row["macro_f1"]),
        "macro_precision": float(rec_row["macro_precision"]),
        "macro_recall": float(rec_row["macro_recall"]),
        "avg_chunks_per_query": float(rec_row["avg_chunks_per_query"]),
        "zero_chunk_queries": int(rec_row["zero_chunk_queries"]),
    }

    return sweep_df, plateau_info, recommended_config


def sweep_chunk_selector(
    scored_records,
    labels,
    thresholds=None,
    fallbacks=None,
    maximums=None,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """Perform grid search sweep over (threshold, fallback, max_chunk) to maximize Macro Chunk F2.

    Returns:
        tuple (best_config, sweep_df, plateau_info)
    """
    mapping = internal_to_official or {}
    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}

    # Pre-process candidates per query to avoid repeating official ID collapse
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

    # Auto-generate dynamic threshold search grid if not explicitly provided
    if thresholds is None:
        if all_scores:
            min_s, max_s = min(all_scores), max(all_scores)
            if min_s == max_s:
                thresholds = [None, min_s]
            else:
                step = (max_s - min_s) / 20.0
                thresholds = [None] + [round(min_s + i * step, 4) for i in range(1, 20)]
        else:
            thresholds = [None, 0.0, 0.5]

    fallbacks = fallbacks or [0, 1, 2, 3, 5]
    maximums = maximums or [3, 5, 10, 15, 20]

    sweep_records = []
    num_queries = len(label_map)
    if num_queries == 0:
        raise ValueError("Cannot sweep with empty labels")

    for th in thresholds:
        for fb in fallbacks:
            for mx in maximums:
                if fb > mx:
                    continue

                total_p, total_r, total_f1, total_f2 = 0.0, 0.0, 0.0, 0.0
                total_chunks = 0

                for qid, truth_chunks in label_map.items():
                    cands = prepared_queries.get(qid, [])
                    selected = select_ids(cands, "chunk_id", th, fb, mx)
                    total_chunks += len(selected)
                    metrics = classification_metrics(truth_chunks, set(selected), zero_division=0.0)
                    total_p += metrics["precision"]
                    total_r += metrics["recall"]
                    total_f1 += metrics["f1"]
                    total_f2 += metrics["f2"]

                macro_p = total_p / num_queries
                macro_r = total_r / num_queries
                macro_f1 = total_f1 / num_queries
                macro_f2 = total_f2 / num_queries
                avg_chunks = total_chunks / num_queries

                sweep_records.append({
                    "chunk_threshold": th,
                    "chunk_fallback": fb,
                    "chunk_max": mx,
                    "macro_f2": round(macro_f2, 5),
                    "macro_f1": round(macro_f1, 5),
                    "macro_precision": round(macro_p, 5),
                    "macro_recall": round(macro_r, 5),
                    "avg_chunks_per_query": round(avg_chunks, 2),
                })

    sweep_df = pd.DataFrame(sweep_records)
    # Sort: highest F2, highest Recall, highest Precision, then smaller chunk output
    sweep_df.sort_values(
        by=["macro_f2", "macro_recall", "macro_precision", "avg_chunks_per_query"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    sweep_df.reset_index(drop=True, inplace=True)

    best_row = sweep_df.iloc[0]
    best_config = {
        "chunk_threshold": None if pd.isna(best_row["chunk_threshold"]) else float(best_row["chunk_threshold"]),
        "chunk_fallback": int(best_row["chunk_fallback"]),
        "chunk_max": int(best_row["chunk_max"]),
        "macro_f2": float(best_row["macro_f2"]),
        "macro_f1": float(best_row["macro_f1"]),
        "macro_precision": float(best_row["macro_precision"]),
        "macro_recall": float(best_row["macro_recall"]),
        "avg_chunks_per_query": float(best_row["avg_chunks_per_query"]),
    }

    # Plateau detection: find threshold range within 1% of best F2 with same fallback/max
    same_limits = sweep_df[
        (sweep_df["chunk_fallback"] == best_config["chunk_fallback"]) &
        (sweep_df["chunk_max"] == best_config["chunk_max"])
    ]
    plateau_info = detect_threshold_plateau(
        same_limits,
        threshold_col="chunk_threshold",
        metric_col="macro_f2",
        tolerance=0.01,
    )

    return best_config, sweep_df, plateau_info


def analyze_false_negatives(
    scored_records,
    labels,
    best_config,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """Categorize False Negatives into P1 Retrieval Miss vs P2 Reranker/Threshold Pruned.

    Returns:
        tuple (fn_df, summary_stats)
    """
    mapping = internal_to_official or {}
    label_map = {row["id"]: list(dict.fromkeys(row.get("relevant_chunks", []))) for row in labels}

    th = best_config.get("chunk_threshold", best_config.get("threshold"))
    fb = best_config.get("chunk_fallback", best_config.get("fallback", 0))
    mx = best_config.get("chunk_max", best_config.get("max_chunks"))

    fn_entries = []
    category_counts = defaultdict(int)

    for row in scored_records:
        qid = row["id"]
        cands = row.get("candidates", [])
        if chunk_to_doc is not None:
            cands = official_candidates(cands, mapping, chunk_to_doc)
        elif mapping:
            cands = [{**c, "chunk_id": mapping.get(c["chunk_id"], c["chunk_id"])} for c in cands]

        cand_map = {c["chunk_id"]: c for c in cands}
        selected = set(select_ids(cands, "chunk_id", th, fb, mx))
        truth = label_map.get(qid, [])

        sorted_cands = sorted(cands, key=lambda c: (-candidate_score(c), c["chunk_id"]))
        cand_ranks = {c["chunk_id"]: rank + 1 for rank, c in enumerate(sorted_cands)}

        for chunk_id in truth:
            if chunk_id in selected:
                continue

            # This is a False Negative
            if chunk_id not in cand_map:
                category = "P1_RETRIEVAL_MISS"
                reason = "Chunk missing from P1 candidates pool"
                score = None
                rank = None
            else:
                score = candidate_score(cand_map[chunk_id])
                rank = cand_ranks.get(chunk_id)
                if th is not None and score < th:
                    category = "P2_THRESHOLD_PRUNED"
                    reason = f"Score {score:.4f} < threshold {th:.4f} (rank {rank})"
                else:
                    category = "P2_MAX_CAP_PRUNED"
                    reason = f"Rank {rank} exceeded max chunk limit {mx}"

            category_counts[category] += 1
            fn_entries.append({
                "query_id": qid,
                "chunk_id": chunk_id,
                "category": category,
                "rerank_score": score,
                "rerank_rank": rank,
                "reason": reason,
            })

    fn_df = pd.DataFrame(fn_entries)
    total_fns = len(fn_entries)
    summary_stats = {
        "total_false_negatives": total_fns,
        "p1_retrieval_miss": category_counts["P1_RETRIEVAL_MISS"],
        "p2_threshold_pruned": category_counts["P2_THRESHOLD_PRUNED"],
        "p2_max_cap_pruned": category_counts["P2_MAX_CAP_PRUNED"],
        "p1_miss_pct": round(category_counts["P1_RETRIEVAL_MISS"] / total_fns * 100, 1) if total_fns else 0.0,
        "p2_miss_pct": round((category_counts["P2_THRESHOLD_PRUNED"] + category_counts["P2_MAX_CAP_PRUNED"]) / total_fns * 100, 1) if total_fns else 0.0,
    }

    return fn_df, summary_stats


def generate_p2_chunk_predictions(
    scored_records,
    best_config,
    registry=None,
    internal_to_official=None,
    chunk_to_doc=None,
):
    """Generate selected chunk predictions per query mapped to official IDs.

    Returns:
        list of dicts: [{"id": query_id, "relevant_chunks": [chunk_ids]}]
    """
    if registry:
        if internal_to_official is None:
            internal_to_official = registry.get("internal_to_official")
        if chunk_to_doc is None:
            chunk_to_doc = registry.get("chunk_to_doc")

    mapping = internal_to_official or {}

    th = best_config.get("chunk_threshold", best_config.get("threshold"))
    fb = best_config.get("chunk_fallback", best_config.get("fallback", 0))
    mx = best_config.get("chunk_max", best_config.get("max_chunks"))

    predictions = []
    for row in scored_records:
        qid = row["id"]
        cands = row.get("candidates", [])
        if chunk_to_doc is not None:
            cands = official_candidates(cands, mapping, chunk_to_doc)
        elif mapping:
            cands = [{**c, "chunk_id": mapping.get(c["chunk_id"], c["chunk_id"])} for c in cands]

        selected = select_ids(cands, "chunk_id", th, fb, mx)
        predictions.append({
            "id": qid,
            "relevant_chunks": selected,
        })
    return predictions


def package_p2_to_p3_handoff(
    output_dir,
    source_run_dir,
    best_config,
    sweep_df,
    fn_df,
    benchmark_report=None,
    archive_zip=True,
):
    """Package complete P2 deliverables into an immutable handoff bundle for Member 3 (P3).

    Returns:
        Path to output directory
    """
    output = Path(output_dir).resolve()
    source = Path(source_run_dir).resolve()

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    # 1. Deliverable 1: Scored Candidates (reranked.jsonl)
    reranked_source = source / "reranked.jsonl"
    if not reranked_source.exists():
        raise FileNotFoundError(f"Missing {reranked_source} needed for P3 doc aggregation")
    shutil.copy2(reranked_source, output / "reranked.jsonl")

    # 2. Supporting context: registry, queries, labels
    for item in ("registry.json", "queries.json", "labels.json"):
        if (source / item).exists():
            shutil.copy2(source / item, output / item)

    registry = read_json(output / "registry.json") if (output / "registry.json").exists() else {}

    # 3. Deliverable 2: Best Chunk Selector Configuration (YAML)
    selector_export = {
        "chunk_selector": {
            "chunk_threshold": best_config["chunk_threshold"],
            "chunk_fallback": best_config["chunk_fallback"],
            "chunk_max": best_config["chunk_max"],
        },
        "metrics": {
            "macro_f2": best_config["macro_f2"],
            "macro_f1": best_config["macro_f1"],
            "macro_precision": best_config["macro_precision"],
            "macro_recall": best_config["macro_recall"],
            "avg_chunks_per_query": best_config["avg_chunks_per_query"],
        },
        "target_stage": "P2_chunk_selection",
        "ready_for_doc_aggregation": True,
    }
    with (output / "best_chunk_selector.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(selector_export, f, allow_unicode=True, sort_keys=False)

    # 4. Deliverable 3: Pre-selected chunks for all queries (official IDs validated)
    scored_records = load_records(output / "reranked.jsonl")
    chunk_preds = generate_p2_chunk_predictions(scored_records, best_config, registry)
    write_json(output / "p2_selected_chunks.json", chunk_preds)

    # Validate predictions if registry has doc_ids and chunk_to_doc
    if registry and "chunk_to_doc" in registry:
        val = SubmissionValidator(
            expected_query_ids=registry.get("expected_query_ids"),
            doc_ids=set(registry.get("doc_ids", [])),
            chunk_to_doc=registry.get("chunk_to_doc"),
            internal_to_official=registry.get("internal_to_official"),
        )
        for item in chunk_preds:
            unknown = set(item["relevant_chunks"]) - set(val.chunk_to_doc)
            if unknown:
                raise ValueError(f"Selected invalid chunk IDs for query {item['id']}: {unknown}")

    # 5. Deliverable 4 & 5: Sweep report CSV & False Negative analysis CSV
    sweep_df.to_csv(output / "threshold_sweep.csv", index=False)
    fn_df.to_csv(output / "fn_analysis.csv", index=False)

    # 5b. P2-02 detailed evaluation report if present in source
    for opt_report in ("p2_f2_evaluation_report.csv", "p2_f2_evaluation_report.json"):
        if (source / opt_report).exists():
            shutil.copy2(source / opt_report, output / opt_report)

    # 6. Benchmark report if available
    if benchmark_report is not None:
        write_json(output / "reranker_benchmark.json", benchmark_report)

    # 7. Deliverable 6: Manifest linking to P1 handoff provenance
    p1_manifest_file = source / "input_manifest.json"
    if not p1_manifest_file.exists():
        p1_manifest_file = source / "manifest.json"
    p1_manifest = read_json(p1_manifest_file) if p1_manifest_file.exists() else {}

    manifest_files = [
        "reranked.jsonl",
        "best_chunk_selector.yaml",
        "p2_selected_chunks.json",
        "threshold_sweep.csv",
        "fn_analysis.csv",
    ]
    for opt in (
        "registry.json",
        "queries.json",
        "labels.json",
        "reranker_benchmark.json",
        "p2_f2_evaluation_report.csv",
        "p2_f2_evaluation_report.json",
    ):
        if (output / opt).exists():
            manifest_files.append(opt)

    manifest = {
        "schema_version": "medical-rag-p2-handoff-v1",
        "stage": "p2_to_p3_handoff",
        "p1_source_run_id": p1_manifest.get("source_run_id"),
        "p1_manifest_provenance": p1_manifest.get("fingerprints"),
        "query_count": len(chunk_preds),
        "best_chunk_selector": best_config,
        "downstream_use_for_p3": [
            "P3-02 Doc Aggregation (consume reranked.jsonl)",
            "P3-04 Doc Selector (use independently of chunk selector)",
            "P3-05 Doc/Chunk Consistency (compare p2_selected_chunks.json with selected_docs)",
            "P3-06 Submission Generator (merge p2_selected_chunks.json with selected_docs)",
            "P3-15 Official Submission (use best_chunk_selector.yaml)",
        ],
        "files": {name: sha256_file(output / name) for name in manifest_files},
    }
    write_json(output / "manifest.json", manifest)

    if archive_zip:
        archive_path = output.parent / f"{output.name}.zip"
        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in manifest_files + ["manifest.json"]:
                zf.write(output / name, arcname=name)

    return output
