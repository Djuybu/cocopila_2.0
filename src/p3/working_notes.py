"""P3-17: assemble working-notes numbers from experiment artifacts.

Reads only existing P1/P2/P3 artifacts (read-only) and renders a filled copy of
``docs/working_notes_outline.md``. Missing results are marked explicitly instead
of being invented; numbers always trace back to a listed source file.
"""
from pathlib import Path

import pandas as pd

from src.p3.submission_log import read_log, summarize_log
from src.utils.io import read_json, write_json

CONTRIBUTORS = (
    ("P1", "Mai Ngọc Duy", "Data adapter, dedup, split, corpus/index, BM25, BGE-M3 dense, E5 comparison, union, RRF, candidate recall, P1→P2 handoff contract"),
    ("P2", "Mạc Duy", "Reranking (BGE/Qwen3), chunk F2 evaluation, threshold/fallback/max sweeps, training pairs, hard negatives, fine-tuning, CV threshold, FN analysis, P2→P3 handoff bundle"),
    ("P3", "Quế", "chunk→doc hierarchy, doc aggregation/ablation, doc selector, consistency, submission generator/validator/ZIP, graph schema/NER/normalization/graph store/GraphRetriever, label audit, official aggregation/submission, submission log, working notes"),
)
REQUIRED_FIGURES = (
    "Figure 1 — End-to-end pipeline (P1 retrieval → P2 reranking/selection → P3 aggregation/submission).",
    "Figure 2 — Candidate Recall@K before reranking (chunk and document).",
    "Figure 3 — Macro Chunk F2 vs threshold (with plateau/confidence interval).",
    "Figure 4 — Document aggregation ablation (aggregation × direct-doc).",
    "Figure 5 — Error composition: P1 retrieval miss vs P2 pruning vs P3 aggregation.",
)
REQUIRED_TABLES = (
    "Table 1 — Dataset audit (queries/documents/chunks, language, duplicates).",
    "Table 2 — Retriever comparison (BM25 / BGE-M3 / E5 / fusion) with Recall@K and latency.",
    "Table 3 — Reranker comparison and out-of-fold chunk selector score.",
    "Table 4 — Document aggregation + direct-doc ablation with holdout F2_doc.",
    "Table 5 — Graph ablation (graph only vs graph + hybrid).",
    "Table 6 — Submission log (run IDs, commits, local/public/private scores).",
)


def _optional_json(path):
    return read_json(path) if path and Path(path).exists() else None


def _optional_csv(path):
    return pd.read_csv(path) if path and Path(path).exists() else None


def _known(value, label="pending official run"):
    return value if value is not None and value != "" else f"n/a ({label})"


def collect_metrics(p2_run_dir=None, p3_doc_dir=None, submission_log=None):
    """Summarize available metrics; every value keeps its source path."""
    metrics = {"p2": {}, "p3": {}, "submissions": {}, "sources": []}
    if p2_run_dir:
        root = Path(p2_run_dir)
        cv_path, bench_path = root / "selector_cv_report.json", root / "reranker_benchmark.json"
        cv = _optional_json(cv_path)
        if cv:
            metrics["p2"]["selector_cv"] = {
                "status": cv.get("status"), "n_splits": cv.get("n_splits"),
                "query_count": cv.get("query_count"),
                "macro_f2": (cv.get("macro") or {}).get("f2"),
                "ci95": cv.get("confidence_interval_95")}
            metrics["sources"].append(str(cv_path))
        bench = _optional_json(bench_path)
        if bench:
            metrics["p2"]["reranker_benchmark"] = {
                "run_id": bench.get("run_id"),
                "before_macro_f2": ((bench.get("before") or {}).get("macro") or {}).get("f2"),
                "after_macro_f2": ((bench.get("after") or {}).get("macro") or {}).get("f2")}
            metrics["sources"].append(str(bench_path))
    if p3_doc_dir:
        root = Path(p3_doc_dir)
        pipeline_path = root / "doc_pipeline_report.json"
        pipeline = _optional_json(pipeline_path)
        if pipeline:
            best = pipeline.get("best_doc_pipeline", {})
            metrics["p3"]["doc_pipeline"] = {
                "ablation_method": best.get("ablation_method"),
                "direct_doc_weight": best.get("direct_doc_weight"),
                "holdout_mean_f2_doc": (best.get("holdout") or {}).get("mean_f2_doc"),
                "holdout_std_f2_doc": (best.get("holdout") or {}).get("std_f2_doc"),
                "full_data_f2_doc": best.get("full_data_f2_doc")}
            metrics["sources"].append(str(pipeline_path))
        for name, key in (("doc_pipeline_ablation.csv", "doc_pipeline_ablation"),
                          ("doc_aggregation_ablation.csv", "doc_aggregation_ablation")):
            table = _optional_csv(root / name)
            if table is not None:
                metrics["p3"][key] = table.to_dict("records")
                metrics["sources"].append(str(root / name))
    if submission_log and Path(submission_log).exists():
        metrics["submissions"] = summarize_log(read_log(submission_log))
        metrics["sources"].append(str(submission_log))
    return metrics


def render_working_notes(metrics):
    """Render the filled outline (markdown) from collected metrics."""
    p2, p3, submissions = metrics.get("p2", {}), metrics.get("p3", {}), metrics.get("submissions", {})
    cv = p2.get("selector_cv") or {}
    bench = p2.get("reranker_benchmark") or {}
    pipeline = p3.get("doc_pipeline") or {}
    lines = [
        "# Working notes — Medical Retrieval & RAG (P3-17)",
        "",
        "> Generated by `scripts/export_working_notes.py` from the artifacts listed at the end.",
        "> Values marked *n/a (pending official run)* must be filled after the official corpus and",
        "> the competition metric are confirmed. Do not report placeholder numbers as results.",
        "",
        "## 1. Problem and constraints",
        "",
        "Multilingual medical retrieval: return `relevant_docs` and `relevant_chunks` per query.",
        "Local objective is Macro F2 separately for the document and chunk branches; the exact",
        "competition rule must still be confirmed against the official scoring description.",
        "",
        "## 2. System architecture",
        "",
        "P1 candidate generation (BM25 + BGE-M3 dense, RRF/CC fusion) → P2 reranking and chunk",
        "selection → P3 hierarchy, document aggregation, submission, graph retrieval.",
        "See `Figure 1`.",
        "",
        "## 3. Data preparation and audit (P1)",
        "",
        "- Canonical adapter, dedup, document-disjoint split, chunk corpus and reverse mappings.",
        "- Prototype labels are weak adjacent-span signals, not official qrels.",
        "- Dataset audit (Table 1) comes from the P1-12 audit once official data is present.",
        "",
        "## 4. Retrieval and fusion (P1)",
        "",
        "Candidate Recall@K before reranking is the retrieval-quality metric (Figure 2); see",
        "`docs/p1_retrieval_results.md` for the measured values and their fingerprints.",
        "",
        "## 5. Reranking and chunk selection (P2)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Reranker Macro Chunk F2 before | {_known(bench.get('before_macro_f2'))} |",
        f"| Reranker Macro Chunk F2 after | {_known(bench.get('after_macro_f2'))} |",
        f"| Out-of-fold selector Macro Chunk F2 | {_known(cv.get('macro_f2'))} |",
        f"| Out-of-fold folds / queries | {_known(cv.get('n_splits'))} / {_known(cv.get('query_count'))} |",
        "",
        "In-sample tuning scores in `best_chunk_selector.yaml` are NOT generalisation scores;",
        "use the out-of-fold numbers above (Table 3).",
        "",
        "## 6. Document aggregation and submission (P3)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Best aggregation / direct-doc weight | {_known(pipeline.get('ablation_method'))} / {_known(pipeline.get('direct_doc_weight'))} |",
        f"| Holdout Macro F2_doc (out-of-fold) | {_known(pipeline.get('holdout_mean_f2_doc'))} ± {_known(pipeline.get('holdout_std_f2_doc'))} |",
        f"| Full-data Macro F2_doc | {_known(pipeline.get('full_data_f2_doc'))} |",
        f"| Public / private submissions logged | {_known(submissions.get('public_count'))} / {_known(submissions.get('private_count'))} |",
        f"| Best local score | {_known(submissions.get('best_local_score'))} |",
        "",
        "Document and chunk branches are tuned and selected independently; consistency between",
        "them is measured, not silently enforced (P3-05). See Table 4 and Table 6.",
        "",
        "## 7. Metrics and evaluation protocol",
        "",
        "- Set metrics per query: Precision, Recall, F1, F2 (F2 = 5·TP / (4·|truth| + |prediction|)).",
        "- Macro = arithmetic mean of per-query metrics, computed separately per branch.",
        "- Recall@K uses the first K unique official chunks as the shared document budget.",
        "- Empty denominators use an explicit `zero_division`, default 0.",
        "",
        "## 8. Error analysis",
        "",
        "- P1 retrieval miss vs P2 pruning (`fn_analysis.csv`).",
        "- Label hierarchy audit: does chunk-relevant imply parent-doc-relevant? (P3-13).",
        "- Graph retrieval ablation: graph only vs graph + hybrid (P3-12).",
        "",
        "## 9. Ablations",
        "",
        "- Aggregation: max / mean(top2) / mean(top3) / weighted × direct-doc on/off (P3-03, P3-14).",
        "- Selector: threshold / fallback / max grid with out-of-fold evaluation (P2-09, P3-04).",
        "- Graph: independent source keep/drop (P3-12).",
        "",
        "## 10. Reproducibility",
        "",
        "Every run records `run_id`, git commit, config and artifact checksums. Metric sources:",
        f"{', '.join(metrics.get('sources', [])) or 'n/a (no artifacts supplied)'}.",
        "",
        "## 11. Member contributions",
        "",
        "| Member | Tasks | Contribution |",
        "| --- | --- | --- |",
    ]
    for code, name, contribution in CONTRIBUTORS:
        lines.append(f"| {code} — {name} | {code}-01…{code}-17 | {contribution} |")
    lines += [
        "",
        "## 12. Limitations and pending work",
        "",
        "- Official corpus/labels (P1-12) and the official metric definition are not in this repository.",
        "- The graph lexicon is a minimal seed, not a medical ontology; no clinical claims.",
        "- Model benchmarks that require GPU/weights were not executed here.",
        "",
        "## Required figures",
        "",
    ]
    lines += [f"- {figure}" for figure in REQUIRED_FIGURES]
    lines += ["", "## Required tables", ""]
    lines += [f"- {table}" for table in REQUIRED_TABLES]
    if metrics.get("sources"):
        lines += ["", "## Metric sources", ""] + [f"- `{source}`" for source in metrics["sources"]]
    lines.append("")
    return "\n".join(lines)


def export_working_notes(output_dir, *, p2_run_dir=None, p3_doc_dir=None, submission_log=None):
    """Write the filled outline + metrics JSON (exclusive creation)."""
    output = Path(output_dir)
    outline_path, metrics_path = output / "working_notes_outline.md", output / "metrics_summary.json"
    existing = [str(path) for path in (outline_path, metrics_path) if path.exists()]
    if existing:
        raise FileExistsError("Working notes output already exists: " + ", ".join(existing))
    metrics = collect_metrics(p2_run_dir, p3_doc_dir, submission_log)
    output.mkdir(parents=True, exist_ok=True)
    outline_path.write_text(render_working_notes(metrics), encoding="utf-8")
    write_json(metrics_path, metrics)
    return {"outline": str(outline_path), "metrics": str(metrics_path)}
