"""P3-13: audit the ground-truth chunk/document hierarchy relation.

Measures whether ``relevant_chunk`` always implies its parent ``relevant_doc``
and whether a ``relevant_doc`` must have at least one labeled chunk. The
official labels (P1-12) are not in the repository yet, so the report records the
decision as pending until it runs on official data.
"""
from pathlib import Path
import json

import pandas as pd

LABEL_AUDIT_COLUMNS = (
    "query_id", "truth_doc_count", "truth_chunk_count", "unknown_parent_chunks_count",
    "chunk_implies_doc_violations_count", "docs_without_labeled_chunk_count",
    "chunk_implies_doc_violations", "docs_without_labeled_chunk", "consistent",
)
PENDING = "pending_official_labels"


def _unique(values):
    return list(dict.fromkeys(values))


def audit_label_hierarchy(labels, chunk_to_doc):
    """One row per query; lists the exact violating chunk/doc IDs."""
    if not isinstance(chunk_to_doc, dict) or not chunk_to_doc:
        raise ValueError("A non-empty chunk_to_doc mapping is required")
    rows = []
    for row in labels:
        query_id = row.get("id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"Label needs a nonempty id: {row!r}")
        docs = _unique(row.get("relevant_docs", []))
        chunks = _unique(row.get("relevant_chunks", []))
        docs_set = set(docs)
        unknown = [chunk for chunk in chunks if chunk not in chunk_to_doc]
        violations = [chunk for chunk in chunks
                      if chunk in chunk_to_doc and chunk_to_doc[chunk] not in docs_set]
        parents = {chunk_to_doc[chunk] for chunk in chunks if chunk in chunk_to_doc}
        docs_without_chunk = [doc_id for doc_id in docs if doc_id not in parents]
        rows.append({
            "query_id": query_id,
            "truth_doc_count": len(docs),
            "truth_chunk_count": len(chunks),
            "unknown_parent_chunks_count": len(unknown),
            "chunk_implies_doc_violations_count": len(violations),
            "docs_without_labeled_chunk_count": len(docs_without_chunk),
            "chunk_implies_doc_violations": violations,
            "docs_without_labeled_chunk": docs_without_chunk,
            "consistent": not unknown and not violations and not docs_without_chunk,
        })
    return pd.DataFrame(rows, columns=list(LABEL_AUDIT_COLUMNS))


def _decision(chunk_implies_parent_doc, doc_requires_labeled_chunk):
    if chunk_implies_parent_doc is None:
        return (PENDING, "No labeled chunks in the supplied labels; wait for official labels (P1-12).")
    if not chunk_implies_parent_doc:
        return ("add_parent_docs",
                "Some relevant chunks have a non-relevant parent document, so chunk => parent doc "
                "does NOT hold. Pruning chunks by selected documents would drop relevant chunks; "
                "prefer adding parent documents of selected chunks.")
    if not doc_requires_labeled_chunk:
        return ("prune_chunks",
                "Every relevant chunk implies its parent relevant document, but a relevant document "
                "may have no labeled chunk. Pruning chunks by selected documents is safe; pruning "
                "documents by selected chunks is NOT.")
    return ("report_only",
            "Chunk => parent document and document => labeled chunk both hold. Either hierarchy "
            "constraint is safe; report-only stays the default until the competition rule is confirmed.")


def summarize_label_audit(report_df):
    """Aggregate the two hierarchy relations into a decision (P3-13)."""
    if report_df.empty:
        raise ValueError("Empty label audit report")
    chunk_count = int(report_df["truth_chunk_count"].sum())
    doc_count = int(report_df["truth_doc_count"].sum())
    violations = int(report_df["chunk_implies_doc_violations_count"].sum())
    unknown = int(report_df["unknown_parent_chunks_count"].sum())
    docs_without_chunk = int(report_df["docs_without_labeled_chunk_count"].sum())
    chunk_implies_parent_doc = None if chunk_count == 0 else violations == 0
    doc_requires_labeled_chunk = None if doc_count == 0 else docs_without_chunk == 0
    recommended_policy, rationale = _decision(chunk_implies_parent_doc, doc_requires_labeled_chunk)
    return {
        "query_count": int(len(report_df)),
        "queries_consistent": int(report_df["consistent"].sum()),
        "queries_inconsistent": int((~report_df["consistent"]).sum()),
        "truth_chunk_count": chunk_count,
        "truth_doc_count": doc_count,
        "unknown_parent_chunks": unknown,
        "chunk_implies_doc_violations": violations,
        "docs_without_labeled_chunk": docs_without_chunk,
        "chunk_implies_parent_doc": chunk_implies_parent_doc,
        "doc_requires_labeled_chunk": doc_requires_labeled_chunk,
        "use_hierarchical_constraint": False if chunk_implies_parent_doc is None else True,
        "recommended_policy": recommended_policy,
        "rationale": rationale,
        "label_source": "supplied labels (not verified as official)",
    }


def render_markdown(report, table):
    """Render ``label_hierarchy_report.md`` content from the summary and table."""
    def flag(value):
        return "n/a" if value is None else ("yes" if value else "no")

    lines = [
        "# P3-13 — Label hierarchy audit (chunk vs document)",
        "",
        "> Generated by `scripts/audit_label_hierarchy.py`. The supplied labels are the weak",
        "> prototype labels unless official labels (P1-12) are provided; no hierarchy rule is",
        "> hard-coded into the pipeline.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Queries | {report['query_count']} |",
        f"| Consistent queries | {report['queries_consistent']} |",
        f"| Inconsistent queries | {report['queries_inconsistent']} |",
        f"| Labeled chunks | {report['truth_chunk_count']} |",
        f"| Labeled documents | {report['truth_doc_count']} |",
        f"| Unknown-parent chunks | {report['unknown_parent_chunks']} |",
        f"| chunk => parent doc violations | {report['chunk_implies_doc_violations']} |",
        f"| Relevant docs without labeled chunk | {report['docs_without_labeled_chunk']} |",
        f"| chunk => parent doc holds | {flag(report['chunk_implies_parent_doc'])} |",
        f"| doc => labeled chunk holds | {flag(report['doc_requires_labeled_chunk'])} |",
        "",
        "## Decision",
        "",
        f"- Use hierarchical constraint: **{flag(report['use_hierarchical_constraint'])}**",
        f"- Recommended policy: `{report['recommended_policy']}`",
        f"- Rationale: {report['rationale']}",
        f"- Label source: {report['label_source']}",
        "",
        "## Per-query detail",
        "",
        "| Query | Chunks | Docs | Unknown parent | chunk=>doc violations | Docs w/o chunk | Consistent |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"| {row['query_id']} | {row['truth_chunk_count']} | {row['truth_doc_count']} | "
            f"{row['unknown_parent_chunks_count']} | {row['chunk_implies_doc_violations_count']} | "
            f"{row['docs_without_labeled_chunk_count']} | {'yes' if row['consistent'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_label_audit(output_dir, labels, chunk_to_doc):
    """Write ``label_hierarchy_report.md`` + ``.csv`` + ``.json`` (exclusive creation)."""
    output = Path(output_dir)
    targets = [output / "label_hierarchy_report.md", output / "label_hierarchy_report.csv",
               output / "label_hierarchy_report.json"]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Label audit output already exists: " + ", ".join(existing))
    table = audit_label_hierarchy(labels, chunk_to_doc)
    report = summarize_label_audit(table)
    output.mkdir(parents=True, exist_ok=True)
    (output / "label_hierarchy_report.md").write_text(render_markdown(report, table), encoding="utf-8")
    table.to_csv(output / "label_hierarchy_report.csv", index=False)
    (output / "label_hierarchy_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
