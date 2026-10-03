"""P3-13: label hierarchy audit tests."""
import pytest

from src.p3.label_audit import (
    LABEL_AUDIT_COLUMNS,
    audit_label_hierarchy,
    render_markdown,
    summarize_label_audit,
    write_label_audit,
)

MAPPING = {"c1": "d1", "c2": "d2", "c3": "d1"}


def labels():
    return [
        {"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]},
        {"id": "q2", "relevant_docs": ["d1"], "relevant_chunks": ["c2"]},
        {"id": "q3", "relevant_docs": ["d2"], "relevant_chunks": []},
    ]


def test_audit_measures_both_relations():
    table = audit_label_hierarchy(labels(), MAPPING)
    assert list(table.columns) == list(LABEL_AUDIT_COLUMNS)
    report = summarize_label_audit(table)
    assert report["chunk_implies_doc_violations"] == 1
    assert report["docs_without_labeled_chunk"] == 2
    assert report["chunk_implies_parent_doc"] is False
    assert report["doc_requires_labeled_chunk"] is False
    assert report["recommended_policy"] == "add_parent_docs"
    assert report["use_hierarchical_constraint"] is True


def test_pending_when_no_labeled_chunks():
    table = audit_label_hierarchy([{"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": []}], MAPPING)
    report = summarize_label_audit(table)
    assert report["chunk_implies_parent_doc"] is None
    assert report["recommended_policy"] == "pending_official_labels"
    assert report["use_hierarchical_constraint"] is False


def test_consistent_labels_recommend_report_only():
    table = audit_label_hierarchy([{"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]}], MAPPING)
    report = summarize_label_audit(table)
    assert report["chunk_implies_parent_doc"] is True
    assert report["doc_requires_labeled_chunk"] is True
    assert report["recommended_policy"] == "report_only"


def test_write_label_audit_outputs_markdown_csv_json(tmp_path):
    output = tmp_path / "audit"
    report = write_label_audit(output, labels(), MAPPING)
    text = render_markdown(report, audit_label_hierarchy(labels(), MAPPING))
    written = (output / "label_hierarchy_report.md").read_text(encoding="utf-8")
    assert "Label hierarchy audit" in written and "Recommended policy" in text
    assert (output / "label_hierarchy_report.csv").exists()
    assert (output / "label_hierarchy_report.json").exists()
    assert report["query_count"] == 3
    with pytest.raises(FileExistsError):
        write_label_audit(output, labels(), MAPPING)


def test_audit_rejects_unknown_parent_and_empty_mapping():
    table = audit_label_hierarchy([{"id": "q", "relevant_docs": ["d1"], "relevant_chunks": ["cX"]}], MAPPING)
    assert table.loc[0, "unknown_parent_chunks_count"] == 1
    with pytest.raises(ValueError):
        audit_label_hierarchy(labels(), {})
    with pytest.raises(ValueError):
        summarize_label_audit(table.iloc[0:0])
