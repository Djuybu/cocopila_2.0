"""P3-01: chunk -> document hierarchy contract and reconciliation tests."""
import pytest

from src.p3.hierarchy import (
    build_hierarchy,
    load_registry,
    reconcile_with_registry,
    run_hierarchy,
    write_hierarchy,
)
from src.utils.io import read_json, write_json


def corpus():
    documents = [{"doc_id": "d1"}, {"doc_id": "d2"}]
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "aspirin"},
        {"chunk_id": "i2", "official_chunk_id": "c2", "doc_id": "d2", "text": "insulin"},
        {"chunk_id": "c3", "doc_id": "d2", "text": "cardiac"},
    ]
    return documents, chunks


def hierarchy():
    return build_hierarchy(*corpus())


def test_build_hierarchy_official_mapping_and_provenance():
    result = build_hierarchy(*corpus())
    assert result["chunk_to_doc"] == {"c1": "d1", "c2": "d2", "c3": "d2"}
    assert result["internal_to_official"] == {"c1": "c1", "i2": "c2", "c3": "c3"}
    report = result["report"]
    assert report["document_count"] == 2
    assert report["official_chunk_count"] == 3
    assert report["internal_rechunk_count"] == 1
    assert report["orphan_chunk_count"] == 0
    assert report["multi_parent_chunk_count"] == 0


@pytest.mark.parametrize("mutation", ["orphan", "multi_parent", "duplicate_chunk",
                                      "duplicate_doc", "no_docs", "bad_doc", "empty_chunk_id"])
def test_build_hierarchy_rejects_invalid_corpus(mutation):
    documents, chunks = corpus()
    if mutation == "orphan":
        chunks.append({"chunk_id": "c4", "doc_id": "dx", "text": "x"})
    elif mutation == "multi_parent":
        chunks.append({"chunk_id": "i9", "official_chunk_id": "c2", "doc_id": "d1", "text": "x"})
    elif mutation == "duplicate_chunk":
        chunks.append({"chunk_id": "c1", "doc_id": "d1", "text": "x"})
    elif mutation == "duplicate_doc":
        documents.append({"doc_id": "d1"})
    elif mutation == "no_docs":
        documents.clear()
    elif mutation == "bad_doc":
        chunks[0]["doc_id"] = None
    else:
        chunks[0]["chunk_id"] = ""
    with pytest.raises(ValueError):
        build_hierarchy(documents, chunks)


def test_internal_id_cannot_collide_with_another_official_id():
    documents, chunks = corpus()
    chunks.append({"chunk_id": "c2", "official_chunk_id": "c9", "doc_id": "d1", "text": "x"})
    with pytest.raises(ValueError):
        build_hierarchy(documents, chunks)


def test_reconcile_accepts_matching_registry():
    registry = {"chunk_to_doc": {"c1": "d1", "c2": "d2", "c3": "d2"},
                "internal_to_official": {"c1": "c1", "i2": "c2", "c3": "c3"}}
    report = reconcile_with_registry(hierarchy(), registry)
    assert report["ok"] is True
    assert report["parent_mismatches"] == []
    assert report["missing_from_registry"] == []


def test_reconcile_rejects_parent_mismatch_and_missing_chunks():
    with pytest.raises(ValueError):
        reconcile_with_registry(hierarchy(), {"chunk_to_doc": {"c1": "d1", "c2": "d1", "c3": "d2"}})
    with pytest.raises(ValueError):
        reconcile_with_registry(hierarchy(), {"chunk_to_doc": {"c1": "d1"}})


def test_reconcile_superset_policy():
    registry = {"chunk_to_doc": {"c1": "d1", "c2": "d2", "c3": "d2", "c9": "d9"}}
    report = reconcile_with_registry(hierarchy(), registry, strict=False)
    assert report["ok"] is False
    assert report["registry_only_chunk_ids"] == ["c9"]
    with pytest.raises(ValueError):
        reconcile_with_registry(hierarchy(), registry)
    allowed = reconcile_with_registry(hierarchy(), registry, allow_registry_superset=True)
    assert allowed["ok"] is True and allowed["registry_only_chunk_ids"] == ["c9"]


def test_reconcile_detects_internal_id_conflict():
    registry = {"chunk_to_doc": {"c1": "d1", "c2": "d2", "c3": "d2"},
                "internal_to_official": {"i2": "c1"}}
    with pytest.raises(ValueError):
        reconcile_with_registry(hierarchy(), registry)
    report = reconcile_with_registry(hierarchy(), registry, strict=False)
    assert report["internal_id_conflicts"][0]["internal_chunk_id"] == "i2"


def test_write_hierarchy_is_exclusive(tmp_path):
    output = tmp_path / "hier"
    result = write_hierarchy(output, hierarchy())
    assert read_json(output / "chunk_to_doc.json") == {"c1": "d1", "c2": "d2", "c3": "d2"}
    assert read_json(output / "internal_to_official.json") == {"c1": "c1", "i2": "c2", "c3": "c3"}
    assert read_json(result["report"])["orphan_chunk_count"] == 0
    with pytest.raises(FileExistsError):
        write_hierarchy(output, hierarchy())


def test_run_hierarchy_end_to_end(tmp_path):
    documents, chunks = corpus()
    documents_path, chunks_path = tmp_path / "documents.json", tmp_path / "chunks.json"
    write_json(documents_path, documents)
    write_json(chunks_path, chunks)
    handoff = tmp_path / "handoff"
    write_json(handoff / "registry.json", {"chunk_to_doc": {"c1": "d1", "c2": "d2", "c3": "d2"}})
    output = tmp_path / "hier"
    result = run_hierarchy(documents_path, chunks_path, output, handoff)
    assert read_json(output / "chunk_to_doc.json") == {"c1": "d1", "c2": "d2", "c3": "d2"}
    assert read_json(result["report"])["registry_reconciliation"]["ok"] is True


def test_run_hierarchy_fails_before_writing_on_mismatch(tmp_path):
    documents, chunks = corpus()
    documents_path, chunks_path = tmp_path / "documents.json", tmp_path / "chunks.json"
    write_json(documents_path, documents)
    write_json(chunks_path, chunks)
    handoff = tmp_path / "handoff"
    write_json(handoff / "registry.json", {"chunk_to_doc": {"c1": "d1", "c2": "d1", "c3": "d2"}})
    output = tmp_path / "hier"
    with pytest.raises(ValueError):
        run_hierarchy(documents_path, chunks_path, output, handoff)
    assert not (output / "chunk_to_doc.json").exists()


def test_load_registry_requires_usable_mapping(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path)
    write_json(tmp_path / "registry.json", {"chunk_to_doc": {}})
    with pytest.raises(ValueError):
        load_registry(tmp_path)

