"""P1 data contracts: deduplication, source reversibility and grouped splits."""
import io
import zipfile
import pytest

from src.data.prototype import adapt_triplets, write_prototype
from src.data.split import group_split
from src.data.mmedc import prepare_mmedc, _sample_archive


def triplet(anchor, article, positive, negatives=()):
    return {"anchor": anchor, "positive": positive,
            "meta": {"article_id": article, "title": article}, "negatives": list(negatives)}


def test_triplet_dedup_merges_negatives_and_preserves_sources():
    rows = [triplet("same query", "a", "same positive", ["negative one"]),
            triplet("same  query", "a", "same  positive", ["negative two", "same positive"])]
    bundle = adapt_triplets(rows)
    assert len(bundle["queries"]) == 1
    assert len(bundle["chunks"]) == 3
    assert len(bundle["labels"][0]["relevant_chunks"]) == 1
    assert len(bundle["queries"][0]["negative_chunk_ids"]) == 2
    report = bundle["dedup_report"]
    assert report["duplicate_query_positive_pairs_removed"] == 1
    assert report["contradictory_negatives_removed"] == 1
    assert report["query_negative_pairs_after"] == 2
    mapping = bundle["source_mappings"]["queries"]
    assert {ref["row_index"] for refs in mapping.values() for ref in refs} == {0, 1}
    assert all(row["id_namespace"] == "prototype" for row in bundle["chunks"])


def test_positive_without_document_identity_is_rejected():
    with pytest.raises(ValueError, match="Positive requires"):
        adapt_triplets([{"anchor": "question", "positive": "passage"}])


def test_structured_anchor_and_passage_keep_original_ids():
    bundle = adapt_triplets([{"id": "row1", "anchor": {"id": "original-q", "text": "question"},
                             "positive": {"id": "original-c", "text": "passage",
                                          "metadata": {"article_id": "article1", "title": "title"}}}])
    query_refs = next(iter(bundle["source_mappings"]["queries"].values()))
    chunk_refs = next(iter(bundle["source_mappings"]["chunks"].values()))
    assert query_refs[0]["source_query_id"] == "original-q"
    assert query_refs[0]["source_row_id"] == "row1"
    assert chunk_refs[0]["source_chunk_id"] == "original-c"


def test_group_split_connects_multiple_positives_and_duplicate_content():
    bundle = adapt_triplets([triplet(f"query{i}", f"doc{i}", f"text{i}") for i in range(8)])
    labels = bundle["labels"]
    labels[0]["relevant_docs"] += labels[1]["relevant_docs"]
    labels[0]["relevant_chunks"] += labels[1]["relevant_chunks"]
    chunks = bundle["chunks"]
    chunks[2]["text"] = chunks[3]["text"]
    ratios = {"train": .6, "val": .2, "test": .2}
    args = [bundle[key] for key in ("documents", "chunks", "queries", "labels")]
    split = group_split(*args, ratios=ratios, seed=42)
    assert split == group_split(*args, ratios=ratios, seed=42)
    for label in labels:
        assert {split["documents"][d] for d in label["relevant_docs"]} == {split["queries"][label["id"]]}
    assert split["documents"][chunks[2]["doc_id"]] == split["documents"][chunks[3]["doc_id"]]
    assert all(value["queries"] > 0 for value in split["counts"].values())


def test_shared_titles_never_cross_split():
    bundle = adapt_triplets([triplet(f"query{i}", f"doc{i}", f"text{i}") for i in range(5)])
    docs = bundle["documents"]
    docs[0]["title"] = docs[1]["title"] = "same title"
    split = group_split(*[bundle[k] for k in ("documents", "chunks", "queries", "labels")],
                        ratios={"train": .6, "val": .2, "test": .2}, seed=7)
    assert split["documents"][docs[0]["doc_id"]] == split["documents"][docs[1]["doc_id"]]


def test_write_prototype_is_exclusive_and_outputs_mappings(tmp_path):
    bundle = adapt_triplets([triplet(f"q{i}", f"d{i}", f"p{i}") for i in range(5)])
    cfg = {"output_dir": tmp_path / "processed", "mappings_dir": tmp_path / "mappings",
           "raw_dir": tmp_path / "raw", "seed": 42, "split_ratios": {"train": .6, "val": .2, "test": .2}}
    # YAML configs are strings; reproduce that public input shape.
    cfg = {key: str(value) if isinstance(value, type(tmp_path)) else value for key, value in cfg.items()}
    write_prototype(bundle, cfg)
    assert (tmp_path / "processed/chunk_corpus.jsonl").is_file()
    assert (tmp_path / "mappings/source_ids.json").is_file()
    with pytest.raises(FileExistsError):
        write_prototype(bundle, cfg)


def test_mmedc_requires_explicit_weak_label_consent():
    with pytest.raises(ValueError, match="no retrieval qrels"):
        prepare_mmedc({"data": {}})


def test_mmedc_zip_sample_is_bounded_and_reproducible():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("medical/a.txt", "medical text " * 1000)
        archive.writestr("medical/b.txt", "different text " * 1000)
        archive.writestr("__MACOSX/no.txt", "excluded " * 1000)
    cfg = {"seed": 42, "max_documents_per_language": 1, "max_chars_per_document": 100,
           "query_chars": 10, "chunk_chars": 40}
    with zipfile.ZipFile(stream) as archive:
        rows, available = _sample_archive(archive, "english", cfg, {"archive": "English.zip"})
        assert (rows, available) == _sample_archive(archive, "english", cfg, {"archive": "English.zip"})
    assert available == 2 and len(rows) == 1
    assert len(rows[0]["text"]) == 100
    assert rows[0]["source"]["truncated"] is True


def test_mmedc_local_preparation_excludes_anchor_spans(tmp_path, monkeypatch):
    import src.data.mmedc as module
    from src.data.download import sha256_file
    archives = tmp_path / "raw"
    archives.mkdir()
    with zipfile.ZipFile(archives / "English.zip", "w") as archive:
        for i in range(4):
            archive.writestr(f"doc{i}.txt", f"question{i}!" + f"passage{i} " * 50)
    zip_path = archives / "English.zip"
    monkeypatch.setattr(module, "_json_url", lambda url: {"sha": "pinned"} if "/revision/" in url else
                        [{"path": "English.zip", "size": zip_path.stat().st_size,
                          "lfs": {"oid": sha256_file(zip_path)}}])
    cfg = {"weak_labels_approved": True, "languages": ["english"], "repo": "Henrychur/MMedC",
           "revision": "pinned", "archives_dir": str(archives), "raw_dir": str(archives),
           "output_dir": str(tmp_path / "processed"), "mappings_dir": str(tmp_path / "mappings"),
           "seed": 42, "max_documents_per_language": 4, "max_chars_per_document": 100,
           "query_chars": 10, "chunk_chars": 40, "overlap_chars": 10,
           "split_ratios": {"train": .5, "val": .25, "test": .25}}
    bundle = prepare_mmedc({"data": cfg})
    assert bundle["label_quality"] == "weak_adjacent_span"
    assert len(bundle["queries"]) == 4
    assert all(c["metadata"]["span_start"] >= 10 for c in bundle["chunks"])
    assert all("question" not in c["text"] for c in bundle["chunks"])
    assert all(len(label["relevant_chunks"]) == 1 for label in bundle["labels"])
