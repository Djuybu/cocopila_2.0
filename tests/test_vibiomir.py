"""Official identity and URL-catalogue limitations, with small test fixtures."""
import hashlib

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.data.vibiomir import analyze_catalogue, study_catalogue
from src.pipeline.retrieve import load_dataset
from src.utils.io import read_json


def catalogue(tmp_path, *, doc_ids=(2, 4, 5), query_ids=(11, 12)):
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(pa.table({"id": list(query_ids), "query": ["Đau đầu?", "Sốt kéo dài?"]}), raw / "query.parquet")
    pq.write_table(pa.table({"id": list(doc_ids), "url": ["https://example.org/a", "https://example.org/a", "ftp://example.org/b"]}), raw / "links_corpus.parquet")
    files = {p.name: {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in raw.glob("*.parquet")}
    return {"raw_dir": str(raw), "output_dir": str(tmp_path / "prepared"),
            "report_dir": str(tmp_path / "report"), "repo": "AIGuruTinix/ViBioMIR",
            "revision": "pinned-test-revision", "files": files}


def test_catalogue_preserves_numeric_identity_and_does_not_invent_labels(tmp_path):
    cfg = catalogue(tmp_path)
    report = study_catalogue(cfg)
    queries = read_json(tmp_path / "prepared/queries.json")
    assert queries[0] == {"id": "11", "text": "Đau đầu?", "official_id": 11}
    assert report["corpus"]["duplicate_url_rows"] == 1
    assert report["corpus"]["missing_ids_within_range"] == 1
    assert report["corpus"]["invalid_http_urls"] == 1
    assert report["availability"]["relevance_labels"] is False
    assert report["availability"]["document_text"] is False
    assert not (tmp_path / "prepared/labels.json").exists()
    assert not (tmp_path / "prepared/chunks.json").exists()
    with pytest.raises(ValueError, match="URL catalogue only"):
        load_dataset({"data": {"manifest_path": str(tmp_path / "prepared/dataset_manifest.json"),
                              "documents_path": str(tmp_path / "prepared/documents.jsonl"),
                              "chunks_path": str(tmp_path / "prepared/chunks.jsonl")}})
    with pytest.raises(FileExistsError):
        study_catalogue(cfg)


@pytest.mark.parametrize("doc_ids,query_ids", [((2, 2, 5), (11, 12)), ((2, 4, 5), (11, 11))])
def test_duplicate_official_ids_rejected(tmp_path, doc_ids, query_ids):
    cfg = catalogue(tmp_path, doc_ids=doc_ids, query_ids=query_ids)
    with pytest.raises(ValueError, match="IDs|ID"):
        study_catalogue(cfg)
    assert not (tmp_path / "prepared/queries.json").exists()


def test_corrupted_official_file_rejected_before_export(tmp_path):
    cfg = catalogue(tmp_path)
    with (tmp_path / "raw/query.parquet").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="SHA256"):
        study_catalogue(cfg)
    assert not (tmp_path / "prepared").exists()


def test_text_corpus_is_not_silently_treated_as_official_url_catalogue(tmp_path):
    cfg = catalogue(tmp_path)
    pq.write_table(pa.table({"id": [1], "text": ["content"]}), tmp_path / "raw/links_corpus.parquet")
    with pytest.raises(ValueError, match="columns"):
        analyze_catalogue(tmp_path / "raw/query.parquet", tmp_path / "raw/links_corpus.parquet")
