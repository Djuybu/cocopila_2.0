import hashlib
import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("trafilatura")

from src.data.rag_dataset import capacity_report, crawl, export, source_identity, stratified_sample
from src.data.web_corpus import WebFetcher, acquire, chunk_document, encoded, extract_document, open_state
from src.pipeline.retrieve import build_bm25_index, load_dataset, run_retrieval
from src.pipeline.reporting import make_submission
from src.utils.config import load_config


@pytest.fixture
def settings():
    return {"seed": 42, "sample_per_host": 3, "workers": 2, "timeout_seconds": 2,
            "max_response_bytes": 4096, "min_text_chars": 20, "delay_per_host_seconds": 0,
            "prefer_https": False, "user_agent": "TestRAG/1", "chunk_chars": 120,
            "chunk_overlap": 20, "keep_raw": False, "bootstrap_replicates": 200,
            "embedding_dimension": 1024}


@pytest.fixture
def config(tmp_path, settings):
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(pa.table({"id": [7, 900], "url": ["https://one.example/a", "https://two.example/b"]}), raw / "links_corpus.parquet")
    pq.write_table(pa.table({"id": [31], "query": ["Truy vấn y khoa"]}), raw / "query.parquet")
    return {"rag": settings, "data": {"repo": "official/test", "revision": "pin", "raw_dir": str(raw),
            "files": {p.name: {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in raw.glob("*.parquet")}}}


def document(id=7, text=None):
    text = text or "Nội dung tiếng Việt. 中文医疗信息。\n" * 30
    return {"doc_id": str(id), "official_id": id, "url": f"https://one.example/{id}", "title": "Medical",
            "text": text, "metadata": {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "language": "vi"}}


def successful_result(id=7, host="one.example", **kwargs):
    d = document(id)
    return {"id": id, "url": d["url"], "host": host, "status": "ok", "document": d,
            "wire_bytes": 1000, "response_body_bytes": 1000, "raw_gzip_bytes": 100,
            "text_bytes": len(d["text"].encode()), "document_jsonl_bytes": 2000,
            "chunks_jsonl_bytes": 2500, "chunk_count": 10, **kwargs}


def test_reservoir_covers_each_host_uniformly_and_reproducibly():
    rows = [{"id": i, "url": "https://large.example/a", "host": "large.example"} for i in range(1000)]
    rows += [{"id": 9999, "url": "https://rare.example/a", "host": "rare.example"}]
    counts, sample = stratified_sample(iter(rows), 5, 42)
    assert counts == {"large.example": 1000, "rare.example": 1}
    assert len(sample) == 6 and sample[-1]["id"] == 9999
    assert (counts, sample) == stratified_sample(iter(rows), 5, 42)
    assert any(r["id"] > 100 for r in sample if r["host"] == "large.example")


def test_estimate_weights_population_not_equal_domain_samples(settings):
    rows = [successful_result(host="large.example", text_bytes=100) for _ in range(3)]
    rows += [{"id": 999, "host": "rare.example", "status": "http_error"}]
    report = capacity_report({"large.example": 1000, "rare.example": 1}, rows, settings)
    assert report["metrics_under_current_fetch_policy"]["text_bytes"]["expected_total"] == 100000
    assert report["weighted_success_fraction"] == pytest.approx(1000 / 1001)
    assert report["urls_in_domains_without_successful_sample"] == 1


def test_failed_samples_are_not_mistaken_for_full_corpus_size(settings):
    rows = [{"id": 1, "host": "blocked.example", "status": "robots_denied"}]
    report = capacity_report({"blocked.example": 5000}, rows, settings)
    assert report["metrics_under_current_fetch_policy"]["document_jsonl_bytes"]["expected_total"] == 0
    assert report["metrics_under_current_fetch_policy"]["chunk_count"]["expected_total"] == 0
    assert report["urls_in_domains_without_successful_sample"] == 5000
    assert report["full_coverage_text_only_scenarios"][0]["total_bytes"] > 0


def test_unicode_chunk_spans_cover_text_and_ids_are_local_and_stable():
    d = document()
    chunks = list(chunk_document(d, 120, 20))
    covered = set()
    for row in chunks:
        start, end = row["metadata"]["start_char"], row["metadata"]["end_char"]
        assert row["text"] == d["text"][start:end]
        assert 0 < end - start <= 120
        assert row["chunk_id"].startswith("rag:7:") and "official_chunk_id" not in row
        covered.update(range(start, end))
    assert covered == set(range(len(d["text"])))
    assert chunks == list(chunk_document(d, 120, 20))
    assert chunks[0]["chunk_id"] != next(chunk_document(d, 130, 20))["chunk_id"]
    with pytest.raises(ValueError):
        list(chunk_document(d, 20, 20))


def test_html_extractor_removes_navigation_and_script_and_preserves_language():
    paragraphs = "".join(f"<p>Paragraph {i}. Medical diagnostic research and therapeutic evidence are evaluated using clinical observational studies.</p>" for i in range(12))
    body = (f'<html lang="en"><head><title>Clinical study</title></head><body><nav>MENU_ONLY</nav><script>SCRIPT_ONLY</script><article>{paragraphs}</article></body></html>').encode()
    d = extract_document(body, "text/html", "https://example.org/a", 583)
    assert d["doc_id"] == "583" and d["official_id"] == 583
    assert d["metadata"]["language"] == "en"
    assert "MENU_ONLY" not in d["text"] and "SCRIPT_ONLY" not in d["text"]
    assert "Paragraph 11" in d["text"]


def test_robots_denied_page_not_requested(settings, monkeypatch):
    fetcher = WebFetcher(settings)
    requests = []
    def fake_get(url, cap):
        requests.append(url)
        return {"code": 200, "headers": {}, "body": b"User-agent: *\nDisallow: /\n", "wire_bytes": 30, "too_large": False}
    monkeypatch.setattr(fetcher, "_get", fake_get)
    result = fetcher.fetch({"id": 7, "url": "https://example.org/a", "host": "example.org"})
    assert result["status"] == "robots_denied"
    assert requests == ["https://example.org/robots.txt"]


def test_oversized_response_is_never_indexed(settings, monkeypatch):
    fetcher = WebFetcher(settings)
    def fake_get(url, cap):
        return {"code": 404 if url.endswith("robots.txt") else 200, "headers": {"content-type": "text/html"},
                "body": b"", "wire_bytes": 50, "too_large": not url.endswith("robots.txt")}
    monkeypatch.setattr(fetcher, "_get", fake_get)
    result = fetcher.fetch({"id": 7, "url": "https://example.org/a", "host": "example.org"})
    assert result["status"] == "response_limit" and "document" not in result


def test_resume_preserves_success_and_retries_only_failed(config, tmp_path):
    class FakeFetcher:
        calls = []
        def fetch(self, row):
            self.calls.append(row["id"])
            return successful_result(row["id"], row["host"])
    rows = [{"id": 7, "url": "https://one.example/a", "host": "one.example"}]
    state, identity, fetcher = tmp_path / "state", source_identity(config), FakeFetcher()
    with open_state(state, identity) as db:
        assert acquire(rows, db, config["rag"], state, fetcher=fetcher) == 1
    with open_state(state, identity) as db:
        assert acquire(rows, db, config["rag"], state, fetcher=fetcher, retry_failed=True) == 0
    assert fetcher.calls == [7]
    altered = {**identity, "revision": "different"}
    with pytest.raises(ValueError, match="different source"):
        with open_state(state, altered):
            pass


def test_streaming_export_builds_real_compatible_retrieval_dataset(config, tmp_path):
    state, output = tmp_path / "state", tmp_path / "rag"
    with open_state(state, source_identity(config)) as db:
        # Identical text at two source IDs must retain both identities.
        for id in (7, 900):
            result = successful_result(id)
            db.execute("INSERT INTO pages VALUES (?, ?, ?)", (id, "ok", encoded(result).decode()))
        db.commit()
    manifest = export(config, state, output)
    assert manifest["counts"]["documents"] == 2 and manifest["competition_submission_ready"] is False
    rag_config = load_config(output / "rag_config.yaml")
    chunks, queries, registry = load_dataset(rag_config)
    assert registry["doc_ids"] == ["7", "900"] and queries[0]["id"] == "31"
    assert all(c["chunk_id"].startswith("rag:") for c in chunks)
    assert Path(build_bm25_index(rag_config)).exists()
    run_dir = run_retrieval(rag_config)
    with pytest.raises(ValueError, match="local chunk IDs"):
        make_submission(run_dir)
    with pytest.raises(FileExistsError):
        export(config, state, output)


def test_crawl_requires_explicit_limit_and_export_has_no_network(config, tmp_path):
    with pytest.raises(ValueError, match="max_urls"):
        crawl(config, tmp_path / "state")
    with pytest.raises(ValueError, match="max_urls"):
        crawl(config, tmp_path / "state", max_urls=1, all_urls=True)
    with pytest.raises(FileNotFoundError, match="state not found"):
        export(config, tmp_path / "missing", tmp_path / "out")
