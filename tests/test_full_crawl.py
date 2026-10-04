import ast
import hashlib
import json
from pathlib import Path
import time

import pytest
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("protego")

from src.data.full_crawl import collection_manifest, document_rankings, plan_full_crawl, range_rows, run_shard
from src.data.kaggle_crawl import build_full_notebook
from src.data.kaggle_full_runner import FullKaggleRunner, choose_shard, publication_folder
from src.data.web_corpus import WebFetcher, extract_document


@pytest.fixture
def config(tmp_path):
    raw = tmp_path/"raw"
    raw.mkdir()
    pq.write_table(pa.table({"id": [7, 900, 12000], "url": ["http://one.example/a", "https://two.example/b", "https://one.example/c"]}), raw/"links_corpus.parquet")
    pq.write_table(pa.table({"id": [31], "query": ["Y khoa"]}), raw/"query.parquet")
    return {"data": {"repo": "official/test", "revision": "pin", "raw_dir": str(raw),
                     "files": {p.name: {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in raw.glob("*.parquet")}},
            "rag": {"seed": 42, "sample_per_host": 3, "workers": 1, "timeout_seconds": 1,
                    "max_response_bytes": 4096, "min_text_chars": 20, "delay_per_host_seconds": 0,
                    "prefer_https": False, "user_agent": "TestRAG/1", "chunk_chars": 120, "chunk_overlap": 20,
                    "keep_raw": False, "bootstrap_replicates": 10, "embedding_dimension": 1024,
                    "retry_attempts": 3, "retry_backoff_seconds": 0},
            "full_crawl": {"shard_size": 2, "budget_seconds": 10}}


class FakeFetcher:
    def __init__(self, failed_ids=(), pause=0):
        self.calls, self.failed_ids, self.pause = [], set(failed_ids), pause

    def fetch(self, row):
        self.calls.append(row["id"])
        time.sleep(self.pause)
        result = {**row, "status": "http_error" if row["id"] in self.failed_ids else "ok",
                  "http_status": 404 if row["id"] in self.failed_ids else 200, "retryable": False}
        if result["status"] == "ok":
            text = "Nội dung y khoa. 中文医学。 "*20
            result["document"] = {"doc_id": str(row["id"]), "official_id": row["id"], "url": row["url"], "title": "Medical",
                                  "text": text, "metadata": {"text_sha256": hashlib.sha256(text.encode()).hexdigest()}}
        return result


def test_plan_partitions_every_row_without_assuming_contiguous_ids(config, tmp_path):
    plan = plan_full_crawl(config, tmp_path/"plan.json", 2)
    assert [(s["start"],s["stop"]) for s in plan["shards"]] == [(0,2),(2,3)]
    assert plan["population_urls"] == 3 and plan["hosts"] == {"one.example": 2,"two.example": 1}
    assert [r["id"] for r in range_rows(Path(config["data"]["raw_dir"])/"links_corpus.parquet", 1, 3)] == [900,12000]
    assert choose_shard(plan["shards"], {"0": {"state": "running"}}) is None
    assert choose_shard(plan["shards"], {"0": {"state": "done"}})["id"] == 1


def test_full_export_covers_failed_urls_without_claiming_complete_content(config, tmp_path):
    plan = plan_full_crawl(config, tmp_path/"plan.json", 3)
    manifest = run_shard(config, plan["shards"][0], tmp_path/"state", tmp_path/"out", fetcher=FakeFetcher([900]))
    assert manifest["acquisition_complete"] and not manifest["content_complete"]
    assert manifest["missing_content"] == 1
    docs = pq.read_table(tmp_path/"out/documents-00000.parquet").to_pylist()
    assert [r["official_id"] for r in docs] == [7,12000]
    chunks = pq.read_table(tmp_path/"out/chunks-00000.parquet").to_pylist()
    assert all(isinstance(r["official_doc_id"], int) and r["chunk_id"].startswith("rag:") for r in chunks)
    assert pq.read_table(tmp_path/"out/sources-00000.parquet")["id"].to_pylist() == [7,900,12000]
    total = collection_manifest(plan, [manifest])
    assert total["state"] == "needs_source_recovery" and total["pending_urls"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        collection_manifest(plan, [manifest,manifest])


def test_zero_successes_still_export_failure_registry(config, tmp_path):
    plan = plan_full_crawl(config, tmp_path/"plan.json", 3)
    manifest = run_shard(config, plan["shards"][0], tmp_path/"state", tmp_path/"out", fetcher=FakeFetcher([7,900,12000]))
    assert manifest["counts"] == {"documents": 0,"chunks": 0,"sources": 3}
    assert pq.ParquetFile(tmp_path/"out/documents-00000.parquet").metadata.num_rows == 0


def test_time_budget_checkpoint_resumes_without_fetching_successes_again(config, tmp_path):
    plan = plan_full_crawl(config, tmp_path/"plan.json", 3)
    shard = plan["shards"][0]
    first = FakeFetcher(pause=0.02)
    m1 = run_shard(config, shard, tmp_path/"state", tmp_path/"out1", budget_seconds=0.005, fetcher=first)
    assert m1["pending_urls"] > 0 and not m1["acquisition_complete"]
    second = FakeFetcher()
    m2 = run_shard(config, shard, tmp_path/"state", tmp_path/"out2", budget_seconds=10, fetcher=second)
    assert m2["content_complete"] and len(first.calls)+len(second.calls) == 3
    assert not set(first.calls).intersection(second.calls)


def test_integer_document_ranking_collapses_passages_by_best_score():
    rows = [{"doc_id": "7", "score": 3}, {"doc_id": "7", "score": 10}, {"official_doc_id": 900, "score": 8}]
    assert document_rankings(rows, [7,900]) == [{"document_id": 7,"score": 10.0},{"document_id": 900,"score": 8.0}]
    with pytest.raises(ValueError, match="official document"):
        document_rankings([{"doc_id": "rag:7:local", "score": 2}], [7])


def test_full_notebook_is_not_a_sample_and_pins_resume_version(config):
    project = Path(__file__).resolve().parents[1]
    nb, meta = build_full_notebook(project, owner="testowner", slug="full-test", shard={"id": 0,"start": 0,"stop": 3},
                                  config=config, resume_kernel="testowner/previous-run/1", retry_transient_only=True)
    assert meta["is_private"] and meta["enable_internet"] and meta["kernel_sources"] == ["testowner/previous-run/1"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    tree = ast.parse(nb["cells"][3]["source"])
    call = next(n.value for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id == "CONFIG")
    assert json.loads(ast.literal_eval(call.args[0]))["full_crawl"]["retry_transient_only"] is True
    assert "run_shard" in nb["cells"][4]["source"] and "estimate(" not in nb["cells"][4]["source"]


def test_http_original_and_transient_page_retry(config, monkeypatch):
    fetcher = WebFetcher(config["rag"])
    calls = []
    def get(url, cap):
        calls.append(url)
        code = 404 if url.endswith("robots.txt") else 503 if len(calls) == 2 else 200
        return {"code": code,"headers": {"content-type":"text/plain"},"body": b"Clinical evidence and therapeutic recommendations."*10,
                "wire_bytes": 100,"too_large": False}
    monkeypatch.setattr(fetcher,"_get",get)
    result = fetcher.fetch({"id":7,"url":"http://one.example/a","host":"one.example"})
    assert result["status"] == "ok" and result["request_attempts"] == 3
    assert calls == ["http://one.example/robots.txt","http://one.example/a","http://one.example/a"]


def test_robots_access_error_is_distinct_from_a_disallow_rule(config, monkeypatch):
    fetcher = WebFetcher(config["rag"])
    monkeypatch.setattr(fetcher,"_get",lambda url, cap: {"code":403,"headers":{},"body":b"Denied","wire_bytes":6,"too_large":False})
    result = fetcher.fetch({"id":7,"url":"https://one.example/a","host":"one.example"})
    assert result["status"] == "robots_http_denied" and result["robots"]["http_status"] == 403
    assert result["request_attempts"] == 1 and not result["retryable"]


def test_robots_temporary_error_cache_expires(config, monkeypatch):
    from src.data import web_corpus
    settings = {**config["rag"],"retry_attempts":1,"robots_failure_ttl_seconds":1}
    fetcher = WebFetcher(settings)
    now = [100]
    monkeypatch.setattr(web_corpus.time,"monotonic",lambda:now[0])
    calls = []
    def get(url, cap):
        calls.append(url)
        code = 503 if len(calls)==1 else 404 if url.endswith("robots.txt") else 200
        return {"code":code,"headers":{"content-type":"text/plain"},"body":b"Clinical evidence and therapeutic recommendations."*10,"wire_bytes":10,"too_large":False}
    monkeypatch.setattr(fetcher,"_get",get)
    row = {"id":7,"url":"https://one.example/a","host":"one.example"}
    assert fetcher.fetch(row)["status"] == "robots_unavailable"
    now[0] += 2
    assert fetcher.fetch(row)["status"] == "ok" and len(calls)==3


def test_malformed_html_and_explicit_article_fallback_preserve_unicode():
    text = "Nội dung y học tiếng Việt. 中文医疗信息。 "*30
    body = f'<html lang="vi"></html><head><meta charset="utf-8"><title>Nghiên cứu</title></head><body><nav>MENU_ONLY</nav><div itemprop="articleBody">{text}</div></body></html>'.encode()
    d = extract_document(body,"text/html","https://example.org/a",7)
    assert "Nội dung" in d["text"] and "中文" in d["text"] and "MENU_ONLY" not in d["text"]


def test_publication_groups_many_shards_below_kaggle_top_level_limit(tmp_path):
    dataset = tmp_path/"dataset"
    dataset.mkdir()
    files = {}
    for i in range(60):
        name = f"documents-{i:05d}.parquet"
        (dataset/name).write_bytes(b"example")
        files[name] = {"size":7,"sha256":hashlib.sha256(b"example").hexdigest()}
    (dataset/"dataset_manifest.json").write_text(json.dumps({"files":files}))
    with publication_folder(dataset) as upload:
        assert len(list(upload.iterdir())) == 2
        manifest = json.loads((upload/"dataset_manifest.json").read_text())
        assert len(manifest["files"]) == 60
        assert all((upload/name).read_bytes() == b"example" for name in manifest["files"])
        assert len(list((upload/"documents").iterdir())) == 60
    assert len(list(dataset.glob("documents-*.parquet"))) == 60


def test_coordinator_survives_dataset_status_propagation_error(config, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.data import kaggle_full_runner
    class API:
        calls = 0
        def dataset_create_new(self, folder, **kwargs):
            assert kwargs["public"] is False and kwargs["dir_mode"] == "zip"
            return SimpleNamespace(error="")
        def dataset_status(self, ref):
            self.calls += 1
            if self.calls == 1:
                raise OSError("Transient dataset activation failure")
            return "ready"
    api = API()
    runner = FullKaggleRunner(Path(__file__).resolve().parents[1], config, tmp_path/"run", "testowner", api=api)
    monkeypatch.setattr(runner,"cycle",lambda:True)
    monkeypatch.setattr(kaggle_full_runner.time,"sleep",lambda value:None)
    runner.run()
    assert api.calls == 3 and runner.progress["dataset_created"]
    assert "Transient" in runner.progress["last_error"]


def test_recovered_content_is_republished_when_source_count_is_unchanged(config, tmp_path, monkeypatch):
    from src.data.download import sha256_file
    from types import SimpleNamespace
    config["full_crawl"].update(slots=0, publish_interval_seconds=0)
    class API:
        versions = 0
        def dataset_status(self, ref): return "ready"
        def dataset_create_version(self, folder, notes, **kwargs):
            self.versions += 1
            return SimpleNamespace(error="")
    api = API()
    runner = FullKaggleRunner(Path(__file__).resolve().parents[1], config, tmp_path/"run", "testowner", api=api)
    runner.progress.update(dataset_created=True, published_sources=3, published_fingerprint="previous_manifest",
                           publication_layout="directories-v1")
    runner.progress["coverage"]["counts"] = {"sources":3,"documents":3}
    monkeypatch.setattr(runner,"refresh_manifest",lambda:None)
    runner.cycle()
    assert api.versions == 1
    assert runner.progress["published_fingerprint"] == sha256_file(runner.dataset/"dataset_manifest.json")


def test_local_coordinator_cannot_restart_after_cloud_transfer(config,tmp_path):
    runner = FullKaggleRunner(Path(__file__).resolve().parents[1],config,tmp_path/"run","testowner",api=object())
    runner.progress["coordinator_backend"]="kaggle_native_schedule"
    with pytest.raises(RuntimeError,match="ownership has moved"):
        runner.run()


def test_http_decode_retains_encoded_wire_size(config, monkeypatch):
    import gzip,io
    from urllib3.response import HTTPResponse
    payload = b"Medical clinical evidence. "*100
    packed = gzip.compress(payload)
    class Response:
        status_code = 200
        headers = {"Content-Type":"text/plain","Content-Encoding":"gzip"}
        raw = HTTPResponse(body=io.BytesIO(packed), headers=headers, preload_content=False)
        def __enter__(self): return self
        def __exit__(self,*args): self.raw.close()
    class Session:
        def get(self,*args,**kwargs): return Response()
    fetcher = WebFetcher(config["rag"])
    monkeypatch.setattr(fetcher,"_session",lambda:Session())
    result = fetcher._get("https://example.org/a",4096)
    assert result["body"] == payload and result["wire_bytes"] == len(packed)
    assert not result["too_large"]
