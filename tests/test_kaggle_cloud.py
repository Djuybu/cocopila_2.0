import ast
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.data.full_crawl import plan_full_crawl, run_shard
from src.data.kaggle_cloud_notebook import build_cloud_notebook
from src.data.kaggle_cloud_runner import CloudKaggleRunner, restore_snapshot, upload_private, iter_cloud_chunks
from src.data.kaggle_full_runner import save_atomic


@pytest.fixture
def cloud_config(tmp_path):
    raw = tmp_path/"raw"
    raw.mkdir()
    pq.write_table(pa.table({"id":[7,900],"url":["http://one.example/a","https://two.example/b"]}),raw/"links_corpus.parquet")
    pq.write_table(pa.table({"id":[31],"query":["Y khoa"]}),raw/"query.parquet")
    return {"data":{"repo":"official/test","revision":"pin","raw_dir":str(raw),
            "files":{p.name:{"size":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in raw.glob("*.parquet")}},
            "rag":{"seed":42,"sample_per_host":1,"workers":1,"timeout_seconds":1,"max_response_bytes":4096,
                   "min_text_chars":20,"delay_per_host_seconds":0,"prefer_https":False,"user_agent":"Test/1",
                   "chunk_chars":120,"chunk_overlap":20,"keep_raw":False,"bootstrap_replicates":10,"embedding_dimension":1024},
            "full_crawl":{"shard_size":1,"budget_seconds":10,"slots":0,"publish_interval_seconds":0,"max_resume_runs":8},
            "cloud_crawl":{"partition_bytes":1}}


class API:
    def __init__(self): self.uploads,self.versions={},{}
    def dataset_status(self,ref,format=None):
        return json.dumps({"status":"ready", "current_version_number":self.versions.get(ref,1)}) if format else "ready"
    def dataset_create_new(self,folder,**kwargs):
        assert kwargs["public"] is False
        return self.dataset_create_version(folder,"",**kwargs)
    def dataset_create_version(self,folder,notes,**kwargs):
        folder=Path(folder)
        ref=json.loads((folder/"dataset-metadata.json").read_text())["id"]
        self.versions[ref]=self.versions.get(ref,0)+1
        self.uploads[ref]={str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob("*") if p.is_file()}
        return SimpleNamespace(error="")
    def dataset_download_files(self,ref,path,**kwargs):
        ref="/".join(ref.split("/")[:2])
        for name,content in self.uploads[ref].items():
            target=Path(path)/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)


class Fetcher:
    def fetch(self,row):
        text="Nội dung y học. 中文医学。 "*30
        return {**row,"status":"ok","http_status":200,"retryable":False,
                "document":{"doc_id":str(row["id"]),"official_id":row["id"],"url":row["url"],"title":"Medical",
                            "text":text,"metadata":{"text_sha256":hashlib.sha256(text.encode()).hexdigest()}}}


def runner(config,tmp_path):
    return CloudKaggleRunner(Path(__file__).resolve().parents[1],config,tmp_path/"runner","testowner",api=API())


def test_cloud_notebook_uses_secret_and_embeds_every_dependency():
    import base64,io,zipfile
    nb,meta=build_cloud_notebook(Path(__file__).resolve().parents[1],owner="testowner",control_dataset="testowner/crawl-control/1")
    assert meta["dataset_sources"]==["testowner/crawl-control/1", "testowner/vibiomir-rag-corpus/1"] and meta["is_private"]
    for cell in nb["cells"]:
        if cell["cell_type"]=="code": ast.parse(cell["source"])
    assert 'get_secret("KAGGLE_API_TOKEN")' in nb["cells"][3]["source"]
    assert "authentication_required" in nb["cells"][3]["source"]
    tree=ast.parse(nb["cells"][2]["source"])
    value=next(n.value.args[0] for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=="payload")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(value)))) as archive:
        assert "src/data/kaggle_cloud_notebook.py" in archive.namelist()
        assert "configs/data/vibiomir_rag.yaml" in archive.namelist()
        assert not any("credential" in n or "kaggle.json" in n for n in archive.namelist())


def test_cloud_partitions_roll_over_and_manifest_points_to_pinned_payload(cloud_config,tmp_path):
    r=runner(cloud_config,tmp_path)
    manifests=[]
    for shard in r.plan["shards"]:
        out=tmp_path/f"out-{shard['id']}"
        manifest=run_shard(cloud_config,shard,tmp_path/f"state-{shard['id']}",out,fetcher=Fetcher())
        manifests.append(r.register_shard(manifest,out))
    assert r.cloud["active_partition"]==1 and len(r.cloud["partitions"])==2
    assert not list(r.dataset.glob("*.parquet"))==[]  # Official catalogue only.
    assert not list(r.dataset.glob("documents-*.parquet"))
    master=json.loads((r.dataset/"dataset_manifest.json").read_text())
    assert master["counts"]["documents"]==2 and master["content_complete"]
    assert master["format"]=="vibiomir-rag-cloud-partitions-v1"
    assert {s["dataset_ref"] for s in master["files"].values()}=={
        "testowner/vibiomir-rag-part-0000/1","testowner/vibiomir-rag-part-0001/1"}
    assert {p.name for p in r.partition.glob("documents-*.parquet")}=={"documents-00001.parquet"}


def test_checkpoint_round_trip_excludes_payload_and_credentials(cloud_config,tmp_path):
    r=runner(cloud_config,tmp_path)
    (r.partition/"not-in-control.parquet").write_bytes(b"payload")
    ref=r.checkpoint()
    assert ref=="testowner/vibiomir-crawl-control/1"
    files=r.api.uploads[r.control_ref]
    assert not any("not-in-control" in name for name in files)
    snapshot=tmp_path/"input"
    for name,body in files.items():
        path=snapshot/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(body)
    original=tmp_path/"original-catalogue"
    shutil.copytree(Path(cloud_config["data"]["raw_dir"]),original)
    config=restore_snapshot(snapshot,tmp_path/"restored")
    assert config["data"]["raw_dir"]==str(tmp_path/"restored/dataset")
    assert json.loads((tmp_path/"restored/progress.json").read_text())["cloud"]["generation"]==0
    (snapshot/"state/progress.json").write_text("corrupted")
    with pytest.raises(ValueError,match="checksum"):
        restore_snapshot(snapshot,tmp_path/"bad-restore")


def test_existing_partition_download_is_verified_and_reused(cloud_config,tmp_path):
    r=runner(cloud_config,tmp_path)
    shard=r.plan["shards"][0]
    out=tmp_path/"out"
    manifest=run_shard(cloud_config,shard,tmp_path/"state",out,fetcher=Fetcher())
    r.register_shard(manifest,out)
    shutil.rmtree(r.partition)
    r.partition.mkdir()
    r.load_active_partition()
    assert (r.partition/"documents-00000.parquet").exists()
    (r.partition/"documents-00000.parquet").unlink()
    r.load_active_partition()
    assert (r.partition/"documents-00000.parquet").exists()
    shutil.rmtree(r.partition)
    r.partition.mkdir()
    r.api.uploads["testowner/vibiomir-rag-part-0000"]["documents/documents-00000.parquet"]=b"bad"
    with pytest.raises(ValueError,match="checksum"):
        r.load_active_partition()


def test_cloud_handoff_pins_cloud_checkpoint_before_submitting_successor(cloud_config,tmp_path):
    r=runner(cloud_config,tmp_path)
    def push(folder,timeout):
        meta=json.loads((Path(folder)/"kernel-metadata.json").read_text())
        assert r.control_ref in r.api.uploads
        assert meta["dataset_sources"][0]==r.cloud["control_ref"]
        assert timeout==43200 and meta["id"]=="testowner/vibiomir-cloud-coordinator"
        return SimpleNamespace(error="",invalid_dataset_sources=[],version_number=2)
    r.api.kernels_push=push
    assert r.handoff()==2 and r.cloud["generation"]==1


def test_cloud_expiry_hands_off_without_needing_another_local_loop(cloud_config,tmp_path,monkeypatch):
    from src.data import kaggle_cloud_runner
    r=runner(cloud_config,tmp_path)
    r.config["cloud_crawl"]["handoff_mode"]="api"
    events=[]
    r.checkpoint=lambda:events.append("checkpoint")
    r.publish=lambda:events.append("publish")
    r.handoff=lambda:events.append("handoff")
    times=iter([100,102])
    monkeypatch.setattr(kaggle_cloud_runner.time,"monotonic",lambda:next(times))
    monkeypatch.setattr(kaggle_cloud_runner,"wait_ready",lambda *a:None)
    r.run(session_seconds=1)
    assert events==["checkpoint","publish","handoff"]


def test_native_schedule_saves_state_without_pushing_an_unbound_secret_session(cloud_config,tmp_path,monkeypatch):
    from src.data import kaggle_cloud_runner
    r=runner(cloud_config,tmp_path)
    events=[]
    r.checkpoint=lambda:events.append("checkpoint")
    r.publish=lambda:events.append("publish")
    r.handoff=lambda:pytest.fail("Native scheduled runs must not push a new controller version")
    times=iter([100,102])
    monkeypatch.setattr(kaggle_cloud_runner.time,"monotonic",lambda:next(times))
    monkeypatch.setattr(kaggle_cloud_runner,"wait_ready",lambda *a:None)
    r.run(session_seconds=1)
    assert events==["checkpoint","publish","checkpoint"]
    assert r.progress["state"]=="waiting_for_schedule" and r.cloud["generation"]==1


def test_dataset_ready_does_not_pin_the_previous_version(tmp_path,monkeypatch):
    from src.data import kaggle_cloud_runner
    api=API()
    ref="testowner/part"
    api.versions[ref]=1
    status=api.dataset_status
    replies=[]
    def delayed(ref,format=None):
        if format=="json" and not replies:
            replies.append("stale-ready")
            return json.dumps({"status":"ready","current_version_number":1})
        return status(ref,format)
    api.dataset_status=delayed
    delays=[]
    monkeypatch.setattr(kaggle_cloud_runner.time,"sleep",lambda seconds:delays.append(seconds))
    save_atomic(tmp_path/"dataset-metadata.json",{"id":ref})
    assert upload_private(api,tmp_path,ref,created=True)==ref+"/2"
    assert delays==[15]


def test_streaming_reader_uses_only_declared_chunks_and_checks_integrity(cloud_config,tmp_path):
    r=runner(cloud_config,tmp_path)
    shard=r.plan["shards"][0]
    out=tmp_path/"out"
    manifest=run_shard(cloud_config,shard,tmp_path/"state",out,fetcher=Fetcher())
    r.register_shard(manifest,out)
    path=r.dataset/"dataset_manifest.json"
    rows=[row for batch in iter_cloud_chunks(path,r.api,tmp_path/"cache",batch_size=2) for row in batch.to_pylist()]
    assert len(rows)==manifest["counts"]["chunks"]
    assert all(row["official_doc_id"]==7 for row in rows)
    assert not list((tmp_path/"cache").iterdir())
    r.api.uploads["testowner/vibiomir-rag-part-0000"]["chunks/chunks-00000.parquet"]=b"corrupt"
    with pytest.raises(ValueError,match="checksum"):
        list(iter_cloud_chunks(path,r.api,tmp_path/"cache"))
