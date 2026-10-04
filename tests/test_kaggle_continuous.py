import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data import kaggle_continuous as module
from src.data.kaggle_continuous import ContinuousKaggleRunner, PARK_MARKER


def minimal_runner(tmp_path):
    r = object.__new__(ContinuousKaggleRunner)
    r.root = tmp_path
    r.progress = {"jobs":{}, "coverage":{"counts":{"sources":0}}, "state":"building"}
    r.cloud = {"handoff_mode":"continuous"}
    r.config = {"full_crawl":{"slots":2}}
    r.plan = {"population_urls":2, "shards":[
        {"id":0, "hosts":{"one.example":1}}, {"id":1, "hosts":{"two.example":1}}]}
    r.save = lambda: None
    r.api = object()
    return r


def test_tick_does_not_touch_live_native_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "version_status", lambda *a: "KernelWorkerStatus.RUNNING")
    monkeypatch.setattr(module, "park_native_coordinator", lambda *a: pytest.fail("Must let native controller finish"))
    result = module.continuous_tick(tmp_path, "owner", tmp_path/"new", object(), workflow_url="cloud")
    assert result["state"] == "waiting_for_legacy_controller"
    assert not (tmp_path/"new").exists()


def test_parked_notebook_is_cpu_only_and_cannot_mutate_the_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "kernel_details", lambda *a: (SimpleNamespace(), "old code"))
    captured = []
    def push(folder, timeout):
        folder = Path(folder)
        meta = json.loads((folder/"kernel-metadata.json").read_text())
        nb = json.loads((folder/meta["code_file"]).read_text())
        captured.append(nb)
        assert meta["enable_gpu"] is False and meta["enable_tpu"] is False
        assert not meta["enable_internet"] and not meta["dataset_sources"]
        assert PARK_MARKER in nb["cells"][0]["source"]
        assert "kaggle_secrets" not in nb["cells"][0]["source"]
        return SimpleNamespace(error="", version_number=11)
    api = SimpleNamespace(kernels_push=push, kernels_status=lambda ref: SimpleNamespace(status="COMPLETE"))
    assert module.park_native_coordinator(api, "owner/controller", tmp_path, "cloud")
    assert len(captured) == 1


def test_parking_does_not_cancel_a_newer_active_native_run(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "kernel_details", lambda *a: (SimpleNamespace(), "old code"))
    api = SimpleNamespace(kernels_status=lambda ref: SimpleNamespace(status="RUNNING"))
    assert not module.park_native_coordinator(api, "owner/controller", tmp_path, "cloud")


def test_accepted_push_is_recovered_after_an_interrupted_actions_job(tmp_path, monkeypatch):
    r = minimal_runner(tmp_path)
    r.progress["jobs"]["0"] = {"state":"submitting", "ref":"owner/worker", "submission_key":"unique", "submitted_at":0}
    monkeypatch.setattr(module, "kernel_details", lambda *a: (SimpleNamespace(current_version_number=7), "VIBIOMIR_SUBMISSION_KEY: unique"))
    r.recover_submissions()
    assert r.progress["jobs"]["0"]["state"] == "running"
    assert r.progress["jobs"]["0"]["version"] == 7


def test_expired_unaccepted_submission_can_be_retried(tmp_path, monkeypatch):
    r = minimal_runner(tmp_path)
    r.progress["jobs"]["0"] = {"state":"submitting", "ref":"owner/worker", "submission_key":"unique", "submitted_at":0}
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    monkeypatch.setattr(module, "kernel_details", lambda *a: (SimpleNamespace(), "other submission"))
    r.recover_submissions()
    assert r.progress["jobs"]["0"]["state"] == "failed"
    assert r.progress["jobs"]["0"]["failed_at"] == 1000


def test_submission_intent_is_checkpointed_before_the_push(tmp_path):
    r = minimal_runner(tmp_path)
    (tmp_path/"kernel-metadata.json").write_text(json.dumps({"code_file":"worker.ipynb"}))
    (tmp_path/"worker.ipynb").write_text(json.dumps({"cells":[{"source":"# Worker\n"}]}))
    job = {"state":"submitting"}
    seen = []
    r.checkpoint = lambda: seen.append(dict(job))
    r.before_worker_push({}, job, tmp_path)
    assert seen[0]["submission_key"] == job["submission_key"]
    nb = json.loads((tmp_path/"worker.ipynb").read_text())
    assert job["submission_key"] in nb["cells"][0]["source"]


def test_unchanged_tick_does_not_publish_an_empty_checkpoint(tmp_path):
    r = minimal_runner(tmp_path)
    r.cycle = lambda: False
    r.checkpoint = lambda: pytest.fail("No checkpoint needed for an unchanged heartbeat")
    result = r.tick()
    assert result["state"] == "building" and result["compute"] == "CPU"


def test_failed_tick_persists_an_earlier_successful_worker_submission(tmp_path):
    r = minimal_runner(tmp_path)
    def cycle():
        r.progress["jobs"]["0"] = {"state":"running", "version":2}
        raise OSError("later collection failed")
    r.cycle = cycle
    checkpoints = []
    r.checkpoint = lambda: checkpoints.append(json.loads(json.dumps(r.progress["jobs"])))
    with pytest.raises(OSError): r.tick()
    assert checkpoints == [{"0":{"state":"running", "version":2}}]


def test_retry_waits_for_cooldown_and_respects_hostname_exclusivity(tmp_path, monkeypatch):
    r = minimal_runner(tmp_path)
    r.progress["jobs"] = {"0":{"state":"failed", "failed_at":0, "runs":8},
                          "1":{"state":"running", "slot":1}}
    r.plan["shards"][1]["hosts"] = {"one.example":1}
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    r.submit = lambda *a: pytest.fail("The hostname is busy")
    r.retry_failed()
    r.plan["shards"][1]["hosts"] = {"two.example":1}
    submissions = []
    r.submit = lambda shard, slot: submissions.append((shard["id"], slot))
    r.retry_failed()
    assert submissions == [(0, 0)]
    assert r.progress["jobs"]["0"]["failure_restarts"] == 1
