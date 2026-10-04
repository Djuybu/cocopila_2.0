"""Kaggle-hosted coordinator with durable cloud checkpoints and bounded storage.

The master dataset indexes private corpus partitions. Only the active partition is
cached locally, so the full corpus need not fit in a notebook's output disk.
Credentials come from Kaggle Secrets, never from a notebook or dataset file.
"""
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

from src.data.download import sha256_file
from src.data.kaggle_full_runner import FullKaggleRunner, coordinator_lock, publication_folder, save_atomic

LOGGER = logging.getLogger(__name__)


def wait_ready(api, ref, timeout=900):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            if api.dataset_status(ref).lower() == "ready":
                return
        except Exception:
            pass  # New dataset permissions/status can take a few seconds to propagate.
        time.sleep(15)
    raise TimeoutError(f"Dataset did not become ready: {ref}")


def upload_private(api, folder, ref, *, created):
    previous = 0
    if created:
        wait_ready(api, ref)
        previous = int(json.loads(api.dataset_status(ref, format="json(current_version_number)"))["current_version_number"])
        response = api.dataset_create_version(str(folder), "Cloud crawl checkpoint", quiet=True,
                                             convert_to_csv=False, delete_old_versions=False, dir_mode="zip")
    else:
        response = api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="zip")
    if response is None or response.error:
        raise RuntimeError(f"Cloud dataset update failed: {ref}")
    deadline = time.monotonic()+900
    while time.monotonic() < deadline:
        try:
            status = json.loads(api.dataset_status(ref, format="json"))
            if status["status"] == "ready" and int(status["current_version_number"]) > previous:
                return f"{ref}/{int(status['current_version_number'])}"
        except Exception:
            pass
        time.sleep(15)
    raise TimeoutError(f"New dataset version did not become ready: {ref}")


def restore_snapshot(snapshot, root, catalogue_root=None):
    snapshot, root = Path(snapshot), Path(root)
    marker = json.loads((snapshot/"control_snapshot.json").read_text())
    for name, spec in marker["files"].items():
        path = (snapshot/name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or not path.is_file():
            raise ValueError("Invalid control snapshot path")
        if path.stat().st_size != spec["size"] or sha256_file(path) != spec["sha256"]:
            raise ValueError("Control snapshot checksum mismatch")
    shutil.copytree(snapshot/"state", root)
    config = json.loads((snapshot/"config.json").read_text())
    config["data"]["raw_dir"] = str(root/"dataset")
    for name, spec in config["data"]["files"].items():
        destination = root/"dataset"/name
        if destination.is_file():
            continue
        candidates = [p for p in Path(catalogue_root or snapshot.parent).rglob(name)
                      if p.stat().st_size == spec["size"] and sha256_file(p) == spec["sha256"]]
        if not candidates:
            raise ValueError("Pinned source catalogue missing from inputs")
        shutil.copy2(candidates[0], destination)
    return config


class CloudKaggleRunner(FullKaggleRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cloud = self.progress.setdefault("cloud", {"active_partition": 0, "partitions": {}, "generation": 0})
        self.partition = self.root/"active_partition"
        self.partition.mkdir(exist_ok=True)
        self.control_ref = f"{self.owner}/vibiomir-crawl-control"

    def refresh_manifest(self):
        manifest = super().refresh_manifest()
        manifest["format"] = "vibiomir-rag-cloud-partitions-v1"
        manifest["storage"] = "Read each listed file from its dataset_ref; the master contains the catalogue and manifests."
        save_atomic(self.dataset/"dataset_manifest.json", manifest)
        readme = self.dataset/"README.md"
        readme.write_text("# ViBioMIR RAG Corpus\n\n"
            f"State: {manifest['state']}. Processed {manifest['counts'].get('sources', 0):,}/{manifest['population_urls']:,} source URLs; "
            f"retained {manifest['counts'].get('documents', 0):,} documents.\n\n"
            "The master dataset contains the original query/URL catalogue and validated corpus manifests. "
            "Document labels retain the official integer ID; rag:* passage IDs are internal.\n\n"
            "Cloud storage: the corpus is partitioned into private Kaggle datasets. "
            "dataset_manifest.json maps every Parquet filename to its pinned dataset_ref. "
            "Use only those listed files; older partial exports may exist in earlier partitions. "
            "Crawl workers run on Kaggle CPU. Cloud orchestration checkpoints progress and starts subsequent workers; "
            "the continuous controller checks at all hours, without a daily crawl window.\n")
        return manifest

    def load_active_partition(self):
        item = self.cloud["partitions"].get(str(self.cloud["active_partition"]))
        if not item:
            return
        def valid_cache():
            return all((self.partition/name).is_file() and (self.partition/name).stat().st_size == spec["size"]
                       and sha256_file(self.partition/name) == spec["sha256"] for name, spec in item["files"].items())
        if valid_cache():
            return
        # A timed-out download must not turn a partial cache into an accepted one.
        shutil.rmtree(self.partition)
        self.partition.mkdir()
        self.api.dataset_download_files(item["ref"], path=str(self.partition), quiet=True, unzip=True)
        for folder in tuple(self.partition.iterdir()):
            if folder.is_dir():
                for path in folder.iterdir():
                    if path.is_file():
                        shutil.move(str(path), self.partition/path.name)
                folder.rmdir()
        # Validate persisted Parquet contents before allowing an incremental update.
        if not valid_cache():
            raise ValueError("Persisted partition checksum mismatch")

    def persist_partition(self):
        index = str(self.cloud["active_partition"])
        ref = f"{self.owner}/vibiomir-rag-part-{int(index):04d}"
        files = {p.name: {"size":p.stat().st_size, "sha256":sha256_file(p)}
                 for p in self.partition.glob("*.parquet")}
        save_atomic(self.partition/"dataset-metadata.json", {"id":ref,
                    "title":f"ViBioMIR Corpus Partition {int(index):04d}", "licenses":[{"name":"other"}],
                    "description":"Private ViBioMIR RAG corpus partition. Official integer document IDs are retained; source content remains attributable to its authors."})
        save_atomic(self.partition/"dataset_manifest.json", {"files":files})
        with publication_folder(self.partition) as folder:
            pinned = upload_private(self.api, folder, ref, created=index in self.cloud["partitions"])
        self.cloud["partitions"][index] = {"ref":pinned, "files":files}
        return pinned

    def register_shard(self, manifest, output):
        self.load_active_partition()
        limit = self.config.get("cloud_crawl", {}).get("partition_bytes", 1024**3)
        incoming = sum(spec["size"] for spec in manifest["files"].values())
        retained = sum(p.stat().st_size for p in self.partition.glob("*.parquet") if p.name not in manifest["files"])
        shard_limit = self.config.get("cloud_crawl", {}).get("partition_shards", 4)
        old_ids = {p.name.split("-")[-1].split(".")[0] for p in self.partition.glob("documents-*.parquet")}
        new_id = f"{manifest['shard']['id']:05d}"
        if retained and (retained+incoming > limit or (new_id not in old_ids and len(old_ids) >= shard_limit)):
            # Previously saved partitions stay in Kaggle; only discard this owned cache.
            shutil.rmtree(self.partition)
            self.partition.mkdir()
            self.cloud["active_partition"] += 1
        for name in manifest["files"]:
            shutil.copy2(Path(output)/name, self.partition/name)
        pinned = self.persist_partition()
        manifest = json.loads(json.dumps(manifest))
        for name, spec in manifest["files"].items():
            spec["dataset_ref"] = pinned
            spec["path"] = name.split("-")[0]+"/"+name
        save_atomic(self.dataset/f"shard-{manifest['shard']['id']:05d}.json", manifest)
        self.refresh_manifest()
        self.save()
        return manifest

    def collect(self, shard, job):
        # Reuse all validation in the local coordinator, but do not accumulate data.
        original = self.dataset
        with tempfile.TemporaryDirectory(prefix="collect-", dir=self.root) as temp:
            staging = Path(temp)
            self.dataset = staging
            refresh = self.refresh_manifest
            self.refresh_manifest = lambda: None
            try:
                # Pin the worker output version even if its slug later executes another shard.
                pinned_job = {**job, "ref": f"{job['ref']}/{job['version']}"}
                manifest = super().collect(shard, pinned_job)
            finally:
                self.dataset = original
                self.refresh_manifest = refresh
            result = self.register_shard(manifest, staging)
        download = self.root/"downloads"/f"{shard['id']:05d}-{job['runs']:02d}"
        if download.exists():
            shutil.rmtree(download)
        return result

    def checkpoint(self):
        self.save()
        with tempfile.TemporaryDirectory(prefix="control-", dir=self.root) as temp:
            folder = Path(temp)
            (folder/"state").mkdir()
            for name in ("progress.json", "plan.json"):
                shutil.copy2(self.root/name, folder/"state"/name)
            # The catalogue is mounted separately from its immutable dataset version.
            # Do not duplicate its 35 MB in every controller checkpoint.
            shutil.copytree(self.dataset, folder/"state"/"dataset",
                            ignore=shutil.ignore_patterns("query.parquet", "links_corpus.parquet"))
            save_atomic(folder/"config.json", self.config)
            files = {str(p.relative_to(folder)): {"size":p.stat().st_size, "sha256":sha256_file(p)}
                     for p in folder.rglob("*") if p.is_file()}
            save_atomic(folder/"control_snapshot.json", {"files":files, "generation":self.cloud["generation"]})
            save_atomic(folder/"dataset-metadata.json", {"id":self.control_ref, "title":"ViBioMIR Crawl Control",
                        "licenses":[{"name":"other"}], "description":"Private coordinator checkpoints. Contains crawl state and source catalogue only; no credentials."})
            pinned = upload_private(self.api, folder, self.control_ref,
                                    created=self.cloud.get("control_created", False))
        self.cloud.update(control_created=True, control_ref=pinned, last_checkpoint=time.time())
        self.save()
        return pinned

    def submit(self, *args, **kwargs):
        # Checkpoint completed work and then the newly assigned worker. Never hand
        # off a controller in the middle of a submission.
        self.checkpoint()
        super().submit(*args, **kwargs)
        self.checkpoint()

    def handoff(self):
        from src.data.kaggle_cloud_notebook import build_cloud_notebook
        self.cloud["generation"] += 1
        pinned = self.checkpoint()
        nb, meta = build_cloud_notebook(self.project, owner=self.owner, control_dataset=pinned)
        folder = self.root/"next_controller"
        folder.mkdir(exist_ok=True)
        save_atomic(folder/meta["code_file"], nb)
        save_atomic(folder/"kernel-metadata.json", meta)
        response = self.api.kernels_push(str(folder), timeout=43200)
        if response is None or response.error or response.invalid_dataset_sources:
            raise RuntimeError("Failed to schedule the next Kaggle coordinator session")
        LOGGER.info("Cloud handoff accepted: %s version %s, checkpoint %s", meta["id"], response.version_number, pinned)
        return response.version_number

    def run(self, poll_seconds=30, session_seconds=None):
        settings = self.config.get("cloud_crawl", {})
        initial = settings.get("first_handoff_seconds", settings.get("session_seconds", 32400))
        mode = settings.get("handoff_mode", "scheduled")
        normal = settings.get("session_seconds", 32400)
        if mode == "scheduled" and not self.cloud.get("native_schedule_verified"):
            normal = settings.get("native_first_run_seconds", 60)
        budget = session_seconds or (initial if mode == "api" and self.cloud["generation"] == 0 else normal)
        deadline = time.monotonic()+budget
        with coordinator_lock(self.root):
            self.progress.update(execution_platform="Kaggle", state="building")
            self.cloud["control_created"] = True  # Booting from an existing control dataset.
            self.checkpoint()
            LOGGER.info("Kaggle coordinator ready: generation=%s, budget=%ss", self.cloud["generation"], budget)
            while True:
                try:
                    if time.monotonic() >= deadline:
                        wait_ready(self.api, self.dataset_ref)
                        self.publish()
                        wait_ready(self.api, self.dataset_ref)
                        if mode == "api":
                            self.handoff()
                        else:
                            self.cloud["generation"] += 1
                            self.cloud["native_schedule_verified"] = True
                            self.progress["state"] = "waiting_for_schedule"
                            self.checkpoint()
                            LOGGER.info("Cloud checkpoint saved; next native scheduled run will continue acquisition")
                        return
                    if self.cycle():
                        wait_ready(self.api, self.dataset_ref)
                        self.checkpoint()
                        LOGGER.info("Full catalogue processing ended: %s", self.progress["state"])
                        return
                    if time.time()-self.cloud.get("last_checkpoint", 0) >= 600:
                        self.checkpoint()
                except Exception as error:
                    LOGGER.error("Cloud cycle failed: %s; retrying", type(error).__name__)
                    self.progress["last_error"] = type(error).__name__
                    self.save()
                time.sleep(poll_seconds)


def migrate_local(runner, local_dataset):
    """Import collected local shards without repeating acquisition."""
    for marker in sorted(Path(local_dataset).glob("shard-*.json")):
        runner.register_shard(json.loads(marker.read_text()), local_dataset)
    runner.publish()
    wait_ready(runner.api, runner.dataset_ref)
    return runner.checkpoint()


def iter_cloud_chunks(manifest_path, api, cache_dir, batch_size=1000):
    """Read exact declared Parquet files, keeping only one pinned partition cached."""
    import pyarrow.parquet as pq
    manifest = json.loads(Path(manifest_path).read_text())
    groups = {}
    for name, spec in manifest["files"].items():
        if Path(name).name.startswith("chunks-"):
            groups.setdefault(spec["dataset_ref"], []).append((name, spec))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ref, files in groups.items():
        with tempfile.TemporaryDirectory(prefix="rag-partition-", dir=cache_dir) as temp:
            root = Path(temp)
            api.dataset_download_files(ref, path=str(root), quiet=True, unzip=True)
            for name, spec in files:
                relative = spec.get("path", name if "/" in name else "chunks/"+name)
                path = (root/relative).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Invalid corpus file path")
                if path.stat().st_size != spec["size"] or sha256_file(path) != spec["sha256"]:
                    raise ValueError("Corpus Parquet checksum mismatch")
                for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                    yield batch
