"""Persistent local coordinator for private Kaggle full-catalogue crawl workers."""
from collections import Counter
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time

from src.data.download import sha256_file
from src.data.full_crawl import collection_manifest, plan_full_crawl
from src.data.kaggle_crawl import build_full_notebook

LOGGER = logging.getLogger(__name__)


def save_atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


@contextmanager
def coordinator_lock(root):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def choose_shard(shards, jobs):
    busy = set()
    for shard in shards:
        if jobs.get(str(shard["id"]), {}).get("state") in {"running", "submitting"}:
            busy.update(shard["hosts"])
    for shard in shards:
        if str(shard["id"]) not in jobs and not busy.intersection(shard["hosts"]):
            return shard
    return None


@contextmanager
def publication_folder(dataset):
    """Group thousands of shards into directories for Kaggle's top-level limit."""
    dataset = Path(dataset)
    with tempfile.TemporaryDirectory(prefix="vibiomir-publish-", dir=dataset.parent) as temp:
        output = Path(temp)
        for path in dataset.iterdir():
            if not path.is_file():
                continue
            category = next((group for group in ("documents", "chunks", "sources", "shard")
                             if path.name.startswith(group+"-")), None)
            if category:
                group = "manifests" if category == "shard" else category
                destination = output/group/path.name
                destination.parent.mkdir(exist_ok=True)
            else:
                destination = output/path.name
            if path.name == "dataset_manifest.json" or category == "shard":
                manifest = json.loads(path.read_text())
                manifest["files"] = {name.split("-")[0]+"/"+name: spec for name, spec in manifest.get("files", {}).items()}
                manifest["file_root"] = "dataset_root"
                save_atomic(destination, manifest)
            else:
                os.link(path, destination)
        yield output


class FullKaggleRunner:
    def __init__(self, project, config, root, owner, *, api=None):
        self.project, self.config, self.root = Path(project), config, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.dataset_ref = f"{owner}/vibiomir-rag-corpus"
        self.dataset = self.root/"dataset"
        self.dataset.mkdir(exist_ok=True)
        path = self.root/"plan.json"
        self.plan = json.loads(path.read_text()) if path.exists() else plan_full_crawl(config, path, config["full_crawl"]["shard_size"])
        from src.data.rag_dataset import source_identity
        if self.plan["source"] != source_identity(config):
            raise ValueError("Existing plan uses different catalogue/fetch settings")
        path = self.root/"progress.json"
        self.progress = json.loads(path.read_text()) if path.exists() else {
            "owner": owner, "dataset": self.dataset_ref, "jobs": {}, "dataset_created": False, "last_publish": 0,
            "state": "building", "population_urls": self.plan["population_urls"]}
        if self.progress["owner"] != owner:
            raise ValueError("Coordinator belongs to another Kaggle account")
        if api is None:
            from kaggle import api
        self.api = api
        raw = Path(config["data"]["raw_dir"])
        for name in ("query.parquet", "links_corpus.parquet"):
            if not (self.dataset/name).exists():
                shutil.copy2(raw/name, self.dataset/name)
        shutil.copy2(self.root/"plan.json", self.dataset/"crawl_plan.json")
        self.refresh_manifest()

    def save(self):
        self.progress["updated_at"] = time.time()
        save_atomic(self.root/"progress.json", self.progress)

    def refresh_manifest(self):
        manifests = [json.loads(p.read_text()) for p in sorted(self.dataset.glob("shard-*.json"))]
        manifest = collection_manifest(self.plan, manifests)
        save_atomic(self.dataset/"dataset_manifest.json", manifest)
        self.progress["coverage"] = {k: v for k, v in manifest.items() if k not in {"source", "files"}}
        description = (f"Full ViBioMIR catalogue crawl. Current state: {manifest['state']}. "
                       f"Processed {manifest['counts'].get('sources', 0):,}/{manifest['population_urls']:,} source URLs; "
                       f"retained {manifest['counts'].get('documents', 0):,} documents. "
                       "Parquet documents/chunks retain official integer document labels; rag:* passage IDs are local. "
                       "sources/sources-*.parquet records failed/missing content explicitly. No relevance labels. "
                       "Source: https://huggingface.co/datasets/AIGuruTinix/ViBioMIR. "
                       "The catalogue states CC BY-NC 4.0; crawled content remains subject to its original authors/source terms. "
                       "A dataset is content-complete only when content_complete is true in dataset_manifest.json.")
        save_atomic(self.dataset/"dataset-metadata.json", {"id": self.dataset_ref, "title": "ViBioMIR RAG Corpus",
                    "licenses": [{"name": "other"}], "subtitle": "Full catalogue acquisition with resumable RAG Parquet shards",
                    "description": description})
        (self.dataset/"README.md").write_text("# ViBioMIR RAG Corpus\n\n"+description+"\n\n"
            "Read documents/documents-*.parquet and chunks/chunks-*.parquet as a PyArrow dataset; metadata_json contains JSON metadata. "
            "Rank passages internally, then group by official_doc_id and emit integer document IDs. "
            "query.parquet and links_corpus.parquet are the pinned official catalogue. "
            "Inspect sources/sources-*.parquet for recovery work. An exported shard may be partial while a worker resumes it.\n")
        self.save()
        return manifest

    def publish(self):
        self.refresh_manifest()
        with publication_folder(self.dataset) as upload:
            if self.progress["dataset_created"]:
                if self.api.dataset_status(self.dataset_ref).lower() != "ready":
                    return False
                response = self.api.dataset_create_version(str(upload), "Full crawl coverage update", quiet=True,
                                                           convert_to_csv=False, delete_old_versions=False, dir_mode="zip")
            else:
                response = self.api.dataset_create_new(str(upload), public=False, quiet=True, convert_to_csv=False, dir_mode="zip")
        if response is None or response.error:
            raise RuntimeError(f"Kaggle dataset publish failed: {response.error if response else 'empty response'}")
        self.progress.update(dataset_created=True, last_publish=time.time(),
                             published_sources=self.progress["coverage"]["counts"].get("sources", 0),
                             published_fingerprint=sha256_file(self.dataset/"dataset_manifest.json"),
                             publication_layout="directories-v1")
        self.save()
        LOGGER.info("Published private dataset %s, coverage=%s", self.dataset_ref, self.progress["coverage"]["counts"])
        return True

    def submit(self, shard, slot, *, previous=None, transient_retry=False):
        old = self.progress["jobs"].get(str(shard["id"]), {})
        runs = old.get("runs", 0)+1
        # Different versioned notebook inputs avoid depending on a running notebook's own output.
        side = "a" if runs % 2 else "b"
        slug = f"vibiomir-rag-full-{slot+1:02d}-{side}"
        worker_config = json.loads(json.dumps(self.config))
        if shard["id"] == 0 and runs == 1:
            worker_config["full_crawl"]["budget_seconds"] = self.config["full_crawl"].get("first_checkpoint_seconds", 60)
        nb, metadata = build_full_notebook(self.project, owner=self.owner, slug=slug, shard=shard,
                                          config=worker_config, resume_kernel=previous,
                                          retry_transient_only=transient_retry,
                                          catalogue_dataset=self.dataset_ref+"/1")
        folder = self.root/"bundles"/f"{shard['id']:05d}-{runs:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        save_atomic(folder/metadata["code_file"], nb)
        save_atomic(folder/"kernel-metadata.json", metadata)
        job = {**old, "slot": slot, "runs": runs, "ref": metadata["id"], "state": "submitting",
               "transient_retry": transient_retry, "submitted_at": time.time()}
        self.progress["jobs"][str(shard["id"])] = job
        self.save()
        self.before_worker_push(shard, job, folder)
        response = self.api.kernels_push(str(folder), timeout=28800)
        if response is None or response.error or response.invalid_kernel_sources or response.invalid_dataset_sources:
            job["state"] = "failed"
            job["error"] = str(response.error if response else "Empty push response")
            self.save()
            raise RuntimeError(f"Worker push failed: {job['error']}")
        job.update(state="running", version=response.version_number)
        self.save()
        LOGGER.info("Started full shard %s (%s URLs): https://www.kaggle.com/code/%s version %s", shard["id"], shard["urls"], job["ref"], job["version"])

    def before_worker_push(self, shard, job, folder):
        """Hook for durable submission intent in ephemeral cloud coordinators."""

    def collect(self, shard, job):
        folder = self.root/"downloads"/f"{shard['id']:05d}-{job['runs']:02d}"
        self.api.kernels_output(job["ref"], str(folder), file_pattern=r"^full_shard/", quiet=True)
        output = folder/"full_shard"
        marker = output/f"shard-{shard['id']:05d}.json"
        manifest = json.loads(marker.read_text())
        if manifest["shard"] != shard:
            raise ValueError("Remote worker exported a different shard")
        source = dict(manifest["source"])
        source.pop("catalogue_range", None)
        if source != self.plan["source"]:
            raise ValueError("Remote worker source/settings mismatch")
        import pyarrow.parquet as pq
        for name, spec in manifest["files"].items():
            if Path(name).name != name or not name.endswith(f"-{shard['id']:05d}.parquet"):
                raise ValueError("Unexpected remote file name")
            path = output/name
            if path.stat().st_size != spec["size"] or sha256_file(path) != spec["sha256"]:
                raise ValueError("Remote shard checksum mismatch")
            category = name.split("-")[0]
            if pq.ParquetFile(path).metadata.num_rows != manifest["counts"][category]:
                raise ValueError("Remote shard row count mismatch")
        for name in manifest["files"]:
            shutil.copy2(output/name, self.dataset/name)
        shutil.copy2(marker, self.dataset/marker.name)
        self.refresh_manifest()
        return manifest

    def cycle(self):
        for shard in self.plan["shards"]:
            job = self.progress["jobs"].get(str(shard["id"]))
            if not job or job["state"] not in {"running", "submitting"}:
                continue
            response = self.api.kernels_status(job["ref"])
            status = str(response.status).upper()
            if "COMPLETE" in status:
                if "version" not in job:
                    job.update(state="failed", error="Push response was lost; inspect worker version before resuming")
                    self.save()
                    continue
                manifest = self.collect(shard, job)
                LOGGER.info("Shard %s export: %s", shard["id"], manifest["counts"])
                needs_retry = manifest["acquisition_complete"] and manifest["retryable_failures"] > 0 and not job.get("transient_retry")
                if not manifest["acquisition_complete"] or needs_retry:
                    if job["runs"] >= self.config["full_crawl"]["max_resume_runs"]:
                        job.update(state="failed", error="Resume run limit reached")
                    else:
                        self.submit(shard, job["slot"], previous=f"{job['ref']}/{job['version']}", transient_retry=needs_retry or job.get("transient_retry", False))
                else:
                    job["state"] = "done"
                self.save()
            elif "ERROR" in status or "FAIL" in status or "CANCEL" in status:
                job.update(state="failed", error=getattr(response, "failure_message", "Worker failed"))
                self.save()
                LOGGER.error("Shard %s failed: %s", shard["id"], job["error"])
        self.before_new_shards()
        jobs = self.progress["jobs"]
        busy_slots = {j["slot"] for j in jobs.values() if j["state"] in {"running", "submitting"}}
        for slot in range(self.config["full_crawl"]["slots"]):
            if slot not in busy_slots:
                shard = choose_shard(self.plan["shards"], jobs)
                if shard:
                    self.submit(shard, slot)
        all_terminal = len(jobs) == len(self.plan["shards"]) and all(j["state"] in {"done", "failed"} for j in jobs.values())
        counts = self.progress["coverage"]["counts"]
        unpublished = (sha256_file(self.dataset/"dataset_manifest.json") != self.progress.get("published_fingerprint")
                       or self.progress.get("publication_layout") != "directories-v1")
        if unpublished and (all_terminal or self.progress.get("publication_layout") != "directories-v1" or
                            not self.progress.get("published_sources") or
                            time.time()-self.progress["last_publish"] >= self.config["full_crawl"]["publish_interval_seconds"]):
            self.publish()
        if all_terminal:
            self.progress["state"] = "needs_attention" if any(j["state"] == "failed" for j in jobs.values()) else self.progress["coverage"]["state"]
        self.save()
        return all_terminal and sha256_file(self.dataset/"dataset_manifest.json") == self.progress.get("published_fingerprint")

    def before_new_shards(self):
        """Hook for recovery work before assigning the next pending shards."""

    def run(self, poll_seconds=30):
        if self.progress.get("coordinator_backend") == "kaggle_native_schedule":
            raise RuntimeError("Crawl ownership has moved to Kaggle native scheduling; use its cloud coordinator notebook")
        with coordinator_lock(self.root):
            self.progress.update(pid=os.getpid(), run_started_at=time.time())
            self.save()
            while True:
                try:
                    if not self.progress["dataset_created"]:
                        self.publish()
                    if self.api.dataset_status(self.dataset_ref).lower() != "ready":
                        LOGGER.info("Waiting for the private catalogue dataset to become ready")
                        time.sleep(poll_seconds)
                        continue
                    if self.cycle():
                        if self.api.dataset_status(self.dataset_ref).lower() != "ready":
                            time.sleep(poll_seconds)
                            continue
                        LOGGER.info("Full catalogue acquisition finished; content state=%s", self.progress["state"])
                        break
                except Exception as error:
                    LOGGER.exception("Coordinator cycle failed; retrying: %s", error)
                    self.progress["last_error"] = f"{type(error).__name__}: {error}"
                    self.save()
                time.sleep(poll_seconds)
