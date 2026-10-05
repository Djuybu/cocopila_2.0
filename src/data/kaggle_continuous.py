"""Short cloud orchestration ticks; CPU Kaggle workers run around the clock."""
import json
import logging
from pathlib import Path
import time
import uuid

from src.data.kaggle_cloud_runner import CloudKaggleRunner, restore_snapshot, wait_ready
from src.data.kaggle_full_runner import coordinator_lock, save_atomic

LOGGER = logging.getLogger(__name__)
PARK_MARKER = "VIBIOMIR_COORDINATOR_MOVED_TO_ACTIONS"
ACTIVE_STATUSES = ("RUNNING", "QUEUED", "STARTING")


def kernel_details(api, ref):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    owner, slug = ref.split("/", 1)
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        response = client.kernels.kernels_api_client.get_kernel(request)
    return response.metadata, response.blob.source


def kernel_exists(api, ref):
    _, slug = ref.split("/", 1)
    kernels = api.kernels_list(mine=True, search=slug, page_size=100)
    return any(item.ref == ref for item in kernels or [])


def version_status(api, ref, version):
    # Kaggle CLI's kernels_status ignores the parsed version. The SDK supports
    # version_label="v10", which is essential during coordinator migration.
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    owner, slug = ref.split("/", 1)
    with api.build_kaggle_client() as client:
        request = ApiGetKernelSessionStatusRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.version_label = f"v{int(version)}"
        response = client.kernels.kernels_api_client.get_kernel_session_status(request)
    return str(response.status).upper()


def is_active(status):
    return any(name in status.upper() for name in ACTIVE_STATUSES)


def park_native_coordinator(api, ref, folder, workflow_url):
    """Make future Daily runs harmless, without exposing or reattaching Secrets."""
    try:
        metadata, source = kernel_details(api, ref)
    except Exception as error:
        if getattr(getattr(error, "response", None), "status_code", None) != 404:
            raise
        if kernel_exists(api, ref):
            raise RuntimeError("Kaggle still lists the native coordinator but its latest source is unavailable") from None
        LOGGER.info("Native coordinator no longer exists on Kaggle; adopting its last durable checkpoint")
        return True
    if PARK_MARKER in source:
        return True
    if is_active(str(api.kernels_status(ref).status)):
        return False  # Let its current cycle finish and persist its latest state.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nb = {"nbformat":4, "nbformat_minor":5,
          "metadata":{"kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"}},
          "cells":[{"cell_type":"code", "id":"continuous-controller",
                    "metadata":{}, "execution_count":None, "outputs":[],
                    "source":f"# {PARK_MARKER}\nimport json\nfrom pathlib import Path\n"
                    f"status = {{'state':'managed_by_continuous_cpu_coordinator', 'workflow':{workflow_url!r}}}\n"
                    "Path('/kaggle/working/coordinator_status.json').write_text(json.dumps(status))\nprint(status)\n"}]}
    meta = {"id":ref, "title":"ViBioMIR Continuous CPU Coordinator Status",
            "code_file":"coordinator.ipynb", "language":"python", "kernel_type":"notebook",
            "is_private":True, "enable_gpu":False, "enable_tpu":False, "enable_internet":False,
            "dataset_sources":[], "kernel_sources":[], "competition_sources":[]}
    save_atomic(folder/"coordinator.ipynb", nb)
    save_atomic(folder/"kernel-metadata.json", meta)
    response = api.kernels_push(str(folder), timeout=300)
    if response is None or response.error:
        raise RuntimeError("Could not retire the native Daily coordinator")
    LOGGER.info("Retired native controller as version %s; all future crawl orchestration uses Actions", response.version_number)
    return True


class ContinuousKaggleRunner(CloudKaggleRunner):
    def before_worker_push(self, shard, job, folder):
        # Save the actual submission intent before the network call, so an
        # interrupted Actions job can discover an accepted push on its next tick.
        key = uuid.uuid4().hex
        job["submission_key"] = key
        job.pop("failed_at", None)
        job.pop("error", None)
        meta = json.loads((folder/"kernel-metadata.json").read_text())
        notebook_path = folder/meta["code_file"]
        notebook = json.loads(notebook_path.read_text())
        notebook["cells"][0]["source"] += f"\nVIBIOMIR_SUBMISSION_KEY: {key}\n"
        save_atomic(notebook_path, notebook)
        self.save()
        self.checkpoint()

    def recover_submissions(self):
        for shard_id, job in self.progress["jobs"].items():
            if job["state"] != "submitting":
                continue
            try:
                metadata, source = kernel_details(self.api, job["ref"])
            except Exception as error:
                if getattr(getattr(error, "response", None), "status_code", None) != 404:
                    raise
                metadata, source = None, ""
            key = job.get("submission_key")
            if key and f"VIBIOMIR_SUBMISSION_KEY: {key}" in source:
                job.update(state="running", version=metadata.current_version_number)
                LOGGER.info("Recovered accepted submission for shard %s", shard_id)
            elif time.time()-job["submitted_at"] >= 900:
                job.update(state="failed", error="Submission did not reach Kaggle", failed_at=time.time())
            else:
                raise RuntimeError("Waiting to resolve an interrupted worker submission")

    def retry_failed(self):
        busy_slots = {j["slot"] for j in self.progress["jobs"].values() if j["state"] in {"running", "submitting"}}
        busy_hosts = {host for shard in self.plan["shards"]
                      if self.progress["jobs"].get(str(shard["id"]), {}).get("state") in {"running", "submitting"}
                      for host in shard["hosts"]}
        for shard in self.plan["shards"]:
            job = self.progress["jobs"].get(str(shard["id"]))
            if not job or job["state"] != "failed":
                continue
            if "failed_at" not in job:
                job["failed_at"] = time.time()
            if time.time()-job["failed_at"] < 900 or job.get("failure_restarts", 0) >= 3:
                continue
            free = next((s for s in range(self.config["full_crawl"]["slots"]) if s not in busy_slots), None)
            if free is None or busy_hosts.intersection(shard["hosts"]):
                continue
            restarts = job.get("failure_restarts", 0)+1
            # Restart only this shard if a failed run did not export resumable
            # state. Existing published partial content is replaced by ID.
            job.update(runs=0, failure_restarts=restarts)
            self.submit(shard, free)
            busy_slots.add(free)
            busy_hosts.update(shard["hosts"])

    def before_new_shards(self):
        self.retry_failed()

    def tick(self):
        with coordinator_lock(self.root):
            self.progress.update(execution_platform="GitHub Actions + Kaggle CPU",
                                 coordinator_backend="github_actions", state="building")
            before = json.dumps(self.progress["jobs"], sort_keys=True)
            fingerprint = json.dumps(self.progress["coverage"], sort_keys=True)
            try:
                self.recover_submissions()
                terminal = self.cycle()
                # Keep the master index useful while acquisition is in progress.
                if terminal:
                    wait_ready(self.api, self.dataset_ref)
            finally:
                # Persist mutations even if a later collection/network request
                # failed. A no-change heartbeat does not create a dataset version.
                if (before != json.dumps(self.progress["jobs"], sort_keys=True)
                        or fingerprint != json.dumps(self.progress["coverage"], sort_keys=True)
                        or self.cloud.get("handoff_mode") != "continuous"):
                    self.cloud["handoff_mode"] = "continuous"
                    self.checkpoint()
            return {"state":self.progress["state"], "counts":self.progress["coverage"]["counts"],
                    "target_urls":self.plan["population_urls"], "compute":"CPU",
                    "running_workers":sum(j["state"] in {"running", "submitting"} for j in self.progress["jobs"].values()),
                    "failed_shards":sum(j["state"] == "failed" for j in self.progress["jobs"].values()),
                    "control_ref":self.cloud.get("control_ref")}


def continuous_tick(project, owner, root, api, *, legacy_version=10, workflow_url):
    """Safely adopt the native controller's final checkpoint, then run one tick."""
    ref = f"{owner}/vibiomir-cloud-coordinator"
    try:
        status = version_status(api, ref, legacy_version)
    except Exception as error:
        if getattr(getattr(error, "response", None), "status_code", None) != 404:
            raise
        # A removed coordinator has no live session to wait for. Confirm the
        # owner-scoped notebook listing before allowing automatic adoption.
        if kernel_exists(api, ref):
            raise RuntimeError("Kaggle lists the native coordinator, but its session status is unavailable") from None
        status = "KERNEL_NOT_FOUND"
    if is_active(status):
        return {"state":"waiting_for_legacy_controller", "legacy_version":legacy_version,
                "legacy_status":status, "compute":"CPU"}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not park_native_coordinator(api, ref, root/"retired_controller", workflow_url):
        return {"state":"waiting_for_legacy_controller", "compute":"CPU"}
    control_ref = f"{owner}/vibiomir-crawl-control"
    wait_ready(api, control_ref)
    version = int(json.loads(api.dataset_status(control_ref, format="json(current_version_number)"))["current_version_number"])
    snapshot, catalogue, state = root/"snapshot", root/"catalogue", root/"state"
    api.dataset_download_files(f"{control_ref}/{version}", path=str(snapshot), quiet=True, unzip=True)
    api.dataset_download_files(f"{owner}/vibiomir-rag-corpus/1", path=str(catalogue), quiet=True, unzip=True)
    config = restore_snapshot(snapshot, state, catalogue_root=catalogue)
    config["cloud_crawl"]["handoff_mode"] = "continuous"
    config["full_crawl"]["publish_interval_seconds"] = 1800
    runner = ContinuousKaggleRunner(project, config, state, owner, api=api)
    LOGGER.info("Continuous CPU controller restored checkpoint v%s", version)
    return runner.tick()
