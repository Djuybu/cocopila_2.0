"""Build a credential-free, self-scheduling Kaggle cloud coordinator notebook."""
import json
import re
from src.data.kaggle_crawl import SOURCE_FILES, _cell, build_notebook

CLOUD_FILES = SOURCE_FILES + (
    "src/data/kaggle_crawl.py", "src/data/kaggle_full_runner.py",
    "src/data/kaggle_cloud_runner.py", "src/data/kaggle_cloud_notebook.py",
    "configs/data/vibiomir.yaml", "configs/data/vibiomir_rag.yaml",
    "configs/data/vibiomir_full_crawl.yaml",
)


def build_cloud_notebook(project, *, owner, control_dataset):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-z0-9-]+/[0-9]+", control_dataset):
        raise ValueError("Control dataset must pin owner/dataset/version")
    nb, meta = build_notebook(project, owner=owner, slug="vibiomir-cloud-coordinator", source_files=CLOUD_FILES)
    nb["cells"] = [
        _cell("markdown", """
        # ViBioMIR Cloud Coordinator

        Runs entirely on Kaggle: collects validated shards, updates private corpus
        partitions and persists coordinator state. Enable Kaggle's native Daily
        schedule to resume automatically. No local computer needs to stay online.

        One-time setup: Add-ons → Secrets → enable `KAGGLE_API_TOKEN` for this
        notebook. Use a Kaggle API token with dataset and notebook write permission.
        Settings → Schedule a notebook to run → Daily; Save & Run All once.
        Keep this notebook private. Secrets are never copied to outputs or code.
        """),
        _cell("code", """
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "kaggle>=2.2,<3", "numpy>=1.26", "PyYAML>=6", "pyarrow>=16",
                        "requests>=2.32", "trafilatura>=2.1,<3", "pypdf>=5,<7", "protego>=0.6,<1"], check=True)
        """),
        nb["cells"][2],
        _cell("code", """
        import os, time
        from kaggle_secrets import UserSecretsClient
        error = None
        for attempt in range(3):
            try:
                os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
                error = None
                break
            except Exception as failure:
                error = failure
                if type(failure).__name__ != "ConnectionError" or getattr(failure.__cause__, "code", None) in (401, 403):
                    break
                if attempt < 2:
                    time.sleep(3 * (attempt+1))
        try:
            if error is not None:
                raise error
        except Exception as error:
            kind = type(error).__name__
            message = str(error).lower()
            service_code = getattr(error.__cause__, "code", None)
            reason = ("secret_not_attached" if "no user secrets exist" in message else
                      "secret_permission_denied" if service_code in (401, 403) else
                      "secret_request_rejected" if service_code == 400 else
                      "runtime_permission_missing" if kind == "CredentialError" else
                      "secret_service_connection_error" if kind == "ConnectionError" else "secret_lookup_failed")
            Path("/kaggle/working/coordinator_status.json").write_text(json.dumps(
                {"state":"authentication_required", "secret_label":"KAGGLE_API_TOKEN", "reason":reason,
                 "error_type":kind, "service_http_status":service_code,
                 "runtime_secret_session":bool(os.environ.get("KAGGLE_USER_SECRETS_TOKEN"))}))
            raise RuntimeError(f"Kaggle Secret lookup failed: {reason} ({kind}). Enable KAGGLE_API_TOKEN for this notebook and Save & Run All.") from None
        from kaggle import api
        print("Kaggle API credential loaded from Secrets; value is not displayed or persisted.")
        """),
        _cell("code", f"""
        from src.data.kaggle_cloud_runner import CloudKaggleRunner, restore_snapshot, wait_ready
        CONTROL_DATASET = {control_dataset!r}
        control_ref = "/".join(CONTROL_DATASET.split("/")[:2])
        wait_ready(api, control_ref)
        version = json.loads(api.dataset_status(control_ref, format="json(current_version_number)"))["current_version_number"]
        control = Path("/tmp/vibiomir_latest_control")
        api.dataset_download_files(f"{{control_ref}}/{{version}}", path=str(control), quiet=True, unzip=True)
        ROOT = Path("/tmp/vibiomir_cloud")
        CONFIG = restore_snapshot(control, ROOT, catalogue_root=Path("/kaggle/input"))
        CONFIG["cloud_crawl"]["handoff_mode"] = "scheduled"
        print("Loaded latest cloud checkpoint version:", version)
        runner = CloudKaggleRunner(TOOL_ROOT, CONFIG, ROOT, {owner!r}, api=api)
        runner.run()
        Path("/kaggle/working/coordinator_status.json").write_text(json.dumps(
            {{"state":runner.progress["state"], "cloud":runner.cloud}}, ensure_ascii=False, indent=2))
        """),
    ]
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cloud-vibiomir-{i:02d}"
    meta.update(title="ViBioMIR Cloud Coordinator", code_file="ViBioMIR_Cloud_Coordinator.ipynb",
                dataset_sources=[control_dataset, f"{owner}/vibiomir-rag-corpus/1"], kernel_sources=[])
    return nb, meta
