"""One-time transfer of local crawl progress to a natively scheduled Kaggle notebook."""
import argparse
import json
from pathlib import Path
import shutil

from src.data.kaggle_cloud_notebook import build_cloud_notebook
from src.data.kaggle_cloud_runner import CloudKaggleRunner, migrate_local
from src.data.kaggle_full_runner import save_atomic
from src.utils.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner",required=True)
    parser.add_argument("--local-root",default="outputs/vibiomir_full_kaggle")
    parser.add_argument("--cloud-root",default="outputs/vibiomir_cloud_deployment")
    args=parser.parse_args()
    local,root=Path(args.local_root),Path(args.cloud_root)
    root.mkdir(parents=True,exist_ok=True)
    if (root/"progress.json").exists():
        raise ValueError("Cloud deployment already exists; resume its notebook instead of importing again")
    # The local controller must be stopped before this point. The existing cloud
    # workers keep running; their references are retained in the progress snapshot.
    from src.data.kaggle_full_runner import coordinator_lock
    with coordinator_lock(local):
        for name in ("plan.json","progress.json"):
            shutil.copy2(local/name,root/name)
    config=load_config("configs/data/vibiomir_full_crawl.yaml")
    config["cloud_crawl"]={"session_seconds":32400,"handoff_mode":"scheduled","native_first_run_seconds":60,"partition_bytes":1024**3,"partition_shards":4}
    project=Path(__file__).resolve().parents[1]
    runner=CloudKaggleRunner(project,config,root,args.owner)
    pinned=migrate_local(runner,local/"dataset")
    nb,meta=build_cloud_notebook(project,owner=args.owner,control_dataset=pinned)
    bundle=root/"notebook"
    bundle.mkdir(exist_ok=True)
    save_atomic(bundle/meta["code_file"],nb)
    save_atomic(bundle/"kernel-metadata.json",meta)
    save_atomic(project/"notebooks"/meta["code_file"],nb)
    response=runner.api.kernels_push(str(bundle),timeout=43200)
    if response is None or response.error or response.invalid_dataset_sources:
        raise RuntimeError("Failed to deploy cloud coordinator notebook")
    print(f"Cloud coordinator: https://www.kaggle.com/code/{meta['id']}",flush=True)
    print(f"Version: {response.version_number}; control snapshot: {pinned}",flush=True)
    print("Enable KAGGLE_API_TOKEN in this notebook's Secrets if not configured yet.",flush=True)
    print("Enable Settings -> Schedule a notebook to run -> Daily; Save & Run All once.",flush=True)


if __name__=="__main__":
    main()
