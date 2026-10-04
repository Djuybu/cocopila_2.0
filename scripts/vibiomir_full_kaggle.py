"""Plan and coordinate a full ViBioMIR crawl on private Kaggle CPU notebooks."""
import argparse
import json
import logging
from pathlib import Path

from src.data.full_crawl import plan_full_crawl
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/vibiomir_full_crawl.yaml")
    parser.add_argument("--root", default="outputs/vibiomir_full_kaggle")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--shard-size", type=int)
    run = commands.add_parser("run")
    run.add_argument("--owner", required=True)
    run.add_argument("--all", action="store_true", required=True, help="Explicitly authorize all catalogue URLs")
    run.add_argument("--poll-seconds", type=int, default=30)
    commands.add_parser("status")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        p = json.loads((root/"progress.json").read_text())
        print(json.dumps({k: v for k, v in p.items() if k != "jobs"}, ensure_ascii=False, indent=2))
        for id, job in p["jobs"].items():
            if job["state"] != "done":
                print(id, job["state"], job["ref"], job.get("error", ""))
        return
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "plan":
        result = plan_full_crawl(config, root/"plan.json", args.shard_size or config["full_crawl"]["shard_size"])
        print(f"Full catalogue: {result['population_urls']:,} URLs; {len(result['shards'])} shards; {len(result['hosts'])} hosts")
    else:
        if args.poll_seconds < 10:
            parser.error("poll-seconds must be >= 10")
        from src.data.kaggle_full_runner import FullKaggleRunner
        FullKaggleRunner(Path(__file__).resolve().parents[1], config, root, args.owner).run(args.poll_seconds)


if __name__ == "__main__":
    main()
