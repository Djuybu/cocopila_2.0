"""Estimate storage before crawling ViBioMIR and export a local RAG dataset."""
import argparse
import json
import logging

from src.data.rag_dataset import crawl, estimate, export
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/vibiomir_rag.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    sizing = commands.add_parser("estimate", help="Fetch a small stratified sample and project storage")
    sizing.add_argument("--state-dir", default="outputs/vibiomir_size_estimate")
    sizing.add_argument("--sample-per-host", type=int)
    sizing.add_argument("--retry-failed", action="store_true")
    sizing.add_argument("--plan-only", action="store_true", help="Scan catalogue and save sample plan without HTTP requests")
    fetching = commands.add_parser("crawl", help="Fetch a bounded prefix or explicitly all catalogue URLs")
    fetching.add_argument("--state-dir", default="outputs/vibiomir_crawl")
    limit = fetching.add_mutually_exclusive_group(required=True)
    limit.add_argument("--max-urls", type=int)
    limit.add_argument("--all", action="store_true", dest="all_urls")
    fetching.add_argument("--retry-failed", action="store_true")
    packing = commands.add_parser("export", help="Export acquired pages without further network requests")
    packing.add_argument("--state-dir", required=True)
    packing.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.command == "estimate":
        if args.sample_per_host is not None:
            config["rag"]["sample_per_host"] = args.sample_per_host
        report = estimate(config, args.state_dir, retry_failed=args.retry_failed, plan_only=args.plan_only)
        if args.plan_only:
            print(json.dumps(report, indent=2))
            return
        print(f"Report: {args.state_dir}/estimate.md")
        print(f"Sample: {report['sample_urls']}; weighted success: {report['weighted_success_fraction']:.1%}")
        print(f"Current-policy working disk projection: {report['storage']['working_disk_bytes_with_30_percent_reserve'] / 2**30:.2f} GiB")
        print("This is not a full-access corpus size; review failures and full-coverage scenarios in the report.")
    elif args.command == "crawl":
        print(json.dumps(crawl(config, args.state_dir, max_urls=args.max_urls, all_urls=args.all_urls,
                               retry_failed=args.retry_failed), indent=2))
    else:
        print(json.dumps(export(config, args.state_dir, args.output_dir)["counts"], indent=2))


if __name__ == "__main__":
    main()
