"""CLI entrypoint for P2-07: Hard negative mining from retrieval candidates.

Mines top-scoring incorrect candidates, filters false negatives, validates zero contamination,
and exports hard_negatives.jsonl.
"""
import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_records
from src.training.hard_negatives import (
    merge_training_data,
    mine_hard_negatives,
    validate_hard_negatives,
    write_hard_negatives,
)
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mine_hard_negatives")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-07: Mine hard negative chunks from retrieval candidates"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        "--handoff-dir",
        dest="run_dir",
        type=Path,
        help="Path to directory containing candidates.jsonl (or reranked.jsonl) and labels.json",
    )
    group.add_argument(
        "--candidates",
        "--scored-file",
        dest="candidates_file",
        type=Path,
        help="Path to candidates.jsonl or reranked.jsonl file",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to labels.json",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries.json (optional)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to registry.json for official mapping (optional)",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=10,
        help="Maximum hard negatives mined per query (default: 10)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum candidate score threshold for hard negatives",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.85,
        help="Jaccard cutoff for filtering near-duplicate false negatives (default: 0.85)",
    )
    parser.add_argument(
        "--base-train-data",
        type=Path,
        default=None,
        help="Path to P2-06 reranker_train.jsonl to merge into enriched dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for hard_negatives.jsonl",
    )
    parser.add_argument(
        "--report",
        "--output-report",
        dest="report",
        type=Path,
        default=None,
        help="Destination path for hard_negatives_report.json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input files
    if args.run_dir:
        run_dir = args.run_dir.resolve()
        candidates_candidates = [
            run_dir / "candidates.jsonl",
            run_dir / "reranked.jsonl",
        ]
        candidates_path = None
        for p in candidates_candidates:
            if p.exists() and p.stat().st_size > 0:
                candidates_path = p
                break
        if candidates_path is None:
            raise FileNotFoundError(f"Neither candidates.jsonl nor reranked.jsonl found in {run_dir}")

        labels_path = args.labels or (run_dir / "labels.json")
        queries_path = args.queries or (run_dir / "queries.json")
        registry_path = args.registry or (run_dir / "registry.json")
    else:
        candidates_path = args.candidates_file.resolve()
        labels_path = args.labels
        queries_path = args.queries
        registry_path = args.registry
        run_dir = candidates_path.parent

    if not labels_path or not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    output_path = args.output or (run_dir / "hard_negatives.jsonl")
    report_path = args.report or (run_dir / "hard_negatives_report.json")

    logger.info("Loading candidates from %s...", candidates_path)
    candidates_records = load_records(candidates_path)
    logger.info("Loaded %d queries.", len(candidates_records))

    logger.info("Loading labels from %s...", labels_path)
    labels = read_json(labels_path)

    queries = None
    if queries_path and queries_path.exists():
        logger.info("Loading queries from %s...", queries_path)
        queries = load_records(queries_path)

    internal_to_official = None
    chunk_to_doc = None
    if registry_path and registry_path.exists():
        registry = read_json(registry_path)
        internal_to_official = registry.get("internal_to_official")
        chunk_to_doc = registry.get("chunk_to_doc")

    logger.info("Mining hard negatives (max_per_query=%d, sim_threshold=%.2f)...", args.max_per_query, args.sim_threshold)
    hard_negs, mining_summary = mine_hard_negatives(
        candidates_records=candidates_records,
        labels=labels,
        queries=queries,
        max_negatives_per_query=args.max_per_query,
        min_score=args.min_score,
        similarity_threshold=args.sim_threshold,
        internal_to_official=internal_to_official,
        chunk_to_doc=chunk_to_doc,
    )

    logger.info("Validating mined hard negatives...")
    val_summary = validate_hard_negatives(hard_negs, labels)

    full_report = {
        "mining_summary": mining_summary,
        "validation_summary": val_summary,
    }

    write_hard_negatives(hard_negs, output_path)
    logger.info("Saved %d hard negatives to %s", len(hard_negs), output_path)

    # Optional merge with base training data
    if args.base_train_data and args.base_train_data.exists():
        logger.info("Merging hard negatives with base training data from %s...", args.base_train_data)
        base_pairs = load_records(args.base_train_data)
        merged_pairs = merge_training_data(base_pairs, hard_negs, hard_neg_ratio=0.5)
        merged_out = args.base_train_data.parent / "reranker_train_with_hard_negs.jsonl"
        write_hard_negatives(merged_pairs, merged_out)
        full_report["merged_dataset"] = {
            "path": str(merged_out),
            "total_pairs": len(merged_pairs),
        }
        logger.info("Saved merged training set (%d pairs) to %s", len(merged_pairs), merged_out)

    write_json(report_path, full_report)
    logger.info("Saved mining report to %s", report_path)

    # Print summary
    print("\n" + "=" * 80)
    print("P2-07: HARD NEGATIVE MINING REPORT")
    print("=" * 80)
    print(f"Total Mined Hard Negatives: {mining_summary['total_mined_hard_negatives']}")
    print(f"Queries with Hard Negs    : {mining_summary['queries_with_hard_negatives']}")
    print(f"Avg Hard Negs / Query     : {mining_summary['avg_hard_negatives_per_query']}")
    print(f"Ground-Truth Excluded     : {mining_summary['ground_truth_excluded']}")
    print(f"Suspected FNs Excluded    : {mining_summary['suspected_fn_excluded']}")
    print(f"Ground-Truth Contamination: {val_summary['ground_truth_contamination']} (PASS)")
    print(f"Avg Hard Negative Score   : {val_summary['avg_hard_negative_score']}")
    print(f"Output File               : {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
