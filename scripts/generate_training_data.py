"""CLI entrypoint for P2-06: Training data generation & leakage check.

Converts queries, corpus chunks, and labels into labeled training pairs (reranker_train.jsonl)
with strict split validation and detailed statistics reporting.
"""
import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_records
from src.training.data_generator import (
    generate_reranker_pairs,
    report_data_statistics,
    validate_no_leakage,
    write_training_data,
)
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_training_data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-06: Generate labeled positive/negative pairs for reranker training"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to directory containing queries.json, chunks.json (or chunk_corpus.jsonl), labels.json",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries.json",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Path to chunks.json or chunk_corpus.jsonl",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to labels.json",
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=None,
        help="Path to split_report.json (optional)",
    )
    parser.add_argument(
        "--target-split",
        type=str,
        default="train",
        help="Split to generate data for ('train', 'val', or 'all', default: 'train')",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=5,
        help="Number of negative chunks per positive (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output reranker_train.jsonl",
    )
    parser.add_argument(
        "--report",
        "--output-report",
        dest="report",
        type=Path,
        default=None,
        help="Path to output statistics report JSON",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input paths
    if args.data_dir:
        data_dir = args.data_dir.resolve()
        queries_path = args.queries or (data_dir / "queries.json")
        labels_path = args.labels or (data_dir / "labels.json")
        chunks_candidates = [
            data_dir / "chunks.json",
            data_dir / "chunk_corpus.jsonl",
            data_dir / "chunks.jsonl",
        ]
        chunks_path = args.chunks
        if not chunks_path:
            for p in chunks_candidates:
                if p.exists():
                    chunks_path = p
                    break
        split_report_path = args.split_report or (data_dir / "split_report.json")
    else:
        queries_path = args.queries
        chunks_path = args.chunks
        labels_path = args.labels
        split_report_path = args.split_report
        data_dir = queries_path.parent if queries_path else Path.cwd()

    if not queries_path or not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not chunks_path or not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not labels_path or not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    target_split = None if args.target_split.lower() in ("all", "none") else args.target_split.lower()

    output_path = args.output or (data_dir / f"reranker_{args.target_split.lower()}.jsonl")
    report_path = args.report or (data_dir / f"reranker_{args.target_split.lower()}_stats.json")

    logger.info("Loading queries from %s...", queries_path)
    queries = load_records(queries_path)
    logger.info("Loading chunks from %s...", chunks_path)
    chunks = load_records(chunks_path)
    logger.info("Loading labels from %s...", labels_path)
    labels = read_json(labels_path)

    split_info = None
    if split_report_path and split_report_path.exists():
        logger.info("Loading split report from %s...", split_report_path)
        split_info = read_json(split_report_path)

    logger.info("Generating reranker pairs for split='%s' (neg_ratio=%d, seed=%d)...", target_split, args.neg_ratio, args.seed)
    pairs = generate_reranker_pairs(
        queries=queries,
        chunks=chunks,
        labels=labels,
        split_info=split_info,
        target_split=target_split,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    stats = report_data_statistics(pairs)
    if target_split in ("train", "val"):
        other_split = "val" if target_split == "train" else "train"
        other_pairs = generate_reranker_pairs(
            queries, chunks, labels, split_info=split_info, target_split=other_split,
            neg_ratio=args.neg_ratio, seed=args.seed,
        )
        train_pairs, val_pairs = (pairs, other_pairs) if target_split == "train" else (other_pairs, pairs)
        stats["leakage_check"] = validate_no_leakage(train_pairs, val_pairs, split_info)
    stats["target_split"] = args.target_split
    stats["neg_ratio"] = args.neg_ratio
    stats["seed"] = args.seed

    write_training_data(pairs, output_path)
    logger.info("Saved %d pairs to %s", len(pairs), output_path)

    write_json(report_path, stats)
    logger.info("Saved data report to %s", report_path)

    print("\n" + "=" * 80)
    print("P2-06: RERANKER TRAINING DATA GENERATION REPORT")
    print("=" * 80)
    print(f"Target Split           : {args.target_split.upper()}")
    print(f"Total Generated Pairs  : {stats['total_pairs']}")
    print(f"Unique Queries         : {stats['num_queries']}")
    print(f"Positive Pairs         : {stats['num_positives']}")
    print(f"Negative Pairs         : {stats['num_negatives']}")
    print(f"Positive:Negative Ratio: {stats['pos_neg_ratio']}")
    print(f"Avg Query Words/Chars  : {stats.get('avg_query_words', 0)} / {stats.get('avg_query_chars', 0)}")
    print(f"Avg Chunk Words/Chars  : {stats.get('avg_chunk_words', 0)} / {stats.get('avg_chunk_chars', 0)}")
    print(f"Output File            : {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
