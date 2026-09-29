"""CLI entrypoint for P2-09: Cross-validation and bootstrap threshold tuning.

Performs K-fold or bootstrap cross-validation of threshold tuning to prevent
overfitting to a single validation split, reporting mean/std F2 and stable threshold.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.data.loader import load_records
from src.scoring.cv_threshold import cross_validate_threshold
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cv_threshold")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-09: Cross-validate reranker threshold tuning and report mean/std F2"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        "--handoff-dir",
        dest="run_dir",
        type=Path,
        help="Path to directory containing reranked.jsonl (or candidates.jsonl) and labels.json",
    )
    group.add_argument(
        "--scored-file",
        "--candidates",
        dest="scored_file",
        type=Path,
        help="Path to reranked.jsonl or candidates.jsonl file",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to labels.json (optional if --run-dir is specified)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to registry.json for ID mapping (optional)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="kfold",
        choices=["kfold", "bootstrap"],
        help="Resampling method ('kfold' or 'bootstrap', default: 'kfold')",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for K-fold CV (default: 5)",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=20,
        help="Number of rounds for bootstrap resampling (default: 20)",
    )
    parser.add_argument(
        "--fallback",
        type=int,
        default=0,
        help="Minimum fallback Top-N (default: 0)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Upper chunk cap per query (default: None = unlimited)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Grid search steps between min and max scores (default: 50)",
    )
    parser.add_argument(
        "--min-th",
        type=float,
        default=None,
        help="Lower threshold bound",
    )
    parser.add_argument(
        "--max-th",
        type=float,
        default=None,
        help="Upper threshold bound",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold partitioning (default: 42)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination path for cv_threshold_sweep.csv",
    )
    parser.add_argument(
        "--output-json",
        "--output-summary",
        dest="output_json",
        type=Path,
        default=None,
        help="Destination path for cv_threshold_report.json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input files
    if args.run_dir:
        run_dir = args.run_dir.resolve()
        scored_candidates = [
            run_dir / "reranked.jsonl",
            run_dir / "candidates.jsonl",
        ]
        scored_path = None
        for p in scored_candidates:
            if p.exists() and p.stat().st_size > 0:
                scored_path = p
                break
        if scored_path is None:
            raise FileNotFoundError(f"Neither reranked.jsonl nor candidates.jsonl with content found in {run_dir}")

        labels_path = args.labels or (run_dir / "labels.json")
        registry_path = args.registry or (run_dir / "registry.json")
    else:
        scored_path = args.scored_file.resolve()
        labels_path = args.labels
        registry_path = args.registry
        run_dir = scored_path.parent

    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    output_csv = args.output_csv or (run_dir / "cv_threshold_sweep.csv")
    output_json = args.output_json or (run_dir / "cv_threshold_report.json")

    logger.info("Loading scored records from %s...", scored_path)
    scored_records = load_records(scored_path)
    logger.info("Loaded %d queries.", len(scored_records))

    logger.info("Loading ground truth labels from %s...", labels_path)
    labels = read_json(labels_path)

    internal_to_official = None
    chunk_to_doc = None
    if registry_path and registry_path.exists():
        logger.info("Loading registry from %s...", registry_path)
        registry = read_json(registry_path)
        internal_to_official = registry.get("internal_to_official")
        chunk_to_doc = registry.get("chunk_to_doc")

    logger.info("Running Cross-Validation Threshold Tuning (method=%s, seed=%d)...", args.method, args.seed)
    cv_df, report = cross_validate_threshold(
        scored_records=scored_records,
        labels=labels,
        steps=args.steps,
        min_th=args.min_th,
        max_th=args.max_th,
        fallback=args.fallback,
        max_chunks=args.max_chunks,
        method=args.method,
        n_folds=args.n_folds,
        n_rounds=args.n_rounds,
        seed=args.seed,
        internal_to_official=internal_to_official,
        chunk_to_doc=chunk_to_doc,
    )

    # Save outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(output_csv, index=False)
    logger.info("Saved CV sweep CSV to %s", output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    logger.info("Saved CV report JSON to %s", output_json)

    # Print summary
    stable = report["recommended_stable_threshold"]
    pure = report["pure_peak_threshold"]
    oof = report["out_of_fold_generalization"]

    print("\n" + "=" * 80)
    print("P2-09: CROSS-VALIDATION THRESHOLD REPORT")
    print("=" * 80)
    print(f"Method: {report['method'].upper()} ({report['n_splits']} splits over {report['total_queries']} queries)")
    print(f"Recommended Stable Threshold: {stable['threshold']}")
    print(f"  • Mean F2 ± Std F2 : {stable['mean_f2']:.4f} ± {stable['std_f2']:.4f}")
    print(f"  • 95% Confidence   : [{stable['confidence_interval_95'][0]:.4f}, {stable['confidence_interval_95'][1]:.4f}]")
    print(f"  • Precision / Recall: P={stable['mean_precision']:.4f}, R={stable['mean_recall']:.4f}")
    print(f"  • Avg Output Chunks: {stable['avg_chunks_per_query']:.2f}")
    print(f"  • Stability Score  : {stable['stability_score']:.4f}")
    print("-" * 80)
    print(f"Peak Validation Threshold    : {pure['threshold']} (Mean F2={pure['mean_f2']:.4f} ± {pure['std_f2']:.4f})")
    print(f"Out-of-Fold Generalization   : Mean F2={oof['mean_oof_f2']:.4f} ± {oof['std_oof_f2']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
