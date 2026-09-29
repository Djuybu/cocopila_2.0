"""CLI entrypoint for P2-04: Sweep minimum fallback Top-N & ablation analysis.

Evaluates candidate selection at fixed threshold across fallback N_min values,
records Macro Precision, Recall, F1, F2, chunk volume, and performs threshold-only vs fallback ablation.
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
from src.scoring.ablation import ablation_threshold_vs_fallback, sweep_fallback
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep_fallback")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-04: Sweep minimum fallback Top-N and perform ablation analysis"
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
        "--threshold",
        type=float,
        default=None,
        help="Score threshold (if not provided, will try reading recommended_threshold from run-dir/plateau_summary.json)",
    )
    parser.add_argument(
        "--fallbacks",
        type=str,
        default="0,1,2,3,5,10",
        help="Comma-separated list of fallback Top-N values to evaluate (default: '0,1,2,3,5,10')",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=999999,
        help="Upper chunk cap per query (default: 999999 = unlimited)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination path for fallback_sweep.csv",
    )
    parser.add_argument(
        "--output-json",
        "--output-summary",
        dest="output_json",
        type=Path,
        default=None,
        help="Destination path for fallback_ablation.json",
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
        summary_path = run_dir / "plateau_summary.json"
    else:
        scored_path = args.scored_file.resolve()
        labels_path = args.labels
        registry_path = args.registry
        run_dir = scored_path.parent
        summary_path = run_dir / "plateau_summary.json"

    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Determine threshold
    threshold = args.threshold
    if threshold is None:
        if summary_path.exists():
            try:
                summ = read_json(summary_path)
                threshold = summ.get("recommended_config", {}).get("threshold") or summ.get("plateau_info", {}).get("recommended_threshold")
                logger.info("Auto-loaded recommended threshold from %s: %s", summary_path, threshold)
            except Exception as e:
                logger.warning("Could not parse plateau_summary.json: %s", e)
        if threshold is None:
            threshold = 0.5
            logger.info("Using default threshold=0.5")

    fallback_list = [int(x.strip()) for x in args.fallbacks.split(",") if x.strip()]

    output_csv = args.output_csv or (run_dir / "fallback_sweep.csv")
    output_json = args.output_json or (run_dir / "fallback_ablation.json")

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

    logger.info("Running ablation threshold vs fallback (threshold=%s, fallbacks=%s)...", threshold, fallback_list)
    ablation_report = ablation_threshold_vs_fallback(
        scored_records=scored_records,
        labels=labels,
        threshold=threshold,
        fallbacks=fallback_list,
        max_chunks=args.max_chunks,
        internal_to_official=internal_to_official,
        chunk_to_doc=chunk_to_doc,
    )

    sweep_df = pd.DataFrame(ablation_report["sweep_table"])

    # Output outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_csv, index=False)
    logger.info("Saved fallback sweep CSV to %s", output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, ablation_report)
    logger.info("Saved fallback ablation JSON to %s", output_json)

    # Print summary
    print("\n" + "=" * 80)
    print("P2-04: FALLBACK TOP-N ABLATION REPORT")
    print("=" * 80)
    print(f"Fixed Threshold: {threshold}")
    print(f"Baseline (fb=0): F2={ablation_report['baseline_threshold_only']['macro_f2']:.4f}, "
          f"P={ablation_report['baseline_threshold_only']['macro_precision']:.4f}, "
          f"R={ablation_report['baseline_threshold_only']['macro_recall']:.4f}, "
          f"Zero-chunk queries={ablation_report['baseline_threshold_only']['zero_chunk_queries']}")
    print(f"Best Fallback (fb={ablation_report['best_fallback_config']['fallback']}): "
          f"F2={ablation_report['best_fallback_config']['macro_f2']:.4f}, "
          f"P={ablation_report['best_fallback_config']['macro_precision']:.4f}, "
          f"R={ablation_report['best_fallback_config']['macro_recall']:.4f}, "
          f"Zero-chunk queries={ablation_report['best_fallback_config']['zero_chunk_queries']}")
    print(f"Delta: F2={ablation_report['delta']['delta_f2']:+.4f}, "
          f"Recall={ablation_report['delta']['delta_recall']:+.4f}, "
          f"Zero-queries eliminated={ablation_report['delta']['zero_queries_eliminated']}")
    print(f"Verdict: {ablation_report['verdict']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
