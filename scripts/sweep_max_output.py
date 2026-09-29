"""CLI entrypoint for P2-05: Sweep max-N output cap & tradeoff analysis.

Evaluates candidate selection at fixed threshold and fallback across max_chunks limits,
records Macro Precision, Recall, F1, F2, chunk volume, and evaluates precision-recall tradeoff.
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
from src.scoring.ablation import analyze_precision_recall_tradeoff, sweep_max_output
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep_max_output")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-05: Sweep max-N output cap and analyze precision/recall tradeoff"
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
        help="Score threshold (if not provided, will try reading from plateau_summary.json)",
    )
    parser.add_argument(
        "--fallback",
        type=int,
        default=None,
        help="Minimum fallback Top-N (if not provided, will check fallback_ablation.json, default: 0)",
    )
    parser.add_argument(
        "--max-values",
        type=str,
        default="1,2,3,5,8,10,15,20,50,100",
        help="Comma-separated list of max chunk cap values (default: '1,2,3,5,8,10,15,20,50,100')",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination path for max_output_sweep.csv",
    )
    parser.add_argument(
        "--output-json",
        "--output-summary",
        dest="output_json",
        type=Path,
        default=None,
        help="Destination path for max_output_tradeoff.json",
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
        fallback_ablation_path = run_dir / "fallback_ablation.json"
    else:
        scored_path = args.scored_file.resolve()
        labels_path = args.labels
        registry_path = args.registry
        run_dir = scored_path.parent
        summary_path = run_dir / "plateau_summary.json"
        fallback_ablation_path = run_dir / "fallback_ablation.json"

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

    # Determine fallback
    fallback = args.fallback
    if fallback is None:
        if fallback_ablation_path.exists():
            try:
                fb_data = read_json(fallback_ablation_path)
                fallback = fb_data.get("best_fallback_config", {}).get("fallback", 0)
                logger.info("Auto-loaded best fallback from %s: %s", fallback_ablation_path, fallback)
            except Exception as e:
                logger.warning("Could not parse fallback_ablation.json: %s", e)
        if fallback is None:
            fallback = 0

    max_list = [int(x.strip()) for x in args.max_values.split(",") if x.strip()]

    output_csv = args.output_csv or (run_dir / "max_output_sweep.csv")
    output_json = args.output_json or (run_dir / "max_output_tradeoff.json")

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

    logger.info("Running max output sweep (threshold=%s, fallback=%s, max_values=%s)...", threshold, fallback, max_list)
    sweep_df = sweep_max_output(
        scored_records=scored_records,
        labels=labels,
        threshold=threshold,
        fallback=fallback,
        maximums=max_list,
        internal_to_official=internal_to_official,
        chunk_to_doc=chunk_to_doc,
    )

    tradeoff_report = analyze_precision_recall_tradeoff(sweep_df)
    tradeoff_report["threshold"] = threshold
    tradeoff_report["fallback"] = fallback
    tradeoff_report["sweep_table"] = sweep_df.to_dict(orient="records")

    # Save outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_csv, index=False)
    logger.info("Saved max output sweep CSV to %s", output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, tradeoff_report)
    logger.info("Saved max output tradeoff JSON to %s", output_json)

    # Print summary
    print("\n" + "=" * 80)
    print("P2-05: MAX-N OUTPUT CAP TRADEOFF REPORT")
    print("=" * 80)
    print(f"Fixed Threshold: {threshold}, Fallback: {fallback}")
    print(f"Best Cap (max={tradeoff_report['best_max_chunks']}): "
          f"F2={tradeoff_report['best_macro_f2']:.4f}, "
          f"P={tradeoff_report['best_macro_precision']:.4f}, "
          f"R={tradeoff_report['best_macro_recall']:.4f}")
    print(f"Baseline Unlimited (max={tradeoff_report['unlimited_baseline']['max_chunks']}): "
          f"F2={tradeoff_report['unlimited_baseline']['macro_f2']:.4f}, "
          f"P={tradeoff_report['unlimited_baseline']['macro_precision']:.4f}, "
          f"R={tradeoff_report['unlimited_baseline']['macro_recall']:.4f}")
    print(f"Precision Gain: {tradeoff_report['tradeoff']['precision_gain']:+.4f}, "
          f"Recall Loss: {tradeoff_report['tradeoff']['recall_loss']:.4f}, "
          f"F2 Delta: {tradeoff_report['tradeoff']['f2_delta']:+.4f}")
    print(f"Analysis: {tradeoff_report['analysis']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
