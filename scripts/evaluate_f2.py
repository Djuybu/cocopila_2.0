"""CLI entrypoint for P2-02 Chunk-level F2 evaluation.

Supports direct candidate evaluation from handoff bundles or prediction files.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_records
from src.evaluation.evaluate_f2 import (
    evaluate_candidate_selection,
    evaluate_chunk_predictions,
    export_metrics_csv,
    export_metrics_json,
    format_metrics_table,
)
from src.utils.io import read_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-02: Chunk-level F2 Evaluation (per-query & macro aggregate)"
    )

    # Input sources: either handoff/run directory or explicit file paths
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--handoff-dir",
        "--run-dir",
        dest="run_dir",
        type=Path,
        help="Path to handoff or run directory containing candidates.jsonl and labels.json (e.g. data/p1_p2_handoff_qwen3)",
    )
    group.add_argument(
        "--predictions",
        "--submission",
        dest="predictions",
        type=Path,
        help="Path to JSON file with predictions ([{'id': ..., 'relevant_chunks': [...]}])",
    )
    group.add_argument(
        "--candidates",
        type=Path,
        help="Path to candidates.jsonl or reranked.jsonl",
    )

    # Ground truth labels
    parser.add_argument(
        "--labels",
        "--ground-truth",
        dest="labels",
        type=Path,
        help="Path to labels.json (optional if --handoff-dir is used)",
    )

    # Candidate selection parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Select top-k candidates per query (default: top-1 if evaluating candidates without threshold)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Score cutoff for chunk selection",
    )
    parser.add_argument(
        "--fallback",
        type=int,
        default=None,
        help="Minimum number of chunks to select if fewer pass threshold",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunk cap per query",
    )

    # Registry and mappings
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to registry.json for internal_to_official chunk ID mapping",
    )

    # Output options
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to export per-query and macro evaluation CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to export full evaluation report JSON",
    )
    parser.add_argument(
        "--zero-division",
        type=float,
        default=0.0,
        help="Score to assign when denominator is zero (default: 0.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the ASCII table to stdout",
    )

    return parser.parse_args()


def run_evaluation_cli(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    labels_path = args.labels
    registry_path = args.registry
    mapping = None

    if args.run_dir is not None:
        run_dir = args.run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Handoff/run directory not found: {run_dir}")

        if labels_path is None:
            labels_path = run_dir / "labels.json"
        if registry_path is None and (run_dir / "registry.json").exists():
            registry_path = run_dir / "registry.json"

        # Check candidate files (reranked.jsonl or candidates.jsonl)
        if (run_dir / "reranked.jsonl").exists():
            candidates_path = run_dir / "reranked.jsonl"
        elif (run_dir / "candidates.jsonl").exists():
            candidates_path = run_dir / "candidates.jsonl"
        else:
            raise FileNotFoundError(f"Neither reranked.jsonl nor candidates.jsonl found in {run_dir}")

        candidates = load_records(candidates_path)
    elif args.candidates is not None:
        if labels_path is None:
            raise ValueError("--labels is required when --candidates is provided")
        candidates = load_records(args.candidates)
    else:
        candidates = None

    if labels_path is None:
        raise ValueError("Provide --labels or --handoff-dir")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = load_records(labels_path)

    if registry_path is not None and registry_path.exists():
        reg = read_json(registry_path)
        mapping = reg.get("internal_to_official")

    if args.predictions is not None:
        predictions = load_records(args.predictions)
        report = evaluate_chunk_predictions(
            predictions,
            labels,
            internal_to_official=mapping,
            zero_division=args.zero_division,
        )
    else:
        # Default top-k to 1 if neither top_k nor threshold is set
        top_k = args.top_k
        if top_k is None and all(value is None for value in (args.threshold, args.fallback, args.max_chunks)):
            top_k = 1

        report = evaluate_candidate_selection(
            candidates,
            labels,
            top_k=top_k,
            threshold=args.threshold,
            fallback=args.fallback,
            max_chunks=args.max_chunks,
            internal_to_official=mapping,
            zero_division=args.zero_division,
        )

    if not args.quiet:
        print("\n" + format_metrics_table(report) + "\n")

    macro = report["macro"]
    logging.info(
        "Macro Metrics (N=%d queries) -> Precision: %.4f, Recall: %.4f, F1: %.4f, F2: %.4f",
        report["query_count"],
        macro["precision"],
        macro["recall"],
        macro["f1"],
        macro["f2"],
    )

    if args.output_csv:
        export_metrics_csv(report, args.output_csv)
        logging.info("Saved CSV report to %s", args.output_csv)

    if args.output_json:
        export_metrics_json(report, args.output_json)
        logging.info("Saved JSON report to %s", args.output_json)

    return report


def main():
    args = parse_args()
    run_evaluation_cli(args)


if __name__ == "__main__":
    main()
