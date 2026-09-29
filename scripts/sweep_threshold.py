"""CLI entrypoint for P2-03: Sweep threshold reranker & plateau analysis.

Evaluates candidates over a grid of score thresholds to optimize Macro Chunk F2,
records Precision, Recall, F1, F2, and chunk volume per query, and identifies
the threshold plateau.
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
from src.scoring.sweep import sweep_reranker_thresholds
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep_threshold")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-03: Sweep threshold reranker to maximize Macro Chunk F2 and detect plateau"
    )

    # Input source options
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

    # Labels and registry
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

    # Threshold grid controls
    parser.add_argument(
        "--min-th",
        type=float,
        default=None,
        help="Minimum threshold to evaluate (default: min score in candidates)",
    )
    parser.add_argument(
        "--max-th",
        type=float,
        default=None,
        help="Maximum threshold to evaluate (default: max score in candidates)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=None,
        help="Step size for threshold sweep (e.g. 0.02)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps if --step is not specified (default: 50)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance below peak F2 to define plateau (default: 0.01 = 1 percent)",
    )

    # Fallback and max limits
    parser.add_argument(
        "--fallback",
        type=int,
        default=0,
        help="Minimum chunks to return per query (default: 0)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to return per query (default: unlimited)",
    )

    # Outputs
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to save threshold_sweep.csv (default: <run-dir>/threshold_sweep.csv)",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Path to save JSON summary of best configuration and plateau metadata",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Path to save optimization curves PNG plot",
    )

    return parser.parse_args()


def plot_sweep_curves(sweep_df, plateau_info, output_path):
    """Generate and save PNG curve visualizing F2, Precision, Recall and the plateau region."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot generation.")
        return

    df_sorted = sweep_df.sort_values("threshold").dropna(subset=["threshold"])
    if df_sorted.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Macro F2, Precision, Recall vs Threshold
    ax1.plot(df_sorted["threshold"], df_sorted["macro_f2"], "b-", linewidth=2, label="Macro F2 (Primary)")
    ax1.plot(df_sorted["threshold"], df_sorted["macro_recall"], "g--", linewidth=1.5, label="Macro Recall")
    ax1.plot(df_sorted["threshold"], df_sorted["macro_precision"], "r:", linewidth=1.5, label="Macro Precision")

    p_min = plateau_info.get("plateau_min_threshold")
    p_max = plateau_info.get("plateau_max_threshold")
    p_rec = plateau_info.get("recommended_threshold")

    if p_min is not None and p_max is not None:
        ax1.axvspan(p_min, p_max, color="yellow", alpha=0.25, label=f"Plateau [{p_min:.3f}, {p_max:.3f}]")
    if p_rec is not None:
        ax1.axvline(p_rec, color="blue", linestyle="-.", linewidth=2, label=f"Recommended ({p_rec:.3f})")

    ax1.set_title("P2-03: Macro Metrics vs Cutoff Threshold", fontsize=12)
    ax1.set_xlabel("Cutoff Threshold", fontsize=10)
    ax1.set_ylabel("Metric Score", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best")

    # Plot 2: Average Chunks & Zero-Chunk Query Count vs Threshold
    ax2_twin = ax2.twinx()
    ax2.plot(df_sorted["threshold"], df_sorted["avg_chunks_per_query"], "m-", linewidth=2, label="Avg Chunks / Query")
    ax2_twin.plot(df_sorted["threshold"], df_sorted["zero_chunk_queries"], "k:", linewidth=1.8, label="Zero-Chunk Queries")

    if p_rec is not None:
        ax2.axvline(p_rec, color="blue", linestyle="-.", linewidth=1.5)

    ax2.set_title("Retrieval Volume & Zero-Output Queries vs Threshold", fontsize=12)
    ax2.set_xlabel("Cutoff Threshold", fontsize=10)
    ax2.set_ylabel("Avg Chunks / Query", color="m", fontsize=10)
    ax2_twin.set_ylabel("Zero-Chunk Queries Count", color="k", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved sweep visualization curve to %s", output_path)


def main():
    args = parse_args()

    # Determine input paths
    if args.run_dir:
        run_dir = args.run_dir.resolve()
        reranked_candidate = run_dir / "reranked.jsonl"
        candidates_candidate = run_dir / "candidates.jsonl"
        if reranked_candidate.exists() and reranked_candidate.stat().st_size > 0:
            scored_path = reranked_candidate
        elif candidates_candidate.exists() and candidates_candidate.stat().st_size > 0:
            scored_path = candidates_candidate
        else:
            raise FileNotFoundError(f"Neither reranked.jsonl nor candidates.jsonl found in {run_dir}")

        labels_path = args.labels if args.labels else run_dir / "labels.json"
        registry_path = args.registry if args.registry else run_dir / "registry.json"
        output_csv = args.output_csv if args.output_csv else run_dir / "threshold_sweep.csv"
    else:
        scored_path = args.scored_file.resolve()
        if not args.labels:
            raise ValueError("--labels is required when --scored-file is provided")
        labels_path = args.labels.resolve()
        registry_path = args.registry.resolve() if args.registry else None
        output_csv = args.output_csv if args.output_csv else scored_path.parent / "threshold_sweep.csv"

    if not scored_path.exists():
        raise FileNotFoundError(f"Scored records file not found: {scored_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    logger.info("Loading scored records from %s...", scored_path)
    scored_records = load_records(scored_path)
    logger.info("Loading labels from %s...", labels_path)
    labels = read_json(labels_path)

    registry = read_json(registry_path) if registry_path and registry_path.exists() else {}
    mapping = registry.get("internal_to_official", {})
    chunk_to_doc = registry.get("chunk_to_doc")

    # Build custom threshold list if step is provided
    thresholds = None
    if args.step is not None:
        if args.min_th is not None and args.max_th is not None:
            cur = args.min_th
            thresholds = []
            while cur <= args.max_th + 1e-9:
                thresholds.append(round(cur, 5))
                cur += args.step

    logger.info(
        "Executing P2-03 threshold sweep (fallback=%d, max=%s, tolerance=%.3f)...",
        args.fallback,
        args.max_chunks,
        args.tolerance,
    )

    sweep_df, plateau_info, recommended_config = sweep_reranker_thresholds(
        scored_records=scored_records,
        labels=labels,
        thresholds=thresholds,
        steps=args.steps,
        min_th=args.min_th,
        max_th=args.max_th,
        fallback=args.fallback,
        max_chunks=args.max_chunks,
        internal_to_official=mapping,
        chunk_to_doc=chunk_to_doc,
        tolerance=args.tolerance,
    )

    # Save deliverable threshold_sweep.csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_csv, index=False)
    logger.info("Successfully exported deliverable to %s", output_csv)

    # Save summary JSON if requested
    if args.output_summary:
        summary_payload = {
            "deliverable": str(output_csv.name),
            "plateau_info": plateau_info,
            "recommended_config": recommended_config,
            "evaluated_thresholds_count": len(sweep_df),
        }
        write_json(args.output_summary, summary_payload)
        logger.info("Saved summary metadata to %s", args.output_summary)

    # Plot if requested
    if args.output_plot:
        plot_sweep_curves(sweep_df, plateau_info, args.output_plot)

    # Print Formatted Report to stdout
    print("\n" + "=" * 78)
    print("        P2-03: THRESHOLD SWEEP & PLATEAU IDENTIFICATION REPORT")
    print("=" * 78)
    print(f"  Deliverable File       : {output_csv.resolve()}")
    print(f"  Evaluated Thresholds   : {len(sweep_df)}")
    print(f"  Best Validation F2     : {plateau_info['best_f2']:.4f}")
    print(f"  Plateau Region [1%]    : [{plateau_info['plateau_min_threshold']}, {plateau_info['plateau_max_threshold']}]")
    print(f"  Plateau Width          : {plateau_info['plateau_width']}")
    print(f"  Plateau Config Count   : {plateau_info['plateau_count']}")
    print(f"  Plateau Mean F2        : {plateau_info['plateau_mean_f2']:.4f}")
    print(f"  Recommended Threshold  : {recommended_config['threshold']} (Robust Center of Plateau)")
    print(f"  Metrics at Rec. Cutoff : F2={recommended_config['macro_f2']:.4f} | R={recommended_config['macro_recall']:.4f} | P={recommended_config['macro_precision']:.4f}")
    print(f"  Avg Chunks per Query   : {recommended_config['avg_chunks_per_query']:.1f}")
    print(f"  Zero-Chunk Queries     : {recommended_config['zero_chunk_queries']} (Motivates P2-04 Fallback)")
    print("=" * 78)

    print("\nTop 5 Evaluated Thresholds (Ranked by Macro F2):")
    cols_to_show = ["threshold", "macro_f2", "macro_recall", "macro_precision", "avg_chunks_per_query", "zero_chunk_queries"]
    print(sweep_df.head(5)[cols_to_show].to_string(index=False))
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
