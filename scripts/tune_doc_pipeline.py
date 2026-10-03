"""CLI entrypoint for P3-14: tune the official document aggregation pipeline.

Runs K-fold query holdout tuning with an aggregation x direct-doc ablation and
writes ``best_doc_pipeline.yaml`` plus the ablation CSV.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path (matches existing P2 script entrypoints).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_records
from src.p3.doc_pipeline import DEFAULT_DIRECT_WEIGHTS, tune_doc_pipeline, write_doc_pipeline
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_doc_pipeline")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P3-14: K-fold holdout tuning of the official doc aggregation pipeline")
    parser.add_argument("--run-dir", "--handoff-dir", dest="run_dir", type=Path,
                        help="P2 handoff/run dir with reranked.jsonl, labels.json, registry.json")
    parser.add_argument("--scored-file", type=Path, help="Explicit reranked.jsonl/candidates.jsonl")
    parser.add_argument("--labels", type=Path, help="Explicit labels.json")
    parser.add_argument("--registry", type=Path, help="Explicit registry.json")
    parser.add_argument("--direct-doc", type=Path,
                        help="Optional direct doc-retriever rankings (JSON/JSONL with id + candidates/docs)")
    parser.add_argument("--folds", type=int, default=5, help="Number of query folds (default 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direct-weights", type=float, nargs="+", default=list(DEFAULT_DIRECT_WEIGHTS),
                        help="Direct-doc fusion weights to ablate (default: 0.0 0.5)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for P3-14 artifacts")
    return parser.parse_args(argv)


def resolve_inputs(args):
    run_dir = args.run_dir
    scored = args.scored_file
    if scored is None and run_dir is not None:
        for name in ("reranked.jsonl", "candidates.jsonl"):
            if (run_dir / name).exists():
                scored = run_dir / name
                break
    labels = args.labels or (run_dir / "labels.json" if run_dir else None)
    registry = args.registry or (run_dir / "registry.json" if run_dir else None)
    return scored, labels, registry


def main(argv=None):
    args = parse_args(argv)
    scored_path, labels_path, registry_path = resolve_inputs(args)
    if scored_path is None or not Path(scored_path).exists():
        raise SystemExit("Provide --run-dir/--handoff-dir or --scored-file with reranked.jsonl")
    if labels_path is None or not Path(labels_path).exists():
        raise SystemExit(f"Labels not found: {labels_path}")
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")

    scored = load_records(scored_path)
    labels = read_json(labels_path)
    registry = read_json(registry_path) if registry_path and Path(registry_path).exists() else {}
    mapping, chunk_to_doc = registry.get("internal_to_official"), registry.get("chunk_to_doc")
    direct_doc = load_records(args.direct_doc) if args.direct_doc else None

    logger.info("P3-14: %d-fold holdout tuning over %d queries (direct_doc=%s)",
                args.folds, len(labels), bool(direct_doc))
    best, ablation = tune_doc_pipeline(
        scored, labels, direct_doc_records=direct_doc, direct_weights=tuple(args.direct_weights),
        n_folds=args.folds, seed=args.seed,
        internal_to_official=mapping, chunk_to_doc=chunk_to_doc)

    output.mkdir(parents=True)
    ablation.to_csv(output / "doc_pipeline_ablation.csv", index=False)
    write_doc_pipeline(output / "best_doc_pipeline.yaml", best,
                       ablation_csv=output / "doc_pipeline_ablation.csv")
    report = {"best_doc_pipeline": best, "query_count": len(labels), "folds": args.folds,
              "ablation_rows": int(len(ablation)), "direct_doc": bool(direct_doc)}
    write_json(output / "doc_pipeline_report.json", report)

    print("\n" + "=" * 72)
    print("P3-14: OFFICIAL DOC AGGREGATION PIPELINE")
    print("=" * 72)
    print(f"Method / direct weight : {best['ablation_method']} / {best['direct_doc_weight']}")
    print(f"Holdout Macro F2_doc   : {best['holdout']['mean_f2_doc']:.4f} "
          f"± {best['holdout']['std_f2_doc']:.4f} ({best['holdout']['n_folds']} folds)")
    print(f"Full-data F2_doc       : {best['full_data_f2_doc']:.4f}")
    print(f"Output                 : {output}")
    print("=" * 72 + "\n")
    return report


if __name__ == "__main__":
    main()
