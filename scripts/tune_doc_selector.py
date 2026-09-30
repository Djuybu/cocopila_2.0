"""CLI entrypoint for P3-02/03/04: doc aggregation ablation + independent doc-selector tuning.

Reads the P2 handoff (reranked.jsonl + labels.json + registry.json), compares
aggregation methods, then tunes doc_threshold/doc_fallback/doc_max on Macro
F2_doc without touching the P2 chunk selector.
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
from src.p3.doc_aggregation import aggregation_spec, choose_baseline, evaluate_aggregation
from src.p3.doc_selector import sweep_doc_selector, write_best_doc_selector
from src.utils.config import load_config
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_doc_selector")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P3-02/03/04: document aggregation ablation and independent doc-selector tuning"
    )
    parser.add_argument("--config", type=Path, help="Optional YAML with a tune_doc_selector: section")
    parser.add_argument("--run-dir", "--handoff-dir", dest="run_dir", type=Path,
                        help="P2 handoff/run directory with reranked.jsonl, labels.json, registry.json")
    parser.add_argument("--scored-file", type=Path, help="Explicit reranked.jsonl/candidates.jsonl")
    parser.add_argument("--labels", type=Path, help="Explicit labels.json (defaults to --run-dir/labels.json)")
    parser.add_argument("--registry", type=Path, help="Explicit registry.json (defaults to --run-dir/registry.json)")
    parser.add_argument("--output-dir", type=Path, help="New directory for P3-03/04 artifacts")
    parser.add_argument("--steps", type=int, default=20, help="Threshold grid steps over the doc score range")
    parser.add_argument("--min-th", type=float, help="Lower threshold bound")
    parser.add_argument("--max-th", type=float, help="Upper threshold bound")
    parser.add_argument("--fallbacks", type=int, nargs="+", help="doc_fallback grid")
    parser.add_argument("--maximums", type=int, nargs="+", help="doc_max grid")
    parser.add_argument("--ablation-doc-threshold", type=float, help="Fixed threshold used by the ablation")
    parser.add_argument("--ablation-doc-fallback", type=int, help="Fixed fallback used by the ablation")
    parser.add_argument("--ablation-doc-max", type=int, help="Fixed doc_max used by the ablation")
    return parser.parse_args(argv)


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def resolve_paths(args, cfg):
    def pick(attr, key):
        return getattr(args, attr) if getattr(args, attr) is not None else (Path(cfg[key]) if cfg.get(key) else None)

    run_dir = pick("run_dir", "source_run_dir")
    scored = pick("scored_file", "scored_path")
    if scored is None and run_dir is not None:
        for name in ("reranked.jsonl", "candidates.jsonl"):
            if (run_dir / name).exists():
                scored = run_dir / name
                break
    labels = pick("labels", "labels_path") or (run_dir / "labels.json" if run_dir else None)
    registry = pick("registry", "registry_path") or (run_dir / "registry.json" if run_dir else None)
    return scored, labels, registry, pick("output_dir", "output_dir")


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config).get("tune_doc_selector", {}) if args.config else {}
    scored_path, labels_path, registry_path, output_dir = resolve_paths(args, cfg)
    if scored_path is None:
        raise SystemExit("Provide --run-dir/--handoff-dir or --scored-file")
    if labels_path is None or not Path(labels_path).exists():
        raise SystemExit(f"Labels not found: {labels_path}")
    if output_dir is None:
        raise SystemExit("--output-dir is required")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    scored = load_records(scored_path)
    labels = read_json(labels_path)
    registry = read_json(registry_path) if registry_path and Path(registry_path).exists() else {}
    mapping, chunk_to_doc = registry.get("internal_to_official"), registry.get("chunk_to_doc")

    ablation_cfg = cfg.get("ablation", {})
    doc_threshold = _first(args.ablation_doc_threshold, ablation_cfg.get("doc_threshold"))
    doc_fallback = _first(args.ablation_doc_fallback, ablation_cfg.get("doc_fallback"), 0)
    doc_max = _first(args.ablation_doc_max, ablation_cfg.get("doc_max"), 10)
    sweep_cfg = cfg.get("sweep", {})
    fallbacks = args.fallbacks or sweep_cfg.get("fallbacks")
    maximums = args.maximums or sweep_cfg.get("maximums")

    logger.info("P3-03: aggregation ablation over %d queries", len(labels))
    ablation = evaluate_aggregation(scored, labels, mapping, chunk_to_doc,
                                    doc_threshold=doc_threshold, doc_fallback=doc_fallback, doc_max=doc_max)
    baseline = choose_baseline(ablation)
    spec = aggregation_spec(baseline["method"])
    logger.info("Baseline aggregation=%s macro F2_doc=%.4f", baseline["method"], baseline["macro_f2"])

    logger.info("P3-04: tuning doc selector independently of the chunk selector")
    best, sweep = sweep_doc_selector(
        scored, labels, aggregation=spec["method"], k=spec["k"], weight=spec["weight"],
        fallbacks=fallbacks, maximums=maximums, steps=args.steps,
        min_th=args.min_th, max_th=args.max_th,
        internal_to_official=mapping, chunk_to_doc=chunk_to_doc)

    output_dir.mkdir(parents=True)
    ablation.to_csv(output_dir / "doc_aggregation_ablation.csv", index=False)
    sweep.to_csv(output_dir / "doc_selector_sweep.csv", index=False)
    write_best_doc_selector(output_dir / "best_doc_selector.yaml", best,
                            baseline=baseline["method"], ablation_csv=output_dir / "doc_aggregation_ablation.csv")
    report = {"baseline_aggregation": baseline, "best_doc_selector": best, "query_count": len(labels),
              "sweep_rows": int(len(sweep)), "chunk_selector_untouched": True}
    write_json(output_dir / "doc_selector_report.json", report)

    print("\n" + "=" * 72)
    print("P3-03/04: DOC AGGREGATION + SELECTOR")
    print("=" * 72)
    print(f"Queries                 : {len(labels)}")
    print(f"Baseline aggregation    : {baseline['method']} (macro F2_doc={baseline['macro_f2']:.4f})")
    print(f"Best doc selector       : threshold={best['doc_threshold']}, fallback={best['doc_fallback']}, "
          f"max={best['doc_max']}")
    print(f"Best macro F2_doc       : {best['macro_f2']:.4f}")
    print(f"Output directory        : {output_dir}")
    print("=" * 72 + "\n")
    return report


if __name__ == "__main__":
    main()
