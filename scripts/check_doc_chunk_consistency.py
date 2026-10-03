"""CLI entrypoint for P3-05: quantify (and optionally reconcile) doc/chunk consistency.

Joins the P2 chunk branch (``best_chunk_selector.yaml``) with the P3-04 document
branch (``best_doc_selector.yaml``) and reports every inconsistency. Default
behaviour is report-only because the ground-truth hierarchy rule is not known yet.
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
from src.p3.consistency import POLICIES, analyze_consistency, apply_policy, summarize_consistency
from src.p3.doc_selector import load_doc_selector_config
from src.p3.predictions import build_predictions, load_chunk_selector_config
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_doc_chunk_consistency")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P3-05: quantify doc/chunk selection consistency and optionally apply a policy"
    )
    parser.add_argument("--run-dir", "--handoff-dir", dest="run_dir", type=Path,
                        help="P2 handoff/run directory with reranked.jsonl and registry.json")
    parser.add_argument("--scored-file", type=Path, help="Explicit reranked.jsonl/candidates.jsonl")
    parser.add_argument("--chunk-selector", type=Path,
                        help="P2 best_chunk_selector.yaml (defaults to --run-dir/best_chunk_selector.yaml)")
    parser.add_argument("--doc-selector", type=Path, required=True, help="P3-04 best_doc_selector.yaml")
    parser.add_argument("--registry", type=Path, help="Explicit registry.json")
    parser.add_argument("--chunk-to-doc", type=Path,
                        help="Explicit chunk_to_doc.json (defaults to registry chunk_to_doc)")
    parser.add_argument("--policy", choices=POLICIES, default="none",
                        help="none=report only (default); others rewrite predictions")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for P3-05 artifacts")
    return parser.parse_args(argv)


def resolve_inputs(args):
    run_dir = args.run_dir
    scored = args.scored_file
    if scored is None and run_dir is not None:
        for name in ("reranked.jsonl", "candidates.jsonl"):
            if (run_dir / name).exists():
                scored = run_dir / name
                break
    if scored is None:
        raise SystemExit("Provide --run-dir/--handoff-dir or --scored-file")
    chunk_selector = args.chunk_selector or (run_dir / "best_chunk_selector.yaml" if run_dir else None)
    if chunk_selector is None or not Path(chunk_selector).exists():
        raise SystemExit("Provide --chunk-selector (or a run dir containing best_chunk_selector.yaml)")
    registry = args.registry or (run_dir / "registry.json" if run_dir else None)
    chunk_to_doc = read_json(args.chunk_to_doc) if args.chunk_to_doc else None
    mapping = None
    if registry is not None and Path(registry).exists():
        payload = read_json(registry)
        mapping = payload.get("internal_to_official")
        chunk_to_doc = chunk_to_doc or payload.get("chunk_to_doc")
    if not chunk_to_doc:
        raise SystemExit("Provide --chunk-to-doc or a registry with a chunk_to_doc mapping")
    return scored, chunk_selector, mapping, chunk_to_doc


def main(argv=None):
    args = parse_args(argv)
    scored_path, chunk_selector, mapping, chunk_to_doc = resolve_inputs(args)
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")

    scored = load_records(scored_path)
    chunk_config = load_chunk_selector_config(chunk_selector)
    doc_config = load_doc_selector_config(args.doc_selector)
    predictions = build_predictions(scored, chunk_config, doc_config,
                                    internal_to_official=mapping, chunk_to_doc=chunk_to_doc)
    report_df = analyze_consistency(predictions, chunk_to_doc)
    summary = summarize_consistency(report_df)
    summary["policy"] = args.policy
    summary["predictions_adjusted"] = args.policy != "none"

    output.mkdir(parents=True)
    report_df.to_csv(output / "consistency_report.csv", index=False)
    if args.policy != "none":
        adjusted = apply_policy(predictions, chunk_to_doc, args.policy)
        write_json(output / "predictions_consistent.json", adjusted)
    write_json(output / "consistency_summary.json", summary)

    print("\n" + "=" * 72)
    print("P3-05: DOC/CHUNK CONSISTENCY")
    print("=" * 72)
    print(f"Queries                        : {summary['query_count']}")
    print(f"Consistent / inconsistent      : {summary['queries_consistent']} / {summary['queries_inconsistent']}")
    print(f"Chunks without selected doc    : {summary['total_chunks_without_doc']}")
    print(f"Docs without selected chunk    : {summary['total_docs_without_chunk']}")
    print(f"Unknown chunks                 : {summary['total_unknown_chunks']}")
    print(f"Policy                         : {args.policy}")
    print(f"Output directory               : {output}")
    print("=" * 72 + "\n")
    return summary


if __name__ == "__main__":
    main()
