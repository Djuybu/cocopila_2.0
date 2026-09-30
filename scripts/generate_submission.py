"""CLI entrypoint for P3-06: generate the final submission JSON.

Builds predictions from the independent P2 chunk selector and P3-04 doc selector
(or normalizes an existing predictions file), sorts/dedups IDs, validates against
the official registry and writes exactly one JSON file.
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
from src.p3.doc_selector import load_doc_selector_config
from src.p3.predictions import build_predictions, load_chunk_selector_config
from src.p3.submission import validator_from_registry, write_submission
from src.utils.io import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_submission")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P3-06: generate a validated, sorted/deduped submission JSON"
    )
    parser.add_argument("--run-dir", "--handoff-dir", dest="run_dir", type=Path,
                        help="P2 handoff/run directory with reranked.jsonl and registry.json")
    parser.add_argument("--scored-file", type=Path, help="Explicit reranked.jsonl/candidates.jsonl")
    parser.add_argument("--chunk-selector", type=Path,
                        help="P2 best_chunk_selector.yaml (defaults to --run-dir/best_chunk_selector.yaml)")
    parser.add_argument("--doc-selector", type=Path, help="P3-04 best_doc_selector.yaml")
    parser.add_argument("--registry", type=Path, help="registry.json (enables full validation)")
    parser.add_argument("--predictions", type=Path,
                        help="Existing predictions JSON to normalize instead of building from selectors")
    parser.add_argument("--output", type=Path, required=True, help="Destination submission JSON (must not exist)")
    return parser.parse_args(argv)


def resolve_inputs(args):
    run_dir = args.run_dir
    registry = args.registry or (run_dir / "registry.json" if run_dir else None)
    scored = args.scored_file
    if scored is None and run_dir is not None:
        for name in ("reranked.jsonl", "candidates.jsonl"):
            if (run_dir / name).exists():
                scored = run_dir / name
                break
    chunk_selector = args.chunk_selector or (run_dir / "best_chunk_selector.yaml" if run_dir else None)
    return scored, chunk_selector, registry


def main(argv=None):
    args = parse_args(argv)
    scored, chunk_selector, registry = resolve_inputs(args)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    registry_payload = read_json(registry) if registry and Path(registry).exists() else None
    validator = validator_from_registry(registry_payload) if registry_payload else None

    if args.predictions is not None:
        logger.info("Normalizing existing predictions from %s", args.predictions)
        predictions = read_json(args.predictions)
    else:
        if scored is None or not Path(scored).exists():
            raise SystemExit("Provide --predictions or a run dir/scored file with reranked.jsonl")
        if args.doc_selector is None or not Path(args.doc_selector).exists():
            raise SystemExit("--doc-selector (P3-04 best_doc_selector.yaml) is required")
        if chunk_selector is None or not Path(chunk_selector).exists():
            raise SystemExit("Provide --chunk-selector (or a run dir containing best_chunk_selector.yaml)")
        mapping = registry_payload.get("internal_to_official") if registry_payload else None
        chunk_to_doc = registry_payload.get("chunk_to_doc") if registry_payload else None
        predictions = build_predictions(
            load_records(scored),
            load_chunk_selector_config(chunk_selector),
            load_doc_selector_config(args.doc_selector),
            internal_to_official=mapping, chunk_to_doc=chunk_to_doc)

    normalized = write_submission(args.output, predictions, validator)
    total_docs = sum(len(row["relevant_docs"]) for row in normalized)
    total_chunks = sum(len(row["relevant_chunks"]) for row in normalized)

    print("\n" + "=" * 72)
    print("P3-06: SUBMISSION GENERATED")
    print("=" * 72)
    print(f"Queries        : {len(normalized)}")
    print(f"Docs / chunks  : {total_docs} / {total_chunks}")
    print(f"Validated      : {validator is not None}")
    print(f"Output         : {args.output}")
    print("=" * 72 + "\n")
    return normalized


if __name__ == "__main__":
    main()
