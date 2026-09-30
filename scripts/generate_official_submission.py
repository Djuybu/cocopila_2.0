"""CLI entrypoint for P3-15: generate the official submission JSON + ZIP.

Validates IDs against the registry, guarantees exactly one JSON inside the ZIP,
and records the config + git commit used for the run.
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
from src.p3.doc_pipeline import load_doc_pipeline_config
from src.p3.official_submission import generate_official_submission
from src.p3.predictions import load_chunk_selector_config
from src.p3.submission import validator_from_registry
from src.utils.io import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_official_submission")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P3-15: generate and pack the official submission JSON/ZIP")
    parser.add_argument("--run-dir", "--handoff-dir", dest="run_dir", type=Path,
                        help="P2 handoff/run dir with reranked.jsonl and registry.json")
    parser.add_argument("--scored-file", type=Path, help="Explicit reranked.jsonl/candidates.jsonl")
    parser.add_argument("--chunk-selector", type=Path,
                        help="P2 best_chunk_selector.yaml (defaults to --run-dir/best_chunk_selector.yaml)")
    parser.add_argument("--doc-pipeline", type=Path, required=True, help="P3-14 best_doc_pipeline.yaml")
    parser.add_argument("--registry", type=Path, help="registry.json (required for validation)")
    parser.add_argument("--run-id", required=True, help="Run ID used in submission_<run_id>.zip")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for JSON + manifest")
    parser.add_argument("--submission-dir", type=Path, help="ZIP destination (default: --output-dir)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir
    scored = args.scored_file
    if scored is None and run_dir is not None:
        for name in ("reranked.jsonl", "candidates.jsonl"):
            if (run_dir / name).exists():
                scored = run_dir / name
                break
    if scored is None or not Path(scored).exists():
        raise SystemExit("Provide --run-dir/--handoff-dir or --scored-file with reranked.jsonl")
    chunk_selector = args.chunk_selector or (run_dir / "best_chunk_selector.yaml" if run_dir else None)
    if chunk_selector is None or not Path(chunk_selector).exists():
        raise SystemExit("Provide --chunk-selector (or a run dir containing best_chunk_selector.yaml)")
    registry = args.registry or (run_dir / "registry.json" if run_dir else None)
    if registry is None or not Path(registry).exists():
        raise SystemExit("Provide --registry (or a run dir containing registry.json)")
    if not Path(args.doc_pipeline).exists():
        raise SystemExit(f"Doc pipeline not found: {args.doc_pipeline}")
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")

    validator = validator_from_registry(read_json(registry))
    logger.info("P3-15: generating official submission run_id=%s", args.run_id)
    result = generate_official_submission(
        load_records(scored), load_chunk_selector_config(chunk_selector),
        load_doc_pipeline_config(args.doc_pipeline), validator,
        run_id=args.run_id, output_dir=output, submission_dir=args.submission_dir)

    print("\n" + "=" * 72)
    print("P3-15: OFFICIAL SUBMISSION")
    print("=" * 72)
    print(f"Run ID        : {args.run_id}")
    print(f"Queries       : {result['manifest_payload']['query_count']}")
    print(f"JSON          : {result['json']}")
    print(f"ZIP           : {result['zip']}")
    print(f"Git commit    : {result['manifest_payload']['git_commit']}")
    print("=" * 72 + "\n")
    return result


if __name__ == "__main__":
    main()
