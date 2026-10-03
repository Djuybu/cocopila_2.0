"""CLI entrypoint for P3-01: build and validate the chunk -> parent-document mapping.

The P1 corpus is authoritative; the P2 handoff registry must agree or the run
fails. P1/P2 modules are imported read-only.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path (matches existing P2 script entrypoints).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p3.hierarchy import run_hierarchy
from src.utils.config import load_config
from src.utils.io import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_chunk_doc_mapping")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P3-01: build/validate the stable chunk -> doc mapping against the P2 registry"
    )
    parser.add_argument("--config", type=Path, help="Optional YAML with a hierarchy: section")
    parser.add_argument("--corpus-dir", type=Path, help="Directory holding documents.json and chunks.json")
    parser.add_argument("--documents", type=Path, help="Explicit documents file (overrides --corpus-dir)")
    parser.add_argument("--chunks", type=Path, help="Explicit chunks file (overrides --corpus-dir)")
    parser.add_argument("--handoff-dir", type=Path, help="P2 handoff directory containing registry.json")
    parser.add_argument("--output-dir", type=Path, help="New directory for the hierarchy artifacts")
    parser.add_argument("--allow-registry-superset", action="store_true",
                        help="Allow registry chunks absent from the corpus (subset runs)")
    parser.add_argument("--skip-registry-check", action="store_true",
                        help="Build from the corpus only (no registry reconciliation)")
    return parser.parse_args(argv)


def resolve_paths(args):
    cfg = load_config(args.config).get("hierarchy", {}) if args.config else {}

    def pick(flag, key):
        return flag if flag is not None else (Path(cfg[key]) if cfg.get(key) else None)

    corpus_dir = pick(args.corpus_dir, "corpus_dir")
    documents = pick(args.documents, "documents_path")
    chunks = pick(args.chunks, "chunks_path")
    if corpus_dir is not None:
        documents = documents or corpus_dir / "documents.json"
        chunks = chunks or corpus_dir / "chunks.json"
    return documents, chunks, pick(args.handoff_dir, "handoff_dir"), pick(args.output_dir, "output_dir")


def main(argv=None):
    args = parse_args(argv)
    documents, chunks, handoff_dir, output_dir = resolve_paths(args)
    if documents is None or chunks is None:
        raise SystemExit("Provide --corpus-dir or explicit --documents/--chunks")
    if output_dir is None:
        raise SystemExit("--output-dir is required")
    if handoff_dir is None and not args.skip_registry_check:
        raise SystemExit("--handoff-dir is required (or pass --skip-registry-check)")
    for label, path in (("documents", documents), ("chunks", chunks)):
        if not Path(path).exists():
            raise SystemExit(f"{label} file not found: {path}")
    if handoff_dir is not None and not (Path(handoff_dir) / "registry.json").exists():
        raise SystemExit(f"registry.json not found in {handoff_dir}")

    logger.info("Building chunk -> doc mapping from corpus %s", documents)
    if handoff_dir is not None:
        logger.info("Reconciling against P2 registry in %s", handoff_dir)
    result = run_hierarchy(documents, chunks, output_dir, handoff_dir,
                           allow_registry_superset=args.allow_registry_superset)
    report = read_json(result["report"])
    print("\n" + "=" * 72)
    print("P3-01: CHUNK -> DOC HIERARCHY")
    print("=" * 72)
    print(f"Documents            : {report['document_count']}")
    print(f"Official chunks      : {report['official_chunk_count']}")
    print(f"Internal rechunks    : {report['internal_rechunk_count']}")
    reconciliation = report.get("registry_reconciliation")
    print(f"Registry reconciled  : {reconciliation['ok'] if reconciliation else 'not checked'}")
    print(f"Output directory     : {output_dir}")
    print("=" * 72 + "\n")
    return result


if __name__ == "__main__":
    main()
