"""CLI entrypoint for P3-13: audit the ground-truth chunk/document hierarchy relation."""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path (matches existing P2 script entrypoints).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p3.label_audit import write_label_audit
from src.utils.io import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_label_hierarchy")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="P3-13: measure the chunk/document relation in the ground-truth labels"
    )
    parser.add_argument("--labels", type=Path, required=True, help="labels.json (official or prototype)")
    parser.add_argument("--handoff-dir", "--run-dir", dest="run_dir", type=Path,
                        help="Directory with registry.json (used for chunk_to_doc)")
    parser.add_argument("--registry", type=Path, help="Explicit registry.json")
    parser.add_argument("--chunk-to-doc", type=Path, help="Explicit chunk_to_doc.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for P3-13 artifacts")
    return parser.parse_args(argv)


def resolve_chunk_to_doc(args):
    if args.chunk_to_doc is not None:
        return read_json(args.chunk_to_doc)
    registry_path = args.registry or (args.run_dir / "registry.json" if args.run_dir else None)
    if registry_path is None or not Path(registry_path).exists():
        raise SystemExit("Provide --chunk-to-doc, --registry or --handoff-dir with registry.json")
    registry = read_json(registry_path)
    if not isinstance(registry.get("chunk_to_doc"), dict) or not registry["chunk_to_doc"]:
        raise SystemExit(f"registry.json has no chunk_to_doc mapping: {registry_path}")
    return registry["chunk_to_doc"]


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.labels).exists():
        raise SystemExit(f"Labels file not found: {args.labels}")
    chunk_to_doc = resolve_chunk_to_doc(args)
    labels = read_json(args.labels)
    logger.info("Auditing %d queries against %d official chunks", len(labels), len(chunk_to_doc))
    report = write_label_audit(args.output_dir, labels, chunk_to_doc)

    print("\n" + "=" * 72)
    print("P3-13: LABEL HIERARCHY AUDIT")
    print("=" * 72)
    print(f"Queries                          : {report['query_count']}")
    print(f"chunk => parent doc violations   : {report['chunk_implies_doc_violations']}")
    print(f"Relevant docs without labeled chunk: {report['docs_without_labeled_chunk']}")
    print(f"Recommended policy               : {report['recommended_policy']}")
    print(f"Output directory                 : {args.output_dir}")
    print("=" * 72 + "\n")
    return report


if __name__ == "__main__":
    main()
