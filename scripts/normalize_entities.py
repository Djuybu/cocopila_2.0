"""CLI entrypoint for P3-10: normalize extracted entities to canonical entities.

Keeps the original surface form and adds canonical ID/name/type so duplicate
aliases collapse to a single graph node.
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
from src.graph.ner import load_lexicon
from src.graph.normalize import normalize_entities, summarize_normalization, write_normalized_entities
from src.utils.io import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("normalize_entities")
DEFAULT_LEXICON = REPO_ROOT / "configs" / "graph" / "lexicon_seed.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-10: normalize entities to canonical entities")
    parser.add_argument("--entities", type=Path, required=True, help="entities.jsonl from P3-09")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON, help="Lexicon YAML (default: seed lexicon)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for normalized_entities.jsonl")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for label, path in (("entities", args.entities), ("lexicon", args.lexicon)):
        if not Path(path).exists():
            raise SystemExit(f"{label} file not found: {path}")
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")

    lexicon = load_lexicon(args.lexicon)
    records = load_records(args.entities)
    normalized = normalize_entities(records, lexicon)
    report = summarize_normalization(records, normalized)

    output.mkdir(parents=True)
    write_normalized_entities(output / "normalized_entities.jsonl", normalized)
    write_json(output / "normalization_report.json", report)

    print("\n" + "=" * 72)
    print("P3-10: ENTITY NORMALIZATION")
    print("=" * 72)
    print(f"Mentions                 : {report['mention_count']}")
    print(f"Unique surfaces          : {report['unique_surfaces']}")
    print(f"Unique canonical entities: {report['unique_canonical_entities']}")
    print(f"Output directory         : {output}")
    print("=" * 72 + "\n")
    return report


if __name__ == "__main__":
    main()
