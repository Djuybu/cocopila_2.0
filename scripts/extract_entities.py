"""CLI entrypoint for P3-09: extract medical entities and relation evidence.

Lexicon-driven and standard-library only; the seed lexicon is minimal and is not
a medical ontology. Every mention keeps its chunk provenance.
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
from src.graph.ner import (
    extract_chunk_entities,
    extract_chunk_relations,
    load_lexicon,
    write_entities,
    write_relations,
)
from src.utils.io import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_entities")
DEFAULT_LEXICON = REPO_ROOT / "configs" / "graph" / "lexicon_seed.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-09: lexicon-driven medical entity extraction")
    parser.add_argument("--chunks", type=Path, required=True, help="Corpus chunks JSON/JSONL with chunk_id/doc_id/text")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON, help="Lexicon YAML (default: seed lexicon)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for entities.jsonl and report")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.chunks).exists():
        raise SystemExit(f"Chunks file not found: {args.chunks}")
    if not Path(args.lexicon).exists():
        raise SystemExit(f"Lexicon file not found: {args.lexicon}")
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")

    lexicon = load_lexicon(args.lexicon)
    chunks = load_records(args.chunks)
    logger.info("Extracting entities from %d chunks (lexicon %s)", len(chunks), lexicon["version"])
    records = extract_chunk_entities(chunks, lexicon)
    edges = extract_chunk_relations(chunks, lexicon)

    output.mkdir(parents=True)
    write_entities(output / "entities.jsonl", records)
    write_relations(output / "relations.jsonl", edges)
    by_type = {}
    for record in records:
        by_type[record["type"]] = by_type.get(record["type"], 0) + 1
    report = {"lexicon_version": lexicon["version"], "chunk_count": len(chunks),
              "entity_mention_count": len(records), "relation_count": len(edges),
              "mentions_by_type": by_type}
    write_json(output / "entity_extraction_report.json", report)

    print("\n" + "=" * 72)
    print("P3-09: ENTITY EXTRACTION")
    print("=" * 72)
    print(f"Chunks            : {len(chunks)}")
    print(f"Entity mentions   : {len(records)} {by_type}")
    print(f"Relation edges    : {len(edges)}")
    print(f"Output directory  : {output}")
    print("=" * 72 + "\n")
    return report


if __name__ == "__main__":
    main()
