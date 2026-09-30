"""CLI entrypoint for P3-11: build the medical graph (Neo4j or in-memory JSON).

Loads Document/Chunk/Entity nodes and HAS_CHUNK/MENTIONS/TREATS/HAS_SYMPTOM edges
from the P3-09/P3-10 artifacts. The Neo4j backend needs ``pip install -e ".[graph]"``
and a running server; the in-memory backend writes a JSON graph for offline use.
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
from src.graph.store import InMemoryGraphStore, Neo4jGraphStore, build_graph, save_graph
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_graph")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-11: build the medical graph")
    parser.add_argument("--chunks", type=Path, required=True, help="Corpus chunks with chunk_id/doc_id/text")
    parser.add_argument("--documents", type=Path, help="Optional documents JSON/JSONL")
    parser.add_argument("--entities", type=Path, required=True, help="normalized_entities.jsonl from P3-10")
    parser.add_argument("--relations", type=Path, help="relations.jsonl from P3-09 (optional)")
    parser.add_argument("--backend", choices=["memory", "neo4j"], default="memory")
    parser.add_argument("--config", type=Path, help="Neo4j connection YAML (with a neo4j: section)")
    parser.add_argument("--output", type=Path, help="JSON graph path (required for the memory backend)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.chunks).exists():
        raise SystemExit(f"Chunks file not found: {args.chunks}")
    if not Path(args.entities).exists():
        raise SystemExit(f"Entities file not found: {args.entities}")
    if args.backend == "memory":
        if args.output is None:
            raise SystemExit("--output is required for the memory backend")
        if args.output.exists():
            raise SystemExit(f"Output already exists: {args.output}")

    documents = load_records(args.documents) if args.documents else []
    chunks = load_records(args.chunks)
    entities = load_records(args.entities)
    relations = load_records(args.relations) if args.relations else []
    logger.info("Building graph: %d docs, %d chunks, %d entity mentions, %d relations",
                len(documents), len(chunks), len(entities), len(relations))

    if args.backend == "memory":
        store = InMemoryGraphStore()
        build_graph(store, documents, chunks, entities, relations)
        save_graph(args.output, store)
        stats = store.stats()
        target = str(args.output)
    else:
        config = load_config(args.config) if args.config else {}
        store = Neo4jGraphStore.from_config(config)
        store.create_constraints()
        build_graph(store, documents, chunks, entities, relations)
        stats = store.stats()
        store.close()
        target = f"{stats.get('uri')}/{stats.get('database')}"

    print("\n" + "=" * 72)
    print("P3-11: GRAPH BUILT")
    print("=" * 72)
    print(f"Backend : {stats['backend']}")
    print(f"Nodes   : {stats['node_count']}  (chunks={stats.get('chunk_count', 'n/a')}, "
          f"entities={stats.get('entity_count', 'n/a')})")
    print(f"Edges   : {stats['edge_count']}")
    print(f"Target  : {target}")
    print("=" * 72 + "\n")
    return stats


if __name__ == "__main__":
    main()
