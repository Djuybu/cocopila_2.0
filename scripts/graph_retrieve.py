"""CLI entrypoint for P3-12: retrieve candidates through the medical graph.

Returns candidates in the P1 candidate contract (source ``graph``) and, when
labels are supplied, an independent chunk/document Recall@K report.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path (matches existing P2 script entrypoints).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.ner import load_lexicon
from src.graph.retriever import GraphRetriever, evaluate_graph_recall, retrieve_candidate_records
from src.graph.store import Neo4jGraphStore, load_graph
from src.utils.config import load_config
from src.utils.io import read_json, write_json, write_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graph_retrieve")
DEFAULT_LEXICON = REPO_ROOT / "configs" / "graph" / "lexicon_seed.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-12: graph retrieval (GraphRetriever)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", type=Path, help="JSON graph from scripts/build_graph.py (memory backend)")
    source.add_argument("--config", type=Path, help="Neo4j connection YAML (with a neo4j: section)")
    parser.add_argument("--queries", type=Path, required=True, help="queries.json ([{id, text}])")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON, help="Lexicon YAML")
    parser.add_argument("--labels", type=Path, help="Optional labels.json for Recall@K")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for graph candidates")
    parser.add_argument("--top-k", type=int, default=200, help="Candidates per query (default 200)")
    parser.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100, 200], help="Recall cutoffs")
    parser.add_argument("--expand", action="store_true", help="Add one clinical-relation hop at half weight")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.queries).exists():
        raise SystemExit(f"Queries file not found: {args.queries}")
    if not Path(args.lexicon).exists():
        raise SystemExit(f"Lexicon file not found: {args.lexicon}")
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")

    if args.graph is not None:
        if not Path(args.graph).exists():
            raise SystemExit(f"Graph file not found: {args.graph}")
        store = load_graph(args.graph)
    else:
        store = Neo4jGraphStore.from_config(load_config(args.config) if args.config else {})

    retriever = GraphRetriever(store, load_lexicon(args.lexicon), expand=args.expand)
    queries = read_json(args.queries)
    logger.info("Retrieving graph candidates for %d queries (top_k=%d, expand=%s)",
                len(queries), args.top_k, args.expand)
    records = retrieve_candidate_records(retriever, queries, args.top_k)

    output.mkdir(parents=True)
    write_jsonl(output / "graph_candidates.jsonl", records)
    report = None
    if args.labels is not None:
        report = evaluate_graph_recall(retriever, queries, read_json(args.labels),
                                       ks=tuple(args.ks), top_k=args.top_k)
        write_json(output / "graph_recall_report.json", report)
    store.close()

    print("\n" + "=" * 72)
    print("P3-12: GRAPH RETRIEVAL")
    print("=" * 72)
    print(f"Queries        : {len(queries)}")
    print(f"Candidates     : {sum(len(record['candidates']) for record in records)}")
    if report is not None:
        print(f"Macro Recall@K : chunks={report['macro']['chunks']}, docs={report['macro']['documents']}")
    print(f"Output         : {output}")
    print("=" * 72 + "\n")
    return report


if __name__ == "__main__":
    main()
