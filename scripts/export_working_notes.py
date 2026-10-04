"""CLI entrypoint for P3-17: fill the working-notes outline from experiment artifacts.

Reads P2/P3/submission-log artifacts (read-only) and writes a filled copy of
``docs/working_notes_outline.md`` plus ``metrics_summary.json``. Missing results
are marked as pending instead of being invented.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path (matches existing P2 script entrypoints).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p3.working_notes import export_working_notes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_working_notes")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-17: export the working-notes outline")
    parser.add_argument("--p2-run-dir", type=Path, help="P2 run dir with selector_cv_report.json")
    parser.add_argument("--p3-doc-dir", type=Path, help="P3-14 output dir with doc_pipeline_report.json")
    parser.add_argument("--submission-log", type=Path, help="submissions/submission_log.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the filled outline")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}")
    logger.info("P3-17: assembling working notes from available artifacts")
    result = export_working_notes(output, p2_run_dir=args.p2_run_dir,
                                  p3_doc_dir=args.p3_doc_dir, submission_log=args.submission_log)
    print("\n" + "=" * 72)
    print("P3-17: WORKING NOTES EXPORTED")
    print("=" * 72)
    print(f"Outline : {result['outline']}")
    print(f"Metrics : {result['metrics']}")
    print("=" * 72 + "\n")
    return result


if __name__ == "__main__":
    main()
