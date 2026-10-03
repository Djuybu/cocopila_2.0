"""CLI entrypoint for P3-16: record a submission in the submission log.

Follows the P3 schema, exports an XLSX mirror on request, and refuses to exceed
five private submissions. Every row records run_id, split, archive, git commit,
config, model, local/public/private score and notes.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path (matches existing P2 script entrypoints).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p3.official_submission import git_provenance
from src.p3.submission_log import (
    PRIVATE_LIMIT,
    append_submission,
    export_xlsx,
    read_log,
    summarize_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("log_submission")
DEFAULT_LOG = REPO_ROOT / "submissions" / "submission_log.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-16: record a leaderboard submission")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="CSV log path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=["public", "private"], required=True)
    parser.add_argument("--archive", required=True, help="submission_<run_id>.zip path")
    parser.add_argument("--git-commit", help="Commit that produced the archive (default: current HEAD)")
    parser.add_argument("--config", help="Config/run directory or YAML used")
    parser.add_argument("--model", help="Model/reranker description")
    parser.add_argument("--local-score", type=float, help="Local metric (e.g. macro F2)")
    parser.add_argument("--public-score", type=float)
    parser.add_argument("--private-score", type=float)
    parser.add_argument("--notes", default="")
    parser.add_argument("--xlsx", type=Path, help="Optional XLSX mirror to write")
    parser.add_argument("--allow-private-overflow", action="store_true",
                        help="Explicitly override the 5-private-submission guard")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    record = {"run_id": args.run_id, "split": args.split, "archive": args.archive,
              "git_commit": args.git_commit, "config": args.config, "model": args.model,
              "local_score": args.local_score if args.local_score is not None else "",
              "public_score": args.public_score if args.public_score is not None else "",
              "private_score": args.private_score if args.private_score is not None else "",
              "notes": args.notes}
    if not record["git_commit"]:
        record["git_commit"] = git_provenance(REPO_ROOT).get("git_commit")
    append_submission(args.log, record, enforce_private_limit=not args.allow_private_overflow)
    rows = read_log(args.log)
    summary = summarize_log(rows)
    if args.xlsx is not None:
        export_xlsx(args.xlsx, rows)

    print("\n" + "=" * 72)
    print("P3-16: SUBMISSION LOGGED")
    print("=" * 72)
    print(f"Run / split      : {args.run_id} / {args.split}")
    print(f"Private used     : {summary['private_count']}/{PRIVATE_LIMIT} "
          f"(remaining {summary['private_remaining']})")
    print(f"Log              : {args.log}")
    if args.xlsx is not None:
        print(f"XLSX             : {args.xlsx}")
    print("=" * 72 + "\n")
    return summary


if __name__ == "__main__":
    main()
