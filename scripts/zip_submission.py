"""CLI entrypoint for P3-07: pack a validated submission JSON into a ZIP.

The ZIP contains exactly one JSON file at its root, matching the competition
rule. The P1/P2 ``pack_submission`` helper is reused read-only.
"""
import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path (matches existing P2 script entrypoints).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p3.submission import validator_from_registry
from src.submission.zipper import pack_submission
from src.utils.io import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zip_submission")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-07: pack a validated submission JSON into a ZIP")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Validated submission JSON")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Destination ZIP (must not exist)")
    parser.add_argument("--registry", type=Path, required=True, help="registry.json for corpus-aware validation")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for label, path in (("input", args.input), ("registry", args.registry)):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    validator = validator_from_registry(read_json(args.registry))
    logger.info("Packing %s -> %s", args.input, args.output)
    result = pack_submission(args.input, args.output, validator)
    print("\n" + "=" * 72)
    print("P3-07: SUBMISSION ZIP")
    print("=" * 72)
    print(f"ZIP (exactly one JSON at root): {result}")
    print("=" * 72 + "\n")
    return result


if __name__ == "__main__":
    main()
