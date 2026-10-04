"""CLI entrypoint for P3-07: validate a submission JSON or ZIP against the registry.

Corpus-aware validation is mandatory: an official ``registry.json`` supplies the
expected queries, document IDs and chunk->doc mapping.
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
from src.submission.zipper import validate_zip
from src.utils.io import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_submission")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3-07: validate a submission JSON or ZIP")
    parser.add_argument("--input", "-i", type=Path, required=True, help="submission JSON or ZIP")
    parser.add_argument("--registry", type=Path, required=True, help="registry.json for corpus-aware validation")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.registry.exists():
        raise SystemExit(f"Registry not found: {args.registry}")
    validator = validator_from_registry(read_json(args.registry))

    if args.input.suffix == ".zip":
        logger.info("Validating ZIP %s", args.input)
        data = validate_zip(args.input, validator)
        valid, errors = True, []
    else:
        logger.info("Validating JSON %s", args.input)
        data = read_json(args.input)
        valid, errors = validator.validate(data)

    print("\n" + "=" * 72)
    print("P3-07: SUBMISSION VALIDATION")
    print("=" * 72)
    print(f"Input   : {args.input}")
    print(f"Queries : {len(data)}")
    print(f"Valid   : {valid}")
    for error in errors:
        print(f"  - {error}")
    print("=" * 72 + "\n")
    if not valid:
        raise SystemExit(1)
    return data


if __name__ == "__main__":
    main()
