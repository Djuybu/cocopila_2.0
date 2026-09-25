"""Legacy entrypoint and import aliases; full validation requires a registry."""
from src.pipeline.cli import main
from src.submission.zipper import pack_submission, validate_json

if __name__ == "__main__":
    main("pack_submission")
