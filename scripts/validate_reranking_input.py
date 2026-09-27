"""Validate a received P1/P2 handoff without loading any model."""
from src.pipeline.cli import main

if __name__ == "__main__":
    main("validate_reranking_input")
