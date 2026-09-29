"""CLI for run_reranking; supports direct invocation or pip install -e ."""
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.cli import main

if __name__ == "__main__":
    main("run_reranking")
