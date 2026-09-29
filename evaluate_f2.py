"""Root CLI entrypoint for P2-02 evaluate_f2.py deliverable.

Directly callable:
    python evaluate_f2.py --handoff-dir data/p1_p2_handoff_qwen3 --top-k 1
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.evaluate_f2 import main

if __name__ == "__main__":
    main()
