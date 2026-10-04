"""Package the current crawler as a self-contained private Kaggle notebook."""
import argparse
from pathlib import Path

from src.data.kaggle_crawl import prepare_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", default="vibiomir-rag-crawl-pilot-20261003")
    parser.add_argument("--sample-per-host", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/kaggle_vibiomir_bundle")
    args = parser.parse_args()
    bundle = prepare_bundle(Path(__file__).resolve().parents[1], args.output_dir,
                            owner=args.owner, slug=args.slug, sample_per_host=args.sample_per_host)
    print(f"Prepared Kaggle notebook: {bundle}/ViBioMIR_RAG_Kaggle.ipynb")
    print(f"Run: kaggle kernels push -p {bundle}")
