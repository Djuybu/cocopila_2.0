"""Argument parsing shared by thin scripts. Install the package before use."""
import argparse
import logging
from pathlib import Path
from src.utils.config import load_config


def main(command):
    parser = argparse.ArgumentParser(description=f"Medical competition: {command}")
    if command in {"prepare_data", "build_bm25_index", "build_dense_index", "run_retrieval", "run_full_pipeline"}:
        parser.add_argument("--config", "--config-path", required=True, type=Path)
    elif command == "pack_submission":
        parser.add_argument("--input", "-i", required=True, type=Path)
        parser.add_argument("--output", "-o", required=True, type=Path)
        parser.add_argument("--registry", required=True, type=Path)
    else:
        parser.add_argument("--run-dir", type=Path)
        if command == "make_submission":
            parser.add_argument("--run")
            parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
            parser.add_argument("--submission-dir", type=Path)
        if command == "run_evaluation":
            parser.add_argument("--ground-truth", type=Path)
            parser.add_argument("--stage", choices=["predictions", "candidates"], default="predictions")
            parser.add_argument("--k", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(args, "config"):
        config = load_config(args.config)
        if command == "prepare_data":
            from src.data.loader import prepare_data
            result = prepare_data(config)
            logging.info("Prepared %s", {kind: len(rows) for kind, rows in result.items()})
            return
        from src.pipeline.retrieve import build_bm25_index, build_dense_index, run_retrieval
        from src.pipeline.full_pipeline import run_full_pipeline
        actions = {"build_bm25_index": build_bm25_index, "build_dense_index": build_dense_index,
                   "run_retrieval": run_retrieval, "run_full_pipeline": run_full_pipeline}
        result = actions[command](config)
    elif command == "pack_submission":
        from src.submission.validator import SubmissionValidator
        from src.submission.zipper import pack_submission
        from src.utils.io import read_json
        result = pack_submission(args.input, args.output, SubmissionValidator(**read_json(args.registry)))
    else:
        run_dir = args.run_dir
        if command == "make_submission" and args.run:
            if run_dir is not None:
                parser.error("Choose --run-dir or --run, not both")
            from src.utils.config import run_directory
            run_dir = run_directory({"run_name": args.run, "output_dir": args.output_dir})
        if run_dir is None:
            parser.error("--run-dir is required (or --run for make_submission)")
        if command == "run_reranking":
            from src.pipeline.rerank import run_reranking
            result = run_reranking(run_dir)
        elif command == "run_prediction":
            from src.pipeline.predict import run_prediction
            run_prediction(run_dir)
            result = run_dir / "predictions.json"
        elif command == "run_evaluation":
            from src.pipeline.reporting import run_evaluation
            result = run_evaluation(run_dir, args.ground_truth, args.stage, args.k)["macro"]
        else:
            from src.pipeline.reporting import make_submission
            result = make_submission(run_dir, args.submission_dir)
    logging.info("Completed %s: %s", command, result)
