"""Selection and official-ID conversion after scoring."""
from pathlib import Path
from src.data.loader import load_records
from src.data.adapter import official_candidates
from src.scoring.chunk_selector import select_results
from src.submission.generator import save_submission
from src.submission.validator import SubmissionValidator
from src.utils.config import load_config
from src.utils.io import read_json
from src.utils.logging import run_logger


def run_prediction(run_dir):
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    registry = read_json(run_dir / "registry.json")
    validator = SubmissionValidator(**registry)
    records = load_records(run_dir / "reranked.jsonl")
    predictions = [
        select_results(row["id"], official_candidates(
            row["candidates"], registry["internal_to_official"], registry["chunk_to_doc"]), config["selector"])
        for row in records
    ]
    output = save_submission(run_dir / "predictions.json", predictions, validator)
    run_logger(run_dir, config).info("selection output_path=%s", run_dir / "predictions.json")
    return output
