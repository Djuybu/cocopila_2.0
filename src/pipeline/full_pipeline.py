"""Retrieval -> reranking -> scoring -> optional labelled evaluation."""
from src.pipeline.retrieve import run_retrieval
from src.pipeline.rerank import run_reranking
from src.pipeline.predict import run_prediction


def run_full_pipeline(config):
    run_dir = run_retrieval(config)
    run_reranking(run_dir)
    run_prediction(run_dir)
    if config.get("evaluation", {}).get("labels_path"):
        from src.pipeline.reporting import run_evaluation
        run_evaluation(run_dir)
    return run_dir
