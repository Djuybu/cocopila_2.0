"""Full end-to-end pipeline execution for Member 2 (P2: Reranking & Chunk Selection)."""
import logging
from pathlib import Path
import shutil

from src.data.loader import load_records
from src.pipeline.handoff import validate_reranking_input
from src.pipeline.rerank import run_reranking
from src.pipeline.reranker_benchmark import benchmark_reranking
from src.scoring.sweep import (
    analyze_false_negatives,
    package_p2_to_p3_handoff,
    sweep_chunk_selector,
)
from src.utils.config import load_config
from src.utils.io import read_json


logger = logging.getLogger("p2_pipeline")


def run_p2_full_pipeline(
    handoff_dir,
    output_dir,
    *,
    reranker_config=None,
    p3_handoff_dir=None,
    reranker=None,
    sweep_thresholds=None,
    sweep_fallbacks=None,
    sweep_maximums=None,
    benchmark=True,
    archive_zip=True,
):
    """Execute the complete end-to-end P2 pipeline from P1 bundle to P3 deliverables.

    Steps:
        1. Validate P1 handoff integrity.
        2. Execute reranking scoring (with benchmark metrics & latency profiling).
        3. Sweep chunk selector parameters (threshold, fallback, max) optimizing Macro Chunk F2.
        4. Analyze False Negative error causes (P1 Retrieval Miss vs P2 Pruning).
        5. Package final deliverables into p2_to_p3_handoff for Member 3.

    Returns:
        dict: Complete pipeline execution summary and artifact paths.
    """
    handoff_path = Path(handoff_dir).resolve()
    output_path = Path(output_dir).resolve()
    p3_handoff_path = (
        Path(p3_handoff_dir).resolve()
        if p3_handoff_dir
        else output_path / "p2_to_p3_handoff"
    )

    logger.info("Stage 1/5: Validating P1 handoff bundle at %s...", handoff_path)
    manifest = validate_reranking_input(handoff_path)

    logger.info("Stage 2/5: Executing Reranker scoring into %s...", output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    benchmark_report = None
    if benchmark:
        benchmark_report = benchmark_reranking(
            handoff_path,
            output_path,
            reranker_config=reranker_config,
            reranker=reranker,
        )
    else:
        run_reranking(
            handoff_path,
            output_dir=output_path,
            reranker_config=reranker_config,
            reranker=reranker,
            measure=True,
        )

    scored_records = load_records(output_path / "reranked.jsonl")
    labels = read_json(output_path / "labels.json")
    registry = read_json(output_path / "registry.json")
    mapping = registry.get("internal_to_official", {})
    chunk_to_doc = registry.get("chunk_to_doc")

    logger.info("Stage 3/5: Running joint chunk selector sweep for %d queries...", len(scored_records))
    best_config, sweep_df, plateau_info = sweep_chunk_selector(
        scored_records=scored_records,
        labels=labels,
        thresholds=sweep_thresholds,
        fallbacks=sweep_fallbacks,
        maximums=sweep_maximums,
        internal_to_official=mapping,
        chunk_to_doc=chunk_to_doc,
    )
    logger.info(
        "Optimal selector found: threshold=%s, fallback=%d, max=%d -> Macro F2=%.4f",
        best_config["chunk_threshold"],
        best_config["chunk_fallback"],
        best_config["chunk_max"],
        best_config["macro_f2"],
    )

    logger.info("Stage 4/5: Analyzing False Negative error causes...")
    fn_df, fn_summary = analyze_false_negatives(
        scored_records=scored_records,
        labels=labels,
        best_config=best_config,
        internal_to_official=mapping,
        chunk_to_doc=chunk_to_doc,
    )
    logger.info(
        "FN Analysis: Total FN=%d (P1 Retrieval Miss: %d, P2 Pruned: %d)",
        fn_summary["total_false_negatives"],
        fn_summary["p1_retrieval_miss"],
        fn_summary["p2_threshold_pruned"] + fn_summary["p2_max_cap_pruned"],
    )

    logger.info("Stage 5/5: Packaging deliverables for Member 3 (P3) into %s...", p3_handoff_path)
    package_p2_to_p3_handoff(
        output_dir=p3_handoff_path,
        source_run_dir=output_path,
        best_config=best_config,
        sweep_df=sweep_df,
        fn_df=fn_df,
        benchmark_report=benchmark_report,
        archive_zip=archive_zip,
    )

    summary = {
        "status": "success",
        "stage": "p2_pipeline_complete",
        "p1_source": {
            "path": str(handoff_path),
            "query_count": manifest["query_count"],
            "candidate_count": manifest["candidate_count"],
        },
        "reranking": {
            "output_dir": str(output_path),
            "reranked_file": str(output_path / "reranked.jsonl"),
            "benchmark_report": benchmark_report,
        },
        "optimization": {
            "best_config": best_config,
            "plateau_info": plateau_info,
            "fn_summary": fn_summary,
        },
        "p3_handoff": {
            "output_dir": str(p3_handoff_path),
            "manifest_file": str(p3_handoff_path / "manifest.json"),
            "best_selector_yaml": str(p3_handoff_path / "best_chunk_selector.yaml"),
            "selected_chunks_json": str(p3_handoff_path / "p2_selected_chunks.json"),
            "reranked_chunks_jsonl": str(p3_handoff_path / "reranked.jsonl"),
            "sweep_csv": str(p3_handoff_path / "threshold_sweep.csv"),
            "fn_analysis_csv": str(p3_handoff_path / "fn_analysis.csv"),
            "archive_zip": str(p3_handoff_path.parent / f"{p3_handoff_path.name}.zip") if archive_zip else None,
        },
    }
    logger.info("P2 Pipeline completed successfully! Ready for Member 3.")
    return summary
