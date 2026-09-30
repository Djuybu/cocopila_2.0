"""Member 3 (P3) workstream: hierarchy, document aggregation/selection, submission.

P1/P2 modules are imported read-only; P3 never rewrites their artifacts.
"""
from src.p3.doc_aggregation import (
    AGGREGATION_METHODS,
    aggregate_documents,
    aggregation_spec,
    choose_baseline,
    evaluate_aggregation,
    prepare_official_candidates,
    validate_query_coverage,
)
from src.p3.doc_selector import (
    DEFAULT_FALLBACKS,
    DEFAULT_MAXIMUMS,
    default_doc_thresholds,
    doc_selector_config,
    load_doc_selector_config,
    sweep_doc_selector,
    write_best_doc_selector,
)
from src.p3.doc_pipeline import (
    build_doc_scores,
    doc_pipeline_config,
    document_scores_for_query,
    load_doc_pipeline_config,
    macro_f2_doc,
    tune_doc_pipeline,
    write_doc_pipeline,
)
from src.p3.hierarchy import (
    build_hierarchy,
    load_registry,
    reconcile_with_registry,
    run_hierarchy,
    write_hierarchy,
)
from src.p3.consistency import (
    POLICIES,
    analyze_consistency,
    apply_policy,
    summarize_consistency,
)
from src.p3.label_audit import (
    audit_label_hierarchy,
    render_markdown,
    summarize_label_audit,
    write_label_audit,
)
from src.p3.predictions import build_predictions, load_chunk_selector_config
from src.p3.submission import (
    build_submission,
    normalize_predictions,
    validator_from_registry,
    write_submission,
)
from src.p3.submission_log import (
    LOG_COLUMNS,
    PRIVATE_LIMIT,
    append_submission,
    export_xlsx,
    read_log,
    summarize_log,
    write_log,
)

from src.p3.official_submission import (
    generate_official_submission,
    git_provenance,
    resolve_doc_config,
)
from src.p3.working_notes import collect_metrics, export_working_notes, render_working_notes

__all__ = [
    "build_hierarchy",
    "load_registry",
    "reconcile_with_registry",
    "run_hierarchy",
    "write_hierarchy",
    "POLICIES",
    "analyze_consistency",
    "apply_policy",
    "summarize_consistency",
    "audit_label_hierarchy",
    "render_markdown",
    "summarize_label_audit",
    "write_label_audit",
    "build_predictions",
    "load_chunk_selector_config",
    "build_submission",
    "normalize_predictions",
    "validator_from_registry",
    "write_submission",
    "generate_official_submission",
    "git_provenance",
    "resolve_doc_config",
    "LOG_COLUMNS",
    "PRIVATE_LIMIT",
    "append_submission",
    "export_xlsx",
    "read_log",
    "summarize_log",
    "write_log",
    "collect_metrics",
    "export_working_notes",
    "render_working_notes",
    "AGGREGATION_METHODS",
    "aggregate_documents",
    "aggregation_spec",
    "choose_baseline",
    "evaluate_aggregation",
    "prepare_official_candidates",
    "validate_query_coverage",
    "DEFAULT_FALLBACKS",
    "DEFAULT_MAXIMUMS",
    "default_doc_thresholds",
    "doc_selector_config",
    "load_doc_selector_config",
    "sweep_doc_selector",
    "write_best_doc_selector",
    "build_doc_scores",
    "doc_pipeline_config",
    "document_scores_for_query",
    "load_doc_pipeline_config",
    "macro_f2_doc",
    "tune_doc_pipeline",
    "write_doc_pipeline",
]
