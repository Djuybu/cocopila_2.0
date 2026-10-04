"""P3-15: generate the official submission JSON and ZIP with full provenance.

Builds predictions from the frozen P2 chunk selector + P3-14 doc pipeline,
validates against the official registry, packs exactly one JSON at the ZIP root,
and records config + git commit so every submission is traceable.
"""
from pathlib import Path
import subprocess
import zipfile

from src.data.download import sha256_file
from src.p3.predictions import build_predictions
from src.p3.submission import build_submission
from src.utils.io import write_json


def resolve_doc_config(pipeline_config):
    """Map a ``best_doc_pipeline.yaml`` config to ``build_predictions`` doc keys."""
    keys = ("doc_aggregation", "doc_top_k", "doc_weight", "doc_threshold", "doc_fallback", "doc_max")
    missing = [key for key in keys if key not in pipeline_config]
    if missing:
        raise ValueError(f"Doc pipeline config is missing keys: {missing}")
    return {key: pipeline_config[key] for key in keys}


def git_provenance(project_root=None):
    """Record the commit and dirty state of the worktree that produced the run."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    state = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return {"git_commit": head.stdout.strip() if head.returncode == 0 else None,
            "worktree_dirty": bool(state.stdout.strip()) if state.returncode == 0 else None}


def generate_official_submission(scored_records, chunk_config, pipeline_config, validator, *,
                                 run_id, output_dir, submission_dir=None, project_root=None):
    """Write ``submission_<run_id>.json`` + ``.zip`` + manifest; returns their paths."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a nonempty string")
    output = Path(output_dir)
    json_path = output / f"submission_{run_id}.json"
    if json_path.exists():
        raise FileExistsError(f"Submission JSON already exists: {json_path}")
    predictions = build_predictions(scored_records, chunk_config, resolve_doc_config(pipeline_config),
                                    internal_to_official=validator.internal_to_official,
                                    chunk_to_doc=validator.chunk_to_doc)
    normalized = build_submission(predictions, validator)
    write_json(json_path, normalized)

    archive = Path(submission_dir or output) / f"submission_{run_id}.zip"
    if archive.exists():
        raise FileExistsError(f"Submission ZIP already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as packed:
        packed.write(json_path, arcname=json_path.name)
    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        if len(names) != 1 or "/" in names[0] or "\\" in names[0] or not names[0].endswith(".json"):
            raise ValueError("ZIP must contain exactly one JSON at its root")

    manifest = {
        "schema_version": "medical-rag-submission-v1",
        "run_id": run_id,
        "query_count": len(normalized),
        "document_count": sum(len(row["relevant_docs"]) for row in normalized),
        "chunk_count": sum(len(row["relevant_chunks"]) for row in normalized),
        "chunk_selector": dict(chunk_config),
        "doc_pipeline": dict(pipeline_config),
        "json_file": json_path.name,
        "zip_file": archive.name,
        "json_sha256": sha256_file(json_path),
        "zip_sha256": sha256_file(archive),
        **git_provenance(project_root),
    }
    manifest_path = output / f"submission_{run_id}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Submission manifest already exists: {manifest_path}")
    write_json(manifest_path, manifest)
    return {"json": str(json_path), "zip": str(archive), "manifest": str(manifest_path),
            "manifest_payload": manifest}
