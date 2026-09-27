"""Export a portable P1 candidate run for P2; no retrieval or model inference."""
from pathlib import Path
import subprocess
import zipfile
import yaml

from src.data.download import sha256_file
from src.data.loader import load_records
from src.pipeline.retrieve import corpus_fingerprint, load_dataset
from src.retrieval.schema import SCHEMA_VERSION, standardize_candidates, validate_candidate_records
from src.submission.validator import SubmissionValidator
from src.utils.io import read_json, write_json, write_jsonl

BUNDLE_FILES = ("config.yaml", "queries.json", "candidates.jsonl", "labels.json", "registry.json", "manifest.json")


def validate_reranking_input(run_dir):
    """Read-only verification; works after copying/unzipping to another machine."""
    root = Path(run_dir)
    manifest = read_json(root / "manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported candidate schema version")
    expected = set(BUNDLE_FILES) - {"manifest.json"}
    if set(manifest["files"]) != expected:
        raise ValueError("Incomplete handoff manifest")
    for name in expected:
        if sha256_file(root / name) != manifest["files"][name]:
            raise ValueError(f"Handoff checksum mismatch: {name}")
    queries, records, registry = read_json(root / "queries.json"), load_records(root / "candidates.jsonl"), read_json(root / "registry.json")
    validate_candidate_records(records, queries, registry)
    if manifest["query_count"] != len(queries) or manifest["candidate_count"] != sum(len(row["candidates"]) for row in records):
        raise ValueError("Handoff manifest counts do not match the artifacts")
    valid, errors = SubmissionValidator(**registry).validate(read_json(root / "labels.json"))
    if not valid:
        raise ValueError("Invalid handoff labels: " + "; ".join(errors))
    return manifest


def export_reranking_input(config):
    cfg = config["handoff"]
    source, output = Path(cfg["source_run_dir"]), Path(cfg["output_dir"])
    archive = Path(cfg["archive_path"]) if cfg.get("archive_path") else None
    if output.exists() or (archive and archive.exists()):
        raise FileExistsError("Handoff directory/archive exists; choose new paths")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Handoff must be outside the source benchmark run")
    if archive and archive.resolve().is_relative_to(source.resolve()):
        raise ValueError("Handoff ZIP must be outside the source benchmark run")
    if archive and archive.resolve().is_relative_to(output.resolve()):
        raise ValueError("Handoff ZIP must be outside the bundle directory")
    summary = read_json(source / "benchmark.json")
    provenance = summary["provenance"]
    method = cfg["method"]
    if method not in summary["methods"] or Path(method).name != method:
        raise ValueError("Unknown/unsafe benchmark method")
    source_config = provenance["config"]
    chunks, queries, registry = load_dataset(source_config)
    labels = load_records(source_config["benchmark"]["labels_path"])
    for key, rows in (("corpus_fingerprint", chunks), ("queries_fingerprint", queries), ("labels_fingerprint", labels)):
        if corpus_fingerprint(rows) != provenance[key]:
            raise ValueError(f"Source data changed since benchmark: {key}")
    if read_json(source / "registry.json") != registry:
        raise ValueError("Benchmark registry differs from source data")
    manifest_path = source_config["data"].get("manifest_path")
    dataset = read_json(manifest_path) if manifest_path else {}
    raw_dir = dataset.get("preparation", {}).get("raw_dir")
    if raw_dir and any(p.resolve().is_relative_to(Path(raw_dir).resolve()) for p in (output, archive) if p):
        raise ValueError("Handoff outputs must not modify raw data")
    records = load_records(source / method / "candidates.jsonl")
    by_chunk = {row["chunk_id"]: row for row in chunks}
    for record in records:
        record["candidates"] = standardize_candidates(record["id"], record["candidates"], method=method)
        for candidate in record["candidates"]:
            original = by_chunk.get(candidate["chunk_id"])
            if original is None or candidate["text"] != original["text"] or candidate["doc_id"] != original["doc_id"]:
                raise ValueError("Candidate ID/text/parent does not match the source corpus")
    validate_candidate_records(records, queries, registry)
    valid, errors = SubmissionValidator(**registry).validate(labels)
    if not valid:
        raise ValueError("Invalid source labels: " + "; ".join(errors))
    reranker = dict(config["reranker"])
    if not reranker.get("enabled") or not reranker.get("model") or not reranker.get("device") or type(reranker.get("batch_size")) is not int or reranker["batch_size"] < 1:
        raise ValueError("Provide an enabled reranker model/device and positive batch_size")
    portable_config = {"run_name": output.name, "output_dir": "..", "reranker": reranker,
                       "evaluation": {"labels_path": "labels.json", "candidate_k": source_config["fusion"]["top_k"],
                                      "metrics": ["precision", "recall", "f1", "f2"], "zero_division": 0.0},
                       "dataset_split": source_config["data"]["split"], "label_quality": provenance["label_quality"]}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(portable_config, handle, allow_unicode=True, sort_keys=False)
    for name, rows in (("queries", queries), ("labels", labels), ("registry", registry)):
        write_json(output / f"{name}.json", rows)
    write_jsonl(output / "candidates.jsonl", records)
    project_root = Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True)
    state = subprocess.run(["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True)
    manifest = {"schema_version": SCHEMA_VERSION, "source_run_id": source_config["run_name"],
                "method": method, "retrieval_git_commit": provenance["git_commit"],
                "export_git_commit": head.stdout.strip() if head.returncode == 0 else None,
                "export_worktree_dirty": bool(state.stdout.strip()) if state.returncode == 0 else None,
                "dataset_split": source_config["data"]["split"], "label_quality": provenance["label_quality"],
                "id_namespace": dataset.get("id_namespace", "unspecified"), "dataset_source": dataset.get("source"),
                "limitations": "Source label quality is not upgraded by export. Weak prototype labels/IDs are not official competition relevance/IDs.",
                "fingerprints": {key: provenance[key] for key in ("corpus_fingerprint", "queries_fingerprint", "labels_fingerprint")},
                "query_count": len(queries), "candidate_count": sum(len(row["candidates"]) for row in records),
                "files": {name: sha256_file(output / name) for name in BUNDLE_FILES if name != "manifest.json"}}
    write_json(output / "manifest.json", manifest)
    validate_reranking_input(output)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as packed:
            for name in BUNDLE_FILES:
                packed.write(output / name, arcname=name)
    return output
