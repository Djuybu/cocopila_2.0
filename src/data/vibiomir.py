"""Inspect the official ViBioMIR URL catalogue without inventing text or labels."""
from collections import Counter
import logging
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from src.data.download import _download_file, sha256_file
from src.utils.io import write_json

LOGGER = logging.getLogger(__name__)
FILES = ("query.parquet", "links_corpus.parquet")


def download_catalogue(cfg):
    destination = Path(cfg["raw_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{cfg['repo']}/resolve/{cfg['revision']}"
    for name in FILES:
        spec = cfg["files"][name]
        _download_file(base, destination, {"path": name, "size": spec["size"],
                                        "lfs": {"oid": spec["sha256"]}}, attempts=3)


def _require_schema(parquet, fields):
    import pyarrow as pa
    schema = parquet.schema_arrow
    if schema.names != list(fields):
        raise ValueError(f"Unexpected ViBioMIR columns: {schema.names}")
    for name, kind in fields.items():
        datatype = schema.field(name).type
        valid = pa.types.is_integer(datatype) if kind == "integer" else pa.types.is_string(datatype)
        if not valid:
            raise ValueError(f"Unexpected type for {name}: {datatype}")


def analyze_catalogue(query_path, corpus_path):
    """Read every URL in batches; exact duplicate checks retain ID and URL sets."""
    import pyarrow.parquet as pq
    queries_file, corpus_file = pq.ParquetFile(query_path), pq.ParquetFile(corpus_path)
    _require_schema(queries_file, {"id": "integer", "query": "string"})
    _require_schema(corpus_file, {"id": "integer", "url": "string"})
    rows = queries_file.read().to_pylist()
    query_ids = [r["id"] for r in rows]
    if any(i is None or i <= 0 for i in query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Query IDs must be unique positive integers")
    if not rows or any(not isinstance(r["query"], str) or not r["query"].strip() for r in rows):
        raise ValueError("Queries must contain nonempty text")
    # String keys satisfy the current pipeline; retain the original numeric identity.
    queries = [{"id": str(r["id"]), "text": r["query"], "official_id": r["id"]} for r in rows]
    lengths = np.array([len(r["query"].split()) for r in rows])
    ids, seen_urls = set(), set()
    hosts, schemes = Counter(), Counter()
    examples, duplicate_examples = {}, []
    count = duplicate_urls = null_urls = invalid_urls = whitespace_urls = 0
    for batch in corpus_file.iter_batches(batch_size=65536):
        for row in batch.to_pylist():
            count += 1
            doc_id, url = row["id"], row["url"]
            if doc_id is None or doc_id <= 0 or doc_id in ids:
                raise ValueError(f"Duplicate or invalid official document ID: {doc_id}")
            ids.add(doc_id)
            if url is None or not url.strip():
                null_urls += 1
                continue
            if url in seen_urls:
                duplicate_urls += 1
                if len(duplicate_examples) < 10:
                    duplicate_examples.append({"id": doc_id, "url": url})
            seen_urls.add(url)
            whitespace_urls += int(url != url.strip())
            try:
                parsed = urlsplit(url)
                host = (parsed.hostname or "").lower()
            except ValueError:
                invalid_urls += 1
                continue
            if parsed.scheme not in {"http", "https"} or not host:
                invalid_urls += 1
            hosts[host] += 1
            schemes[parsed.scheme] += 1
            examples.setdefault(host, {"id": doc_id, "url": url})
        if count % (8 * 65536) == 0 or count == corpus_file.metadata.num_rows:
            LOGGER.info("Inspected %s/%s links", count, corpus_file.metadata.num_rows)
    minimum, maximum = (min(ids), max(ids)) if ids else (None, None)
    top_hosts = [{"host": host, "count": n, "fraction": n / count, "example": examples[host]}
                 for host, n in hosts.most_common(30)]
    report = {
        "stage": "official_url_catalogue",
        "availability": {"queries": True, "document_urls": True, "document_text": False,
                         "chunks": False, "relevance_labels": False, "held_out_split": False},
        "query": {"rows": len(rows), "schema": str(queries_file.schema_arrow),
                  "id_min": min(query_ids), "id_max": max(query_ids),
                  "duplicate_texts": len(rows) - len({r["query"] for r in rows}),
                  "whitespace_words": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                                       "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max())}},
        "corpus": {"rows": count, "schema": str(corpus_file.schema_arrow),
                   "id_min": minimum, "id_max": maximum, "unique_ids": len(ids),
                   "missing_ids_within_range": maximum - minimum + 1 - len(ids) if ids else 0,
                   "unique_nonempty_urls": len(seen_urls), "duplicate_url_rows": duplicate_urls,
                   "duplicate_examples": duplicate_examples, "null_or_blank_urls": null_urls,
                   "invalid_http_urls": invalid_urls, "surrounding_whitespace_urls": whitespace_urls,
                   "schemes": dict(schemes), "domain_count": len(hosts),
                   "domain_counts": dict(hosts.most_common()), "top_hosts": top_hosts},
    }
    return queries, report


def study_catalogue(cfg, download=False):
    """Verify pinned files, export queries and write provenance plus full-data EDA."""
    raw, output, reports = map(Path, (cfg["raw_dir"], cfg["output_dir"], cfg["report_dir"]))
    targets = [output / "queries.json", output / "dataset_manifest.json", reports / "dataset_report.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Research outputs exist; select new output_dir and report_dir")
    if any(p.resolve().is_relative_to(raw.resolve()) for p in targets):
        raise ValueError("Research outputs must be outside the raw directory")
    if download:
        download_catalogue(cfg)
    fingerprints = {}
    for name in FILES:
        path, spec = raw / name, cfg["files"][name]
        digest = sha256_file(path)
        if path.stat().st_size != spec["size"] or digest != spec["sha256"]:
            raise ValueError(f"Official file fails pinned size/SHA256 check: {path}")
        fingerprints[name] = {"size": path.stat().st_size, "sha256": digest}
    queries, report = analyze_catalogue(raw / FILES[0], raw / FILES[1])
    manifest = {"source": {"repo": cfg["repo"], "revision": cfg["revision"]},
                "files": fingerprints, "id_namespace": "official", "stage": report["stage"],
                "availability": report["availability"], "counts": {"queries": len(queries),
                "document_urls": report["corpus"]["rows"]},
                "official_id_type": "integer", "pipeline_query_id_type": "decimal_string",
                "limitations": "URL catalogue only: fetch document text before indexing; no qrels or official chunk IDs supplied."}
    report["source"] = manifest["source"]
    report["files"] = fingerprints
    write_json(targets[0], queries)
    write_json(targets[1], manifest)
    write_json(targets[2], report)
    return report
