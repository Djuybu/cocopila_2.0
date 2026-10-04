"""Stratified capacity estimation and streaming export of an acquired URL corpus."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import logging
from pathlib import Path
import random

import numpy as np
import yaml

from src.data.download import sha256_file
from src.data.vibiomir import _require_schema
from src.data.web_corpus import acquire, chunk_document, encoded, open_state
from src.utils.io import read_json, write_json

LOGGER = logging.getLogger(__name__)
METRICS = ("wire_bytes", "response_body_bytes", "raw_gzip_bytes", "text_bytes",
           "document_jsonl_bytes", "chunks_jsonl_bytes", "chunk_count", "state_json_bytes", "mapping_bytes", "fetch_log_jsonl_bytes")


def validate_settings(settings):
    for name in ("workers", "sample_per_host", "max_response_bytes", "min_text_chars", "bootstrap_replicates", "embedding_dimension"):
        if not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["timeout_seconds"] <= 0 or settings["delay_per_host_seconds"] < 0:
        raise ValueError("Invalid timeout or host delay")
    if not isinstance(settings["chunk_chars"], int) or not isinstance(settings["chunk_overlap"], int):
        raise ValueError("Chunk parameters must be integers")
    if not 0 <= settings["chunk_overlap"] < settings["chunk_chars"]:
        raise ValueError("Require chunk_chars > chunk_overlap >= 0")
    for name in ("retry_attempts", "max_extracted_text_bytes"):
        if name in settings and (not isinstance(settings[name], int) or settings[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("robots_cache_seconds", "robots_failure_ttl_seconds", "retry_backoff_seconds"):
        if name in settings and settings[name] < 0:
            raise ValueError(f"{name} must be nonnegative")


def source_identity(config):
    data, settings = config["data"], config["rag"]
    validate_settings(settings)
    fingerprints = {}
    for name in ("query.parquet", "links_corpus.parquet"):
        path = Path(data["raw_dir"]) / name
        spec = data["files"][name]
        digest = sha256_file(path)
        if path.stat().st_size != spec["size"] or digest != spec["sha256"]:
            raise ValueError(f"Source fails pinned size/SHA256 check: {path}")
        fingerprints[name] = digest
    return {"repo": data["repo"], "revision": data["revision"], "files": fingerprints,
            "settings": {k: v for k, v in settings.items() if k not in {"workers", "bootstrap_replicates", "sample_per_host", "seed"}}}


def catalogue_rows(path):
    import pyarrow.parquet as pq
    from urllib.parse import urlsplit
    source = pq.ParquetFile(path)
    _require_schema(source, {"id": "integer", "url": "string"})
    for batch in source.iter_batches(batch_size=65536):
        for row in batch.to_pylist():
            if not isinstance(row["id"], int) or row["id"] <= 0 or not isinstance(row["url"], str):
                raise ValueError("Invalid catalogue ID or URL")
            parsed = urlsplit(row["url"])
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Invalid catalogue URL")
            yield {**row, "host": parsed.hostname.lower()}


def stratified_sample(rows, per_host, seed):
    """Uniform reservoir within every hostname, including small sources in full."""
    counts, samples, generators = Counter(), defaultdict(list), {}
    for row in rows:
        host = row["host"]
        counts[host] += 1
        if host not in generators:
            generators[host] = random.Random(int(hashlib.sha256(f"{seed}:{host}".encode()).hexdigest(), 16))
        bucket = samples[host]
        if len(bucket) < per_host:
            bucket.append(row)
        else:
            index = generators[host].randrange(counts[host])
            if index < per_host:
                bucket[index] = row
    return dict(counts), sorted((row for bucket in samples.values() for row in bucket), key=lambda r: (r["host"], r["id"]))


def _result_metrics(result, settings):
    values = [float(result.get(name, 0)) for name in METRICS]
    values[METRICS.index("state_json_bytes")] = len(encoded(result))
    values[METRICS.index("fetch_log_jsonl_bytes")] = len(encoded({k: v for k, v in result.items() if k != "document"}))
    if result["status"] == "ok":
        chunks = chunk_document(result["document"], settings["chunk_chars"], settings["chunk_overlap"])
        values[METRICS.index("mapping_bytes")] = sum(
            len(encoded(c["chunk_id"]).strip()) * 3 + len(encoded(c["doc_id"]).strip()) + 8 for c in chunks)
    return values


def capacity_report(counts, results, settings):
    """Weight by population, never by equal-domain sample sizes; failures retain zero text."""
    by_host = defaultdict(list)
    for row in results:
        by_host[row["host"]].append(row)
    if set(counts) != set(by_host):
        raise ValueError("Estimate must cover every selected hostname")
    rng = np.random.default_rng(settings["seed"])
    bootstrap = np.zeros((settings["bootstrap_replicates"], len(METRICS)))
    total = np.zeros(len(METRICS))
    statuses, domains = Counter(), []
    expected_successes, unknown_urls, successful_log_bytes = 0.0, 0, 0.0
    for host, population in sorted(counts.items()):
        rows = by_host[host]
        values = np.array([_result_metrics(r, settings) for r in rows])
        total += population * values.mean(axis=0)
        if len(rows) == population:
            bootstrap += population * values.mean(axis=0)
        else:
            indices = rng.integers(0, len(rows), (settings["bootstrap_replicates"], len(rows)))
            bootstrap += population * values[indices].mean(axis=1)
        success = sum(r["status"] == "ok" for r in rows)
        expected_successes += population * success / len(rows)
        successful_log_bytes += population / len(rows) * sum(
            len(encoded({k: v for k, v in row.items() if k != "document"})) for row in rows if row["status"] == "ok")
        if not success:
            unknown_urls += population
        statuses.update(r["status"] for r in rows)
        domains.append({"host": host, "population_urls": population, "sample_urls": len(rows),
                        "sample_successes": success, "status_counts": dict(Counter(r["status"] for r in rows)),
                        "sample_mean_text_bytes_per_success": sum(r.get("text_bytes", 0) for r in rows) / success if success else None})
    metrics = {name: {"expected_total": float(total[i]),
                       "bootstrap_95_percent": [float(x) for x in np.percentile(bootstrap[:, i], [2.5, 97.5])]}
               for i, name in enumerate(METRICS)}
    urls = sum(counts.values())
    payload_indices = [METRICS.index(name) for name in ("document_jsonl_bytes", "chunks_jsonl_bytes", "mapping_bytes", "fetch_log_jsonl_bytes")]
    payload = total[payload_indices].sum()
    payload_bootstrap = bootstrap[:, payload_indices].sum(axis=1)
    raw = total[METRICS.index("raw_gzip_bytes")] if settings.get("keep_raw") else 0
    state = total[METRICS.index("state_json_bytes")]
    chunks = total[METRICS.index("chunk_count")]
    hypothetical = None
    if expected_successes > 0:
        hypothetical = {"assumption": "All currently unavailable URLs have the same mean extracted sizes/chunk counts as weighted successful URLs; this assumption is unverified.",
                        "mean_success_text_bytes": float(total[METRICS.index("text_bytes")] / expected_successes),
                        "all_url_text_bytes": float(total[METRICS.index("text_bytes")] / expected_successes * urls),
                        "all_url_rag_jsonl_and_mapping_bytes": float((payload - total[METRICS.index("fetch_log_jsonl_bytes")] + successful_log_bytes) / expected_successes * urls),
                        "all_url_vector_float32_bytes": float(chunks / expected_successes * urls * settings["embedding_dimension"] * 4)}
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "population_urls": urls,
            "sample_urls": len(results), "sample_status_counts": dict(statuses),
            "weighted_success_fraction": expected_successes / urls if urls else 0,
            "urls_in_domains_without_successful_sample": unknown_urls,
            "metrics_under_current_fetch_policy": metrics,
            "storage": {"rag_jsonl_and_mapping_bytes": float(payload),
                        "dataset_bootstrap_95_percent_bytes": [float(x) for x in np.percentile(payload_bootstrap, [2.5, 97.5])],
                        "raw_gzip_bytes_if_enabled": float(raw),
                        "state_json_payload_bytes": float(state),
                        "working_disk_bytes_with_30_percent_reserve": float((payload + state + raw) * 1.3),
                        "reserve_note": "30% is a planning allowance, not a measured SQLite/index overhead bound."},
            "vectors_only": {"dimension": settings["embedding_dimension"],
                             "float32_bytes": float(chunks * settings["embedding_dimension"] * 4),
                             "float16_bytes": float(chunks * settings["embedding_dimension"] * 2),
                             "int8_bytes": float(chunks * settings["embedding_dimension"])},
            "full_coverage_text_only_scenarios": [
                {"assumed_mean_utf8_bytes_per_url": size, "total_bytes": urls * size}
                for size in (2048, 5120, 20480, 51200)],
            "hypothetical_full_coverage_at_measured_success_mean": hypothetical,
            "domains": sorted(domains, key=lambda r: -r["population_urls"]),
            "confidence_note": "Small per-domain samples are an initial sizing pilot. Increase sample_per_host before committing to full-corpus capacity, especially for large/heterogeneous domains.",
            "limitations": [
                "Estimation downloads a small stratified sample, not the full corpus.",
                "Failure/robots-denied samples contribute zero retained text: current-policy estimates are NOT full-corpus content sizes.",
                "No successful sample means content size for that domain is unknown; inspect the affected URL population.",
                "Bootstrap covers within-domain sample variability only; it does not cover crawler/extractor bias, future access changes or unseen long tails.",
                "Byte-limited/unsupported/dynamic pages are excluded; full collection may require other extractors and more storage.",
                "wire_bytes measures page/redirect response payload, excluding HTTP headers and robots.txt overhead; sampled robots bytes are reported separately.",
                "Vector estimates exclude Qdrant/HNSW, metadata, quantization scales and BM25 overhead.",
                "Character chunks are not model tokenizer tokens. No competition labels or official chunk IDs are synthesized."]}


def report_markdown(report):
    gib = lambda value: f"{value / 2**30:,.2f} GiB"
    lines = ["# Ước lượng dung lượng ViBioMIR cho RAG", "",
             f"Lấy mẫu {report['sample_urls']:,} / {report['population_urls']:,} URL, theo hostname.",
             f"Tỷ lệ trích xuất thành công có trọng số: {report['weighted_success_fraction']:.1%}.",
             f"Số URL thuộc domain chưa lấy được mẫu nội dung: {report['urls_in_domains_without_successful_sample']:,}.", "",
             "Các số dự phóng dưới đây áp dụng cho khả năng tải/trích xuất hiện tại; mẫu thất bại đóng góp 0 byte nội dung.",
             "**Chúng không phải dung lượng toàn bộ corpus nếu mọi URL đều truy cập được.**", "",
             "| Thành phần | Dự phóng | Bootstrap 95% (dao động mẫu) |", "|---|---:|---:|"]
    for name, stat in report["metrics_under_current_fetch_policy"].items():
        if name == "chunk_count":
            lines.append(f"| Số chunks | {stat['expected_total']:,.0f} | {stat['bootstrap_95_percent'][0]:,.0f}–{stat['bootstrap_95_percent'][1]:,.0f} |")
        else:
            lines.append(f"| {name} | {gib(stat['expected_total'])} | {gib(stat['bootstrap_95_percent'][0])}–{gib(stat['bootstrap_95_percent'][1])} |")
    lines += ["", f"Dataset JSONL + mapping + fetch log: {gib(report['storage']['rag_jsonl_and_mapping_bytes'])}.",
              f"Khoảng bootstrap cho dataset (gồm fetch log): {gib(report['storage']['dataset_bootstrap_95_percent_bytes'][0])}–{gib(report['storage']['dataset_bootstrap_95_percent_bytes'][1])}.",
              f"Dung lượng làm việc dự trù (gồm state + raw nếu bật + 30% dự phòng): {gib(report['storage']['working_disk_bytes_with_30_percent_reserve'])}.",
              f"Vector float32 riêng: {gib(report['vectors_only']['float32_bytes'])}; chưa tính index overhead.", "",
              "## Kịch bản toàn bộ URL có nội dung", "", "Chỉ là giả định về kích thước text trung bình, chưa gồm chunks/mappings/state/index:", "",
              "| Text UTF-8 trung bình / URL | Toàn bộ text |", "|---|---:|"]
    for item in report["full_coverage_text_only_scenarios"]:
        lines.append(f"| {item['assumed_mean_utf8_bytes_per_url'] / 1024:g} KiB | {gib(item['total_bytes'])} |")
    hypothetical = report["hypothetical_full_coverage_at_measured_success_mean"]
    if hypothetical:
        lines += ["", f"Nếu mọi URL có kích thước trung bình như các URL tải thành công: text {gib(hypothetical['all_url_text_bytes'])}; JSONL + mapping {gib(hypothetical['all_url_rag_jsonl_and_mapping_bytes'])}.",
                  "Giả định này chưa được xác minh cho các domain không lấy được nội dung."]
    lines += ["", "## Kết quả mẫu theo trạng thái", "", "```json", json.dumps(report["sample_status_counts"], indent=2), "```", "",
              "## Giới hạn", ""] + [f"- {note}" for note in report["limitations"]]
    return "\n".join(lines) + "\n"


def estimate(config, directory, *, retry_failed=False, plan_only=False):
    identity = source_identity(config)
    directory = Path(directory)
    with open_state(directory, identity) as db:
        plan_path = directory / "sample_plan.json"
        sampling = {"seed": config["rag"]["seed"], "per_host": config["rag"]["sample_per_host"]}
        if plan_path.exists():
            plan = read_json(plan_path)
            if plan["sampling"] != sampling:
                raise ValueError("Sample settings differ from saved plan; use another state directory")
        else:
            counts, rows = stratified_sample(catalogue_rows(Path(config["data"]["raw_dir"]) / "links_corpus.parquet"), **{
                "per_host": sampling["per_host"], "seed": sampling["seed"]})
            plan = {"sampling": sampling, "counts": counts, "rows": rows}
            write_json(plan_path, plan)
        LOGGER.info("Plan: %s sample URLs across %s hosts; population %s URLs", len(plan["rows"]), len(plan["counts"]), sum(plan["counts"].values()))
        if plan_only:
            return {"population_urls": sum(plan["counts"].values()), "sample_urls": len(plan["rows"]),
                    "sample_primary_response_cap_bytes_excluding_redirects_and_robots": len(plan["rows"]) * config["rag"]["max_response_bytes"],
                    "note": "Offline sampling plan only. Content sizes require fetching a small sample."}
        acquire(plan["rows"], db, config["rag"], directory, retry_failed=retry_failed)
        results = [json.loads(db.execute("SELECT result FROM pages WHERE id=?", (r["id"],)).fetchone()[0]) for r in plan["rows"]]
        report = capacity_report(plan["counts"], results, config["rag"])
        report["sample_robots_wire_bytes"] = sum(r.get("robots_wire_bytes", 0) for r in results)
        report["source"] = identity
        # Reports are replaceable views of the committed acquisition state.
        _replace_json(directory / "estimate.json", report)
        _replace_text(directory / "estimate.md", report_markdown(report))
        return report


def _replace_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _replace_json(path, value):
    _replace_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def crawl(config, directory, *, max_urls=None, all_urls=False, retry_failed=False):
    if (max_urls is None) == (not all_urls) or (max_urls is not None and max_urls < 1):
        raise ValueError("Choose a positive max_urls OR all_urls=True")
    identity = source_identity(config)
    rows = catalogue_rows(Path(config["data"]["raw_dir"]) / "links_corpus.parquet")
    if max_urls is not None:
        rows = itertools.islice(rows, max_urls)
    with open_state(directory, identity) as db:
        completed = acquire(rows, db, config["rag"], directory, retry_failed=retry_failed)
        return {"processed_this_invocation": completed,
                "state_status_counts": dict(db.execute("SELECT status, COUNT(*) FROM pages GROUP BY status"))}


def export(config, state_dir, output_dir):
    """Stream all acquired successes; preserve every doc ID even for duplicate text."""
    import pyarrow.parquet as pq
    identity = source_identity(config)
    output = Path(output_dir).resolve()
    state = Path(state_dir).resolve()
    raw = Path(config["data"]["raw_dir"]).resolve()
    if not (state / "state.sqlite").is_file():
        raise FileNotFoundError("Crawl state not found; run estimate or crawl first")
    if output.is_relative_to(raw) or raw.is_relative_to(output) or output.is_relative_to(state) or state.is_relative_to(output):
        raise ValueError("Export directory must not overlap raw/source or crawl state")
    if output.exists():
        raise FileExistsError("RAG export exists; choose a new output directory")
    with open_state(state, identity) as db:
        if not db.execute("SELECT 1 FROM pages WHERE status='ok' LIMIT 1").fetchone():
            raise ValueError("No successfully extracted documents to export; inspect crawl/estimate errors")
        output.mkdir(parents=True)
        mappings = output / "mappings"
        mappings.mkdir()
        count = chunks_count = 0
        with (output / "documents.jsonl").open("xb") as documents, (output / "chunks.jsonl").open("xb") as chunks, \
             (output / "fetch_log.jsonl").open("xb") as logs, \
             (mappings / "chunk_to_doc.json").open("w", encoding="utf-8") as parents, \
             (mappings / "internal_to_official_id.json").open("w", encoding="utf-8") as local_map:
            parents.write("{")
            local_map.write("{")
            for (serialized,) in db.execute("SELECT result FROM pages ORDER BY id"):
                result = json.loads(serialized)
                logs.write(encoded({k: v for k, v in result.items() if k != "document"}))
                if result["status"] != "ok":
                    continue
                document = result["document"]
                documents.write(encoded(document))
                count += 1
                for chunk in chunk_document(document, config["rag"]["chunk_chars"], config["rag"]["chunk_overlap"]):
                    chunks.write(encoded(chunk))
                    separator = ",\n" if chunks_count else "\n"
                    parents.write(separator + json.dumps(chunk["chunk_id"]) + ":" + json.dumps(chunk["doc_id"]))
                    # Identity map is for local pipeline compatibility, NOT a BTC chunk map.
                    local_map.write(separator + json.dumps(chunk["chunk_id"]) + ":" + json.dumps(chunk["chunk_id"]))
                    chunks_count += 1
            parents.write("\n}\n")
            local_map.write("\n}\n")
        query_file = pq.ParquetFile(raw / "query.parquet")
        _require_schema(query_file, {"id": "integer", "query": "string"})
        queries = [{"id": str(r["id"]), "text": r["query"], "official_id": r["id"]} for r in query_file.read().to_pylist()]
        write_json(output / "queries.json", queries)
        manifest = {"source": identity, "stage": "rag_corpus", "id_namespace": "official_document_and_local_rag_chunk",
                    "competition_submission_ready": False, "labels_available": False,
                    "counts": {"documents": count, "chunks": chunks_count, "queries": len(queries)},
                    "status_counts": dict(db.execute("SELECT status, COUNT(*) FROM pages GROUP BY status")),
                    "chunking": {"unit": "unicode_characters", "size": config["rag"]["chunk_chars"],
                                 "overlap": config["rag"]["chunk_overlap"]},
                    "limitations": "Crawled subset; rag:* IDs are local citation/retrieval IDs, not official competition chunks. No qrels. Duplicate text retains all official document IDs."}
        write_json(output / "dataset_manifest.json", manifest)
        rag_config = {"data": {"documents_path": "documents.jsonl", "chunks_path": "chunks.jsonl", "queries_path": "queries.json",
                               "mappings_dir": "mappings", "manifest_path": "dataset_manifest.json", "split": "rag_crawl",
                               "competition_submission_ready": False},
                      "retrieval": {"bm25": {"enabled": True, "index_path": "indexes/bm25.json", "top_k": 20,
                                              "k1": 1.5, "b": 0.75, "epsilon": 0.25, "text_key": "text",
                                              "lowercase": True, "tokenizer": "unicode_cjk"}, "dense": {"enabled": False}},
                      "fusion": {"method": "rrf", "rrf_k": 60, "top_k": 20}, "retrieval_cache": {"enabled": False},
                      "reranker": {"enabled": False}, "run_name": "rag_bm25", "output_dir": "runs"}
        with (output / "rag_config.yaml").open("x", encoding="utf-8") as handle:
            yaml.safe_dump(rag_config, handle, allow_unicode=True, sort_keys=False)
        # Written last: absence identifies interrupted/incomplete exports.
        write_json(output / "export_complete.json", {"documents": count, "chunks": chunks_count})
        return manifest
