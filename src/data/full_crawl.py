"""Full-catalogue planning, resumable acquisition and bounded Parquet shard exports."""
from collections import Counter
import json
from pathlib import Path
from urllib.parse import urlsplit

from src.data.rag_dataset import source_identity
from src.data.web_corpus import acquire, chunk_document, open_state
from src.utils.io import read_json, write_json


def range_rows(path, start, stop):
    import pyarrow.parquet as pq
    if not 0 <= start < stop:
        raise ValueError("Invalid catalogue row range")
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        end = offset + len(batch)
        if end > start and offset < stop:
            lower, upper = max(0, start-offset), min(len(batch), stop-offset)
            for row in batch.slice(lower, upper-lower).to_pylist():
                yield {**row, "host": urlsplit(row["url"]).hostname.lower()}
        offset = end
        if offset >= stop:
            break


def plan_full_crawl(config, output, shard_size=5000):
    import pyarrow.parquet as pq
    if not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard_size must be positive")
    path = Path(config["data"]["raw_dir"]) / "links_corpus.parquet"
    identity = source_identity(config)
    total = pq.ParquetFile(path).metadata.num_rows
    shards, hosts = [], Counter()
    # One scan instead of re-reading the entire catalogue for each planned shard.
    bucket, start = Counter(), 0
    for index, row in enumerate(range_rows(path, 0, total)):
        bucket[row["host"]] += 1
        hosts[row["host"]] += 1
        if index+1-start == shard_size or index+1 == total:
            shards.append({"id": len(shards), "start": start, "stop": index+1,
                           "urls": index+1-start, "hosts": dict(bucket)})
            bucket, start = Counter(), index+1
    plan = {"source": identity, "population_urls": total, "shard_size": shard_size,
            "label_unit": "official_document_id_integer", "hosts": dict(hosts), "shards": shards}
    write_json(output, plan)
    return plan


class ParquetSink:
    def __init__(self, path, schema, batch_size=100):
        import pyarrow.parquet as pq
        self.writer, self.schema = pq.ParquetWriter(path, schema, compression="zstd"), schema
        self.rows, self.count, self.batch_size = [], 0, batch_size

    def add(self, row):
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.rows:
            import pyarrow as pa
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows = []

    def close(self):
        self.flush()
        self.writer.close()


def shard_identity(config, shard):
    identity = source_identity(config)
    identity["catalogue_range"] = {"start": shard["start"], "stop": shard["stop"], "id": shard["id"]}
    return identity


def export_shard(config, db, shard, output):
    import pyarrow as pa
    from src.data.download import sha256_file
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    doc_schema = pa.schema([(n, pa.int64() if n == "official_id" else pa.string()) for n in
                           ("doc_id", "official_id", "url", "title", "text", "metadata_json")])
    chunk_schema = pa.schema([(n, pa.int64() if n == "official_doc_id" else pa.string()) for n in
                             ("chunk_id", "doc_id", "official_doc_id", "title", "url", "text", "metadata_json")])
    source_schema = pa.schema([(n, pa.int64() if n in {"id", "http_status"} else pa.bool_() if n == "retryable" else pa.string()) for n in
                              ("id", "url", "host", "status", "http_status", "retryable", "error", "metadata_json")])
    suffix = f"{shard['id']:05d}.parquet"
    sinks = {"documents": ParquetSink(output/f"documents-{suffix}", doc_schema),
             "chunks": ParquetSink(output/f"chunks-{suffix}", chunk_schema, 500),
             "sources": ParquetSink(output/f"sources-{suffix}", source_schema)}
    statuses = Counter()
    retryable = 0
    try:
        for serialized, in db.execute("SELECT result FROM pages ORDER BY id"):
            result = json.loads(serialized)
            statuses[result["status"]] += 1
            retryable += int(result["status"] != "ok" and result.get("retryable", False))
            log = {k: v for k, v in result.items() if k != "document"}
            sinks["sources"].add({**log, "metadata_json": json.dumps(log, ensure_ascii=False)})
            if result["status"] != "ok":
                continue
            document = result["document"]
            sinks["documents"].add({**document, "metadata_json": json.dumps(document["metadata"], ensure_ascii=False)})
            for chunk in chunk_document(document, config["rag"]["chunk_chars"], config["rag"]["chunk_overlap"]):
                sinks["chunks"].add({**chunk, "official_doc_id": document["official_id"],
                                     "metadata_json": json.dumps(chunk["metadata"], ensure_ascii=False)})
    finally:
        for sink in sinks.values():
            sink.close()
    counts = {name: sink.count for name, sink in sinks.items()}
    expected = shard["stop"]-shard["start"]
    manifest = {"shard": shard, "source": shard_identity(config, shard), "counts": counts,
                "status_counts": dict(statuses), "acquisition_complete": counts["sources"] == expected,
                "content_complete": counts["documents"] == expected,
                "missing_content": counts["sources"]-counts["documents"],
                "retryable_failures": retryable,
                "pending_urls": expected-counts["sources"], "label_unit": "official_document_id_integer",
                "execution_platform": config.get("execution_platform", "local"),
                "tool_sha256": config.get("tool_sha256"),
                "files": {p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in output.glob("*.parquet")}}
    # Completion marker is written only after all writers close successfully.
    write_json(output/f"shard-{shard['id']:05d}.json", manifest)
    return manifest


def run_shard(config, shard, state_dir, output_dir, *, budget_seconds=21600, retry_failed=False, fetcher=None, retry_transient_only=False):
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    raw = Path(config["data"]["raw_dir"]) / "links_corpus.parquet"
    import pyarrow.parquet as pq
    if not 0 <= shard["start"] < shard["stop"] <= pq.ParquetFile(raw).metadata.num_rows:
        raise ValueError("Shard range exceeds source catalogue")
    identity = shard_identity(config, shard)
    with open_state(state_dir, identity) as db:
        acquire(range_rows(raw, shard["start"], shard["stop"]), db, config["rag"], state_dir,
                retry_failed=retry_failed, fetcher=fetcher, stop_after_seconds=budget_seconds,
                retry_transient_only=retry_transient_only)
        return export_shard(config, db, shard, output_dir)


def collection_manifest(plan, completed):
    """Coverage and content completeness are deliberately different assertions."""
    expected = {s["id"]: s for s in plan["shards"]}
    seen, counts, statuses = set(), Counter(), Counter()
    for manifest in completed:
        shard = manifest["shard"]
        if shard["id"] in seen or shard != expected.get(shard["id"]):
            raise ValueError("Duplicate or foreign shard manifest")
        source = dict(manifest["source"])
        source.pop("catalogue_range", None)
        if source != plan["source"]:
            raise ValueError("Shard source/settings mismatch")
        seen.add(shard["id"])
        counts.update(manifest["counts"])
        statuses.update(manifest["status_counts"])
    population = plan["population_urls"]
    covered = counts["sources"]
    acquisition_complete = len(seen) == len(expected) and covered == population
    content_complete = acquisition_complete and counts["documents"] == population
    return {"population_urls": population, "planned_shards": len(expected), "exported_shards": len(seen),
            "counts": dict(counts), "status_counts": dict(statuses), "acquisition_complete": acquisition_complete,
            "content_complete": content_complete, "pending_urls": population-covered,
            "missing_content": covered-counts["documents"], "label_unit": plan["label_unit"],
            "state": "complete" if content_complete else "needs_source_recovery" if acquisition_complete else "building",
            "source": plan["source"], "format": "vibiomir-rag-parquet-shards-v1",
            "files": {name: spec for manifest in completed for name, spec in manifest["files"].items()}}


def document_rankings(candidates, valid_document_ids):
    """Best passage score per official document; emit integer document labels."""
    valid = set(valid_document_ids)
    scores = {}
    for row in candidates:
        id = row.get("official_doc_id", row.get("doc_id"))
        if isinstance(id, bool) or not str(id).isdigit() or int(id) not in valid:
            raise ValueError("Candidate does not reference an official document")
        id = int(id)
        import math
        score = float(row["rerank_score"] if "rerank_score" in row else row["score"])
        if not math.isfinite(score):
            raise ValueError("Candidate score must be finite")
        scores[id] = max(score, scores.get(id, float("-inf")))
    return [{"document_id": id, "score": score} for id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
