"""Triplet adaptation/deduplication with explicit prototype IDs and source references."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

from src.data.adapter import build_mappings
from src.data.schema import validate_corpus
from src.data.split import group_split
from src.utils.io import write_json, write_jsonl


def text_key(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prototype text must be a nonempty string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def prototype_id(kind, namespace, value):
    digest = hashlib.sha256(json.dumps([namespace, value], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    return f"prototype_{kind}_{digest}"


def _passage(value, default_meta=None):
    if isinstance(value, str):
        return value, dict(default_meta or {}), None
    if not isinstance(value, dict):
        raise ValueError("Passage must be a string or a text/metadata record")
    text = value.get("text")
    meta = {**(default_meta or {}), **value.get("meta", value.get("metadata", {}))}
    return text, meta, value.get("chunk_id", value.get("id"))


def adapt_triplets(rows, namespace="prototype", require_document_metadata=True):
    documents, chunks, queries = {}, {}, {}
    source_maps = {"queries": defaultdict(list), "documents": defaultdict(list), "chunks": defaultdict(list)}
    positives, negatives = defaultdict(set), defaultdict(set)
    texts = defaultdict(set)
    statistics = {"rows_before": len(rows), "positive_occurrences_before": 0,
                  "negative_occurrences_before": 0, "inferred_negative_documents": 0}
    pending = []

    def add_chunk(text, meta, ref, allow_inferred=False):
        key = text_key(text)
        identity = meta.get("article_id", meta.get("doc_id")) or meta.get("title")
        inferred = not identity
        if inferred and not allow_inferred and require_document_metadata:
            raise ValueError("Positive requires meta.article_id/doc_id or meta.title for safe document split")
        identity = ["article", str(meta.get("article_id", meta.get("doc_id")))] if meta.get("article_id", meta.get("doc_id")) else ["title", text_key(str(meta["title"]))] if meta.get("title") else ["inferred_text", key]
        doc_id = prototype_id("doc", namespace, identity)
        title = str(meta.get("title", ""))
        document = documents.setdefault(doc_id, {"doc_id": doc_id, "title": title,
                                                 "metadata": dict(meta), "id_namespace": "prototype"})
        if title and not document["title"]:
            document["title"] = title
        chunk_id = prototype_id("chunk", namespace, [doc_id, key])
        chunks.setdefault(chunk_id, {"chunk_id": chunk_id, "doc_id": doc_id, "text": text,
                                     "title": title, "metadata": dict(meta), "id_namespace": "prototype"})
        source_maps["documents"][doc_id].append({**ref, "source_document": identity, "inferred": inferred})
        source_maps["chunks"][chunk_id].append(ref)
        texts[key].add(chunk_id)
        if inferred:
            statistics["inferred_negative_documents"] += 1
        return chunk_id

    for index, row in enumerate(rows):
        anchor = row.get("anchor")
        anchor_id = anchor.get("id", anchor.get("query_id")) if isinstance(anchor, dict) else None
        if isinstance(anchor, dict):
            anchor = anchor.get("text")
        query_key = text_key(anchor)
        query_id = prototype_id("query", namespace, query_key)
        ref = {"row_index": index, "source_row_id": row.get("id", row.get("row_id")),
               "source_query_id": row.get("query_id", anchor_id)}
        queries.setdefault(query_id, {"id": query_id, "query_id": query_id,
                                     "text": anchor, "id_namespace": "prototype"})
        source_maps["queries"][query_id].append(ref)
        positive_values = row.get("positive")
        if not isinstance(positive_values, list):
            positive_values = [positive_values]
        for value in positive_values:
            text, meta, source_chunk_id = _passage(value, row.get("meta", row.get("metadata", {})))
            chunk_id = add_chunk(text, meta, {**ref, "role": "positive", "source_chunk_id": source_chunk_id})
            positives[query_id].add(chunk_id)
            statistics["positive_occurrences_before"] += 1
        negative_values = row.get("negatives", row.get("negative", []))
        if not isinstance(negative_values, list):
            negative_values = [negative_values]
        for value in negative_values:
            text, meta, source_chunk_id = _passage(value)
            pending.append((query_id, text, meta, {**ref, "role": "negative", "source_chunk_id": source_chunk_id}))
            statistics["negative_occurrences_before"] += 1

    for query_id, text, meta, ref in pending:
        key = text_key(text)
        if not meta and key in texts:
            matches = texts[key]
            for chunk_id in matches:
                source_maps["chunks"][chunk_id].append(ref)
        else:
            matches = {add_chunk(text, meta, ref, allow_inferred=True)}
        negatives[query_id].update(matches)
    false_negatives = 0
    for query_id in queries:
        false_negatives += len(negatives[query_id] & positives[query_id])
        negatives[query_id] -= positives[query_id]
        queries[query_id]["negative_chunk_ids"] = sorted(negatives[query_id])
    documents = sorted(documents.values(), key=lambda row: row["doc_id"])
    chunks = sorted(chunks.values(), key=lambda row: row["chunk_id"])
    queries = sorted(queries.values(), key=lambda row: row["id"])
    validate_corpus(documents, chunks, queries)
    parents = {row["chunk_id"]: row["doc_id"] for row in chunks}
    labels = [{"id": row["id"], "relevant_chunks": sorted(positives[row["id"]]),
               "relevant_docs": sorted({parents[c] for c in positives[row["id"]]})} for row in queries]
    statistics.update({"queries_after": len(queries), "documents_after": len(documents),
                       "chunks_after": len(chunks), "query_positive_pairs_after": sum(len(v) for v in positives.values()),
                       "query_negative_pairs_after": sum(len(v) for v in negatives.values()),
                       "duplicate_query_positive_pairs_removed": statistics["positive_occurrences_before"] - sum(len(v) for v in positives.values()),
                       "contradictory_negatives_removed": false_negatives})
    return {"documents": documents, "chunks": chunks, "queries": queries, "labels": labels,
            "source_mappings": {kind: dict(mapping) for kind, mapping in source_maps.items()},
            "dedup_report": statistics, "label_quality": "source_triplet"}


def write_prototype(bundle, cfg):
    output, mappings = Path(cfg["output_dir"]), Path(cfg["mappings_dir"])
    raw = Path(cfg["raw_dir"]).resolve() if cfg.get("raw_dir") else None
    if raw is not None and any(path.resolve().is_relative_to(raw) for path in (output, mappings)):
        raise ValueError("Prepared outputs must be outside raw")
    if output.exists() and any(output.iterdir()) or mappings.exists() and any(mappings.iterdir()):
        raise FileExistsError("Prototype output/mappings must be fresh directories")
    documents, chunks, queries = (bundle[key] for key in ("documents", "chunks", "queries"))
    split = group_split(documents, chunks, queries, bundle["labels"],
                        ratios=cfg["split_ratios"], seed=cfg.get("split_seed", cfg["seed"]))
    for row in documents:
        row["split"] = split["documents"][row["doc_id"]]
    for row in chunks:
        row["split"] = split["documents"][row["doc_id"]]
    filtered_negatives = 0
    chunk_parents = {row["chunk_id"]: row["doc_id"] for row in chunks}
    for query in queries:
        query["split"] = split["queries"][query["id"]]
        negative_ids = query.get("negative_chunk_ids", [])
        retained = [c for c in negative_ids if split["documents"][chunk_parents[c]] == query["split"]]
        filtered_negatives += len(negative_ids) - len(retained)
        if "negative_chunk_ids" in query:
            query["negative_chunk_ids"] = retained
    for kind in ("documents", "chunks", "queries", "labels"):
        write_json(output / f"{kind}.json", bundle[kind])
    write_jsonl(output / "chunk_corpus.jsonl", chunks)
    write_json(output / "dedup_report.json", {**bundle["dedup_report"], "cross_split_negatives_removed": filtered_negatives})
    write_json(output / "split_report.json", split)
    write_json(output / "dataset_manifest.json", {
        "id_namespace": "prototype", "label_quality": bundle["label_quality"],
        "seed": cfg["seed"], "split_seed": cfg.get("split_seed", cfg["seed"]),
        "source": bundle.get("source", "triplet input"),
        "preparation": cfg, "counts": {"documents": len(documents), "chunks": len(chunks), "queries": len(queries)},
    })
    chunk_to_doc, internal = build_mappings(chunks)
    write_json(mappings / "chunk_to_doc.json", chunk_to_doc)
    write_json(mappings / "internal_to_official_id.json", internal)
    write_json(mappings / "source_ids.json", bundle["source_mappings"])
    for name in cfg["split_ratios"]:
        query_ids = {key for key, value in split["queries"].items() if value == name}
        write_json(output / "splits" / name / "queries.json", [q for q in queries if q["id"] in query_ids])
        write_json(output / "splits" / name / "labels.json", [q for q in bundle["labels"] if q["id"] in query_ids])
        write_jsonl(output / "splits" / name / "chunks.jsonl", [c for c in chunks if c["split"] == name])
    return bundle


def prepare_triplets(config):
    from src.data.loader import load_records
    cfg = config["data"]
    bundle = adapt_triplets(load_records(cfg["source_path"]), cfg.get("namespace", "prototype"),
                            cfg.get("require_document_metadata", True))
    bundle["source"] = cfg["source_path"]
    return write_prototype(bundle, cfg)
