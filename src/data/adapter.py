"""Explicit source-field adapters; IDs are never synthesized."""


def adapt_rows(rows, fields):
    result = []
    for row in rows:
        adapted = dict(row)
        for target, source in fields.items():
            if source not in row:
                raise ValueError(f"Missing source field {source}")
            adapted[target] = row[source]
        result.append(adapted)
    return result


def build_mappings(chunks):
    chunk_to_doc, internal_to_official = {}, {}
    for chunk in chunks:
        official = chunk.get("official_chunk_id", chunk["chunk_id"])
        chunk_to_doc[official] = chunk["doc_id"]
        internal_to_official[chunk["chunk_id"]] = official
    return chunk_to_doc, internal_to_official


def official_candidates(candidates, internal_to_official, chunk_to_doc):
    """Collapse rechunks by official ID before selection; retain the best score."""
    result = {}
    for row in candidates:
        internal = row["chunk_id"]
        official = internal_to_official.get(internal, internal)
        if official not in chunk_to_doc or row["doc_id"] != chunk_to_doc[official]:
            raise ValueError(f"Unknown chunk or mismatched parent: {internal}")
        score = float(row["rerank_score"] if "rerank_score" in row else row["score"])
        old = result.get(official)
        old_score = float(old["rerank_score"] if "rerank_score" in old else old["score"]) if old else None
        if old is None or score > old_score:
            result[official] = {**row, "internal_chunk_id": internal, "chunk_id": official}
    return list(result.values())
