"""Validate corpus identity before indexing or producing official mappings."""


def unique_ids(rows, key):
    ids = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing/non-string {key}: {row}")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {key}")
    return set(ids)


def validate_corpus(documents, chunks, queries=()):
    doc_ids = unique_ids(documents, "doc_id")
    unique_ids(chunks, "chunk_id")
    unique_ids(queries, "id")
    official_parents = {}
    for chunk in chunks:
        if chunk.get("doc_id") not in doc_ids:
            raise ValueError(f"Unknown parent document: {chunk}")
        if not isinstance(chunk.get("text"), str):
            raise ValueError("Chunk text must be a string")
        official = chunk.get("official_chunk_id", chunk["chunk_id"])
        if not isinstance(official, str) or not official:
            raise ValueError("Official chunk ID must be a nonempty string")
        parent = official_parents.setdefault(official, chunk["doc_id"])
        if parent != chunk["doc_id"]:
            raise ValueError(f"Official chunk {official} belongs to multiple documents")
    for chunk in chunks:
        if chunk["chunk_id"] in official_parents and chunk.get("official_chunk_id", chunk["chunk_id"]) != chunk["chunk_id"]:
            raise ValueError("Internal ID collides with an official chunk ID")
    for query in queries:
        if not isinstance(query.get("text"), str):
            raise ValueError("Query text must be a string")
