"""P3-10: entity normalization — collapse aliases/abbreviations to canonical entities.

The original surface form is kept, and each mention gains the canonical ID, name
and type. The node index reduces duplicate graph nodes.
"""
from src.utils.io import write_jsonl


def normalize_entities(records, lexicon):
    """Attach canonical ID/name/type to every extracted mention."""
    entities = lexicon["entities"]
    output = []
    for record in records:
        canonical_id = record.get("canonical_candidate")
        spec = entities.get(canonical_id)
        if not isinstance(spec, dict):
            raise ValueError(f"Unknown canonical entity: {canonical_id!r}")
        output.append({**record, "canonical_id": canonical_id,
                       "canonical_name": spec.get("name", canonical_id),
                       "canonical_type": spec["type"]})
    return output


def summarize_normalization(records, normalized):
    """Quantify how many duplicate surface forms collapse into canonical entities."""
    surfaces = {(row["chunk_id"], row["surface"].casefold()) for row in records}
    canonical_ids = {row["canonical_id"] for row in normalized}
    by_type = {}
    for row in normalized:
        by_type[row["canonical_type"]] = by_type.get(row["canonical_type"], 0) + 1
    return {
        "mention_count": len(records),
        "normalized_mention_count": len(normalized),
        "unique_surfaces": len(surfaces),
        "unique_canonical_entities": len(canonical_ids),
        "duplicate_surface_reduction": len(surfaces) - len(canonical_ids),
        "mentions_by_canonical_type": by_type,
    }


def build_node_index(normalized):
    """canonical_id -> {name, type, chunk_ids, surfaces} for the P3-11 graph build."""
    index = {}
    for row in normalized:
        entry = index.setdefault(row["canonical_id"], {
            "canonical_id": row["canonical_id"], "canonical_name": row["canonical_name"],
            "canonical_type": row["canonical_type"], "chunk_ids": [], "surfaces": []})
        if row["chunk_id"] not in entry["chunk_ids"]:
            entry["chunk_ids"].append(row["chunk_id"])
        if row["surface"] not in entry["surfaces"]:
            entry["surfaces"].append(row["surface"])
    return index


def write_normalized_entities(path, normalized):
    """Write ``normalized_entities.jsonl`` (exclusive creation)."""
    write_jsonl(path, normalized)
    return normalized
