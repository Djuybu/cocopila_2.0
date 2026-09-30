"""P3-09: lexicon/gazetteer-driven medical entity extraction (standard library only).

Extraction is data-driven: the seed lexicon in ``configs/graph/lexicon_seed.yaml``
is minimal and is not a medical ontology. Every mention keeps its ``chunk_id``
provenance; confidence depends on the match kind. Clinical relations are emitted
only when an explicit cue connects two mentions (never plain co-occurrence).
"""
from pathlib import Path
import re

import yaml

from src.utils.io import write_jsonl

ENTITY_TYPES = ("disease", "drug", "symptom", "treatment")
MATCH_CONFIDENCE = {"canonical": 1.0, "alias": 0.9, "abbreviation": 0.7}
MATCH_KINDS = ("canonical", "alias", "abbreviation")


def load_lexicon(path):
    """Load and validate a lexicon YAML file."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    entities = payload.get("entities") if isinstance(payload, dict) else None
    if not isinstance(entities, dict) or not entities:
        raise ValueError(f"Lexicon has no entities: {path}")
    for canonical_id, spec in entities.items():
        if not isinstance(spec, dict) or spec.get("type") not in ENTITY_TYPES:
            raise ValueError(f"Invalid entity {canonical_id!r}: {spec!r}")
        for key in ("aliases", "abbreviations"):
            if not isinstance(spec.get(key, []), list):
                raise ValueError(f"Entity {canonical_id!r} field {key} must be a list")
    relations = payload.get("relations", {}) or {}
    if not isinstance(relations, dict):
        raise ValueError(f"Lexicon relations must be a mapping: {path}")
    return {"version": payload.get("version", "unspecified"), "entities": entities, "relations": relations}


def _surface_regex(surface):
    escaped = re.escape(surface.strip()).replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE | re.UNICODE)


def build_matcher(lexicon):
    """Compile every canonical name/alias/abbreviation into a match entry."""
    entries = []
    for canonical_id, spec in lexicon["entities"].items():
        groups = (("canonical", [spec.get("name", canonical_id)]),
                  ("alias", spec.get("aliases", [])),
                  ("abbreviation", spec.get("abbreviations", [])))
        for kind, keys in groups:
            for key in keys:
                if not isinstance(key, str) or not key.strip():
                    continue
                entries.append({"regex": _surface_regex(key), "canonical_id": canonical_id,
                                "type": spec["type"], "name": spec.get("name", canonical_id),
                                "match": kind})
    if not entries:
        raise ValueError("Lexicon produced no matchable surfaces")
    return entries


def extract_entities(text, matcher):
    """Longest, highest-confidence, non-overlapping mentions with character offsets."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    spans = []
    for entry in matcher:
        for found in entry["regex"].finditer(text):
            spans.append({"start": found.start(), "end": found.end(), "surface": found.group(0),
                          "canonical_id": entry["canonical_id"], "type": entry["type"],
                          "name": entry["name"], "match": entry["match"],
                          "confidence": MATCH_CONFIDENCE[entry["match"]]})
    spans.sort(key=lambda span: (-span["confidence"], -(span["end"] - span["start"]),
                                 span["start"], span["canonical_id"]))
    chosen, occupied = [], []
    for span in spans:
        if any(span["start"] < end and start < span["end"] for start, end in occupied):
            continue
        occupied.append((span["start"], span["end"]))
        chosen.append(span)
    chosen.sort(key=lambda span: (span["start"], span["end"], span["canonical_id"]))
    return chosen


def extract_chunk_entities(chunks, lexicon, text_key="text"):
    """Extract entity records (with chunk_id provenance) from corpus chunks."""
    matcher = build_matcher(lexicon)
    records = []
    for chunk in chunks:
        chunk_id, doc_id = chunk.get("chunk_id"), chunk.get("doc_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Chunk needs a nonempty chunk_id: {chunk!r}")
        text = chunk.get(text_key)
        if not isinstance(text, str):
            raise ValueError(f"Chunk {chunk_id} text must be a string")
        for span in extract_entities(text, matcher):
            records.append({"chunk_id": chunk_id, "doc_id": doc_id,
                            "surface": span["surface"], "type": span["type"],
                            "canonical_candidate": span["canonical_id"],
                            "canonical_name": span["name"], "match": span["match"],
                            "confidence": span["confidence"],
                            "start": span["start"], "end": span["end"]})
    return records


def extract_relations(text, entities, lexicon, window=120):
    """Emit TREATS/HAS_SYMPTOM edges only when an explicit cue links two mentions.

    The two mentions are oriented by entity type (via the P3-08 schema), so the
    edge direction is correct regardless of word order; pairings that do not
    match the edge semantics are skipped.
    """
    from src.graph.schema import orient_edge

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    edges = []
    for edge_type, spec in lexicon.get("relations", {}).items():
        for cue in (spec or {}).get("cues", []):
            if not isinstance(cue, str) or not cue.strip():
                continue
            for found in _surface_regex(cue).finditer(text):
                left = [entity for entity in entities if entity["end"] <= found.start()]
                right = [entity for entity in entities if entity["start"] >= found.end()]
                if not left or not right:
                    continue
                nearest_left = max(left, key=lambda entity: entity["end"])
                nearest_right = min(right, key=lambda entity: entity["start"])
                if found.start() - nearest_left["end"] > window or nearest_right["start"] - found.end() > window:
                    continue
                oriented = orient_edge(edge_type, nearest_left, nearest_right)
                if oriented is None:
                    continue
                source, target = oriented
                start = min(source["start"], target["start"])
                end = max(source["end"], target["end"])
                edges.append({"edge_type": edge_type, "source": source["canonical_id"],
                              "target": target["canonical_id"], "evidence": text[start:end]})
    unique, seen = [], set()
    for edge in edges:
        key = (edge["edge_type"], edge["source"], edge["target"], edge["evidence"])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def extract_chunk_relations(chunks, lexicon, text_key="text"):
    """Relation edges with chunk provenance; validated by the P3-08 schema."""
    from src.graph.schema import validate_edge

    matcher = build_matcher(lexicon)
    edges = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get(text_key)
        if not isinstance(chunk_id, str) or not chunk_id or not isinstance(text, str):
            raise ValueError(f"Chunk needs a nonempty chunk_id and text: {chunk!r}")
        for edge in extract_relations(text, extract_entities(text, matcher), lexicon):
            candidate = {**edge, "provenance": {"chunk_id": chunk_id, "doc_id": chunk.get("doc_id")}}
            validate_edge(candidate)
            edges.append(candidate)
    return edges


def write_entities(path, records):
    """Write ``entities.jsonl`` (exclusive creation)."""
    write_jsonl(path, records)
    return records


def write_relations(path, edges):
    """Write ``relations.jsonl`` (exclusive creation)."""
    write_jsonl(path, edges)
    return edges
