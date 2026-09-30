"""P3-08/09/10: graph schema, entity extraction and normalization tests."""
import pytest

from src.graph.ner import (
    build_matcher,
    extract_chunk_entities,
    extract_chunk_relations,
    extract_entities,
    load_lexicon,
    write_entities,
)
from src.graph.normalize import (
    build_node_index,
    normalize_entities,
    summarize_normalization,
    write_normalized_entities,
)
from src.graph.schema import EDGE_TYPES, NODE_TYPES, entity_node_type, validate_edge, validate_node


def lexicon():
    return {
        "version": "test",
        "entities": {
            "nhoi_mau_co_tim": {"type": "disease", "name": "nhồi máu cơ tim",
                                "aliases": ["nhồi máu cơ tim", "myocardial infarction"],
                                "abbreviations": ["NMCT"]},
            "statin": {"type": "treatment", "name": "statin", "aliases": ["statin"], "abbreviations": []},
            "dau_nguc": {"type": "symptom", "name": "đau ngực", "aliases": ["đau ngực"], "abbreviations": []},
        },
        "relations": {"TREATS": {"cues": ["điều trị"]}, "HAS_SYMPTOM": {"cues": ["gây"]}},
    }


def test_node_and_edge_validation():
    validate_node({"node_id": "d1", "node_type": "Document"})
    validate_node({"node_id": "nhoi_mau_co_tim", "node_type": "Disease", "properties": {}})
    with pytest.raises(ValueError):
        validate_node({"node_id": "", "node_type": "Document"})
    with pytest.raises(ValueError):
        validate_node({"node_id": "x", "node_type": "Unknown"})
    validate_edge({"source": "c1", "target": "nhoi_mau_co_tim", "edge_type": "MENTIONS",
                   "provenance": {"chunk_id": "c1"}})
    with pytest.raises(ValueError):
        validate_edge({"source": "c1", "target": "n", "edge_type": "MENTIONS"})
    with pytest.raises(ValueError):
        validate_edge({"source": "statin", "target": "nhoi_mau_co_tim", "edge_type": "TREATS",
                       "provenance": {"chunk_id": "c1"}})
    with pytest.raises(ValueError):
        validate_edge({"source": "x", "target": "x", "edge_type": "HAS_CHUNK"})
    assert NODE_TYPES[0] == "Document" and "TREATS" in EDGE_TYPES
    assert entity_node_type("disease") == "Disease"
    with pytest.raises(ValueError):
        entity_node_type("bogus")


def test_extract_entities_marks_match_kind_and_confidence():
    spans = extract_entities("Bệnh nhân NMCT nhập viện.", build_matcher(lexicon()))
    assert [span["canonical_id"] for span in spans] == ["nhoi_mau_co_tim"]
    assert spans[0]["match"] == "abbreviation" and spans[0]["confidence"] == 0.7
    canonical = extract_entities("nhồi máu cơ tim cấp", build_matcher(lexicon()))
    assert canonical[0]["match"] == "canonical" and canonical[0]["confidence"] == 1.0


def test_extract_chunk_entities_keeps_provenance():
    records = extract_chunk_entities([{"chunk_id": "c1", "doc_id": "d1", "text": "NMCT và đau ngực"}], lexicon())
    assert {record["canonical_candidate"] for record in records} == {"nhoi_mau_co_tim", "dau_nguc"}
    assert all(record["chunk_id"] == "c1" and record["doc_id"] == "d1" for record in records)


def test_relations_require_an_explicit_cue_and_correct_direction():
    with_cue = extract_chunk_relations(
        [{"chunk_id": "c1", "doc_id": "d1", "text": "statin điều trị nhồi máu cơ tim"}], lexicon())
    assert len(with_cue) == 1 and with_cue[0]["edge_type"] == "TREATS"
    assert with_cue[0]["source"] == "statin" and with_cue[0]["target"] == "nhoi_mau_co_tim"
    assert with_cue[0]["provenance"]["chunk_id"] == "c1" and with_cue[0]["evidence"]
    # Vietnamese word order "X được điều trị bằng Y" must still orient drug -> disease.
    reversed_order = extract_chunk_relations(
        [{"chunk_id": "c1", "doc_id": "d1", "text": "NMCT được điều trị bằng statin"}], lexicon())
    assert reversed_order[0]["source"] == "statin" and reversed_order[0]["target"] == "nhoi_mau_co_tim"
    cooccurrence = extract_chunk_relations(
        [{"chunk_id": "c2", "doc_id": "d1", "text": "statin và nhồi máu cơ tim"}], lexicon())
    assert cooccurrence == []
    # A cue linking entity types that do not fit the edge semantics creates nothing.
    unsupported = extract_chunk_relations(
        [{"chunk_id": "c3", "doc_id": "d1", "text": "đau ngực điều trị statin"}], lexicon())
    assert unsupported == []



def test_normalization_collapses_aliases_and_keeps_surface():
    records = [
        {"chunk_id": "c1", "doc_id": "d1", "surface": "NMCT", "type": "disease",
         "canonical_candidate": "nhoi_mau_co_tim", "canonical_name": "nhồi máu cơ tim",
         "match": "abbreviation", "confidence": 0.7, "start": 0, "end": 4},
        {"chunk_id": "c2", "doc_id": "d2", "surface": "nhồi máu cơ tim", "type": "disease",
         "canonical_candidate": "nhoi_mau_co_tim", "canonical_name": "nhồi máu cơ tim",
         "match": "canonical", "confidence": 1.0, "start": 0, "end": 15},
    ]
    normalized = normalize_entities(records, lexicon())
    assert all(record["canonical_id"] == "nhoi_mau_co_tim" for record in normalized)
    assert {record["surface"] for record in normalized} == {"NMCT", "nhồi máu cơ tim"}
    report = summarize_normalization(records, normalized)
    assert report["unique_surfaces"] == 2 and report["unique_canonical_entities"] == 1
    index = build_node_index(normalized)
    assert index["nhoi_mau_co_tim"]["chunk_ids"] == ["c1", "c2"]
    assert index["nhoi_mau_co_tim"]["canonical_type"] == "disease"


def test_normalization_rejects_unknown_canonical_entity():
    with pytest.raises(ValueError):
        normalize_entities([{"canonical_candidate": "nope"}], lexicon())


def test_write_helpers_are_exclusive(tmp_path):
    records = extract_chunk_entities([{"chunk_id": "c1", "doc_id": "d1", "text": "NMCT"}], lexicon())
    path = tmp_path / "entities.jsonl"
    write_entities(path, records)
    with pytest.raises(FileExistsError):
        write_entities(path, records)
    normalized = normalize_entities(records, lexicon())
    normalized_path = tmp_path / "normalized_entities.jsonl"
    write_normalized_entities(normalized_path, normalized)
    with pytest.raises(FileExistsError):
        write_normalized_entities(normalized_path, normalized)


def test_load_lexicon_validates_entity_types(tmp_path):
    good = tmp_path / "lex.yaml"
    good.write_text("entities:\n  e1:\n    type: drug\n    name: e1\n    aliases: []\n    abbreviations: []\n",
                    encoding="utf-8")
    assert load_lexicon(good)["entities"]["e1"]["type"] == "drug"
    bad = tmp_path / "bad.yaml"
    bad.write_text("entities:\n  e1:\n    type: bogus\n    aliases: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lexicon(bad)
