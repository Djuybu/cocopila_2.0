import copy
import pytest
from src.data.adapter import adapt_rows, build_mappings
from src.data.loader import prepare_data
from src.data.schema import validate_corpus
from src.data.split import validate_split_leakage
from src.utils.io import read_json


def test_data_and_official_mapping(dataset):
    config, docs, chunks, queries = dataset
    validate_corpus(docs, chunks, queries)
    mapping, internal = build_mappings(chunks)
    assert mapping["d1_c000"] == "d1"
    assert internal["internal1"] == "d1_c000"
    assert read_json(config["data"]["chunks_path"]) == chunks
    assert read_json(config["data"]["output_dir"] + "/chunks.json") == chunks


@pytest.mark.parametrize("kind,key", [("documents", "doc_id"), ("chunks", "chunk_id"), ("queries", "id")])
def test_duplicate_ids_rejected(dataset, kind, key):
    _, docs, chunks, queries = dataset
    data = {"documents": docs, "chunks": chunks, "queries": queries}
    data[kind] = data[kind] + [data[kind][0]]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_corpus(**data)


def test_parent_document_must_exist(dataset):
    _, docs, chunks, queries = dataset
    chunks = copy.deepcopy(chunks)
    chunks[0]["doc_id"] = "missing"
    with pytest.raises(ValueError, match="parent"):
        validate_corpus(docs, chunks, queries)


def test_splits_document_disjoint():
    validate_split_leakage({"train": [{"doc_id": "d1"}, {"doc_id": "d1"}], "val": [{"doc_id": "d2"}]})
    with pytest.raises(ValueError, match="leakage"):
        validate_split_leakage({"train": [{"doc_id": "d1"}], "val": [{"doc_id": "d1"}]})


def test_prepare_never_overwrites(dataset):
    with pytest.raises(FileExistsError):
        prepare_data(dataset[0])


def test_adapter_preserves_metadata_and_ids():
    result = adapt_rows([{"official": "c1", "language": "vi"}], {"chunk_id": "official"})
    assert result == [{"official": "c1", "chunk_id": "c1", "language": "vi"}]


def test_official_parent_conflict(dataset):
    _, docs, chunks, queries = dataset
    chunks = copy.deepcopy(chunks)
    chunks[1]["official_chunk_id"] = "d1_c000"
    with pytest.raises(ValueError, match="multiple documents"):
        validate_corpus(docs, chunks, queries)


def test_internal_namespace_collision(dataset):
    _, docs, chunks, queries = dataset
    chunks = copy.deepcopy(chunks)
    chunks[0]["chunk_id"] = "d2_c000"
    chunks[1]["chunk_id"] = "internal2"
    chunks[1]["official_chunk_id"] = "d2_c000"
    with pytest.raises(ValueError, match="collides"):
        validate_corpus(docs, chunks, queries)
