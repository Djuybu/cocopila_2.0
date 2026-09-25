import pytest
from src.utils.io import write_json
from src.data.loader import prepare_data


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    documents = [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}]
    chunks = [
        {"chunk_id": "internal1", "official_chunk_id": "d1_c000", "doc_id": "d1", "text": "aspirin treatment"},
        {"chunk_id": "d2_c000", "doc_id": "d2", "text": "insulin diabetes"},
        {"chunk_id": "d3_c000", "doc_id": "d3", "text": "cardiac pressure"},
    ]
    queries = [{"id": "q1", "text": "aspirin"}, {"id": "q2", "text": "insulin"}]
    paths = {}
    for kind, rows in [("documents", documents), ("chunks", chunks), ("queries", queries)]:
        path = raw / f"{kind}.json"
        write_json(path, rows)
        paths[f"{kind}_path"] = str(path)
    preparation = {"data": {**paths, "raw_dir": str(raw),
                            "output_dir": str(tmp_path / "processed"),
                            "mappings_dir": str(tmp_path / "mappings")}}
    prepare_data(preparation)
    return preparation, documents, chunks, queries


@pytest.fixture
def experiment(dataset, tmp_path):
    preparation, _, _, _ = dataset
    data = preparation["data"]
    return {
        "run_name": "test_run", "output_dir": str(tmp_path / "outputs"),
        "data": {**{f"{kind}_path": str(tmp_path / "processed" / f"{kind}.json")
                   for kind in ("documents", "chunks", "queries")},
                 "mappings_dir": data["mappings_dir"], "split": "test"},
        "retrieval": {"bm25": {"enabled": True, "index_path": str(tmp_path / "indexes" / "bm25.json"),
                               "k1": 1.5, "b": 0.75, "epsilon": 0.25,
                               "lowercase": False, "text_key": "text", "top_k": 3}},
        "fusion": {"method": "rrf", "top_k": 3, "rrf_k": 60},
        "reranker": {"enabled": False},
        "selector": {"chunk_threshold": None, "chunk_fallback": 1, "chunk_max": 1,
                     "doc_threshold": None, "doc_fallback": 1, "doc_max": 1,
                     "doc_aggregation": "max"},
        "evaluation": {"candidate_k": 3, "zero_division": 0.0},
        "submission": {"output_dir": str(tmp_path / "submissions")},
    }
