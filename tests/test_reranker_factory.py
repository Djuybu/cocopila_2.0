"""Unit tests for reranker adapters and factory."""
import pytest
from src.reranking.base import BaseReranker
from src.reranking.bge import CrossEncoderReranker
from src.reranking.factory import create_reranker
from src.reranking.qwen_reranker import QwenReranker


class DummyCrossEncoder:
    """Mock CrossEncoder that records predict calls and returns deterministic scores."""

    def __init__(self, scores=None):
        self.scores = scores
        self.last_pairs = None

    def predict(self, pairs, batch_size=32):
        self.last_pairs = pairs
        if self.scores is not None:
            return self.scores[: len(pairs)]
        return [float(len(doc)) for _, doc in pairs]


def test_bge_reranker_with_instruction():
    mock_model = DummyCrossEncoder()
    reranker = CrossEncoderReranker(
        model=mock_model,
        instruction="Medical instruct:",
    )
    candidates = [
        {"chunk_id": "c1", "text": "aspirin dosage", "doc_id": "d1"},
        {"chunk_id": "c2", "text": "paracetamol contraindications", "doc_id": "d1"},
    ]
    results = reranker.rerank("headache", candidates)

    assert len(results) == 2
    assert mock_model.last_pairs[0][0] == "Medical instruct: headache"
    assert "rerank_score" in results[0]
    # Check descending order
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]


def test_qwen_reranker_basic_and_instruction():
    mock_model = DummyCrossEncoder()
    custom_instruction = "Evaluate medical relevance:"
    reranker = QwenReranker(
        model=mock_model,
        batch_size=8,
        instruction=custom_instruction,
    )
    candidates = [
        {"chunk_id": "c1", "text": "short", "doc_id": "d1"},
        {"chunk_id": "c2", "text": "much longer medical text passage", "doc_id": "d1"},
    ]
    results = reranker.rerank("query text", candidates, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c2"
    assert mock_model.last_pairs[0][0] == "Evaluate medical relevance: query text"


def test_qwen_reranker_empty_and_validation():
    mock_model = DummyCrossEncoder()
    reranker = QwenReranker(model=mock_model)

    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", [{"chunk_id": "c1", "text": "t", "doc_id": "d1"}], top_k=0) == []

    with pytest.raises(ValueError, match="top_k must be nonnegative"):
        reranker.rerank("query", [{"chunk_id": "c1", "text": "t", "doc_id": "d1"}], top_k=-1)


def test_qwen_reranker_batch():
    mock_model = DummyCrossEncoder()
    reranker = QwenReranker(model=mock_model)
    queries = ["q1", "q2"]
    cand_lists = [
        [{"chunk_id": "c1", "text": "t1", "doc_id": "d1"}],
        [{"chunk_id": "c2", "text": "t2", "doc_id": "d2"}],
    ]
    batch_results = reranker.batch_rerank(queries, cand_lists)
    assert len(batch_results) == 2
    assert batch_results[0][0]["chunk_id"] == "c1"
    assert batch_results[1][0]["chunk_id"] == "c2"


def test_reranker_factory():
    # Test BGE config
    bge_config = {
        "reranker": {
            "type": "bge",
            "model": "BAAI/bge-reranker-v2-m3",
            "device": "cpu",
            "batch_size": 32,
            "instruction": "Test instruction",
        }
    }
    bge_instance = create_reranker(bge_config)
    assert isinstance(bge_instance, CrossEncoderReranker)
    assert bge_instance.model_name == "BAAI/bge-reranker-v2-m3"
    assert bge_instance.device == "cpu"
    assert bge_instance.instruction == "Test instruction"

    # Test Qwen config
    qwen_config = {
        "reranker": {
            "type": "qwen",
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "device": "cuda:1",
            "batch_size": 16,
            "instruction": "Given a medical query...",
        }
    }
    qwen_instance = create_reranker(qwen_config)
    assert isinstance(qwen_instance, QwenReranker)
    assert qwen_instance.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert qwen_instance.device == "cuda:1"
    assert qwen_instance.batch_size == 16

    # Test invalid type
    with pytest.raises(ValueError, match="Unsupported reranker type"):
        create_reranker({"type": "unknown_reranker"})
