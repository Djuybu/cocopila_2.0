import pytest
from src.evaluation.fbeta import classification_metrics, fbeta_score
from src.evaluation.evaluator import evaluate_submission
from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k, mrr, ndcg


def test_hand_calculated_f2():
    result = classification_metrics({"A", "B", "C"}, {"A", "B", "D"})
    for key in ("precision", "recall", "f1", "f2"):
        assert result[key] == pytest.approx(2 / 3)


def test_f2_weights_recall():
    # TP=1, |truth|=3, |prediction|=1 -> 5/(4*3+1)
    result = classification_metrics({"A", "B", "C"}, {"A"})
    assert result["precision"] == 1
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == 0.5
    assert result["f2"] == pytest.approx(5 / 13)


@pytest.mark.parametrize("truth,predicted", [(set(), set()), ({"A"}, set()), (set(), {"A"})])
def test_empty_sets(truth, predicted):
    assert fbeta_score(truth, predicted) == 0.0


def test_empty_policy_and_duplicates():
    assert fbeta_score([], [], zero_division=1) == 1
    assert fbeta_score(["A"], ["A", "A"]) == 1
    with pytest.raises(ValueError):
        fbeta_score(["A"], ["A"], beta=0)


def test_rank_metrics():
    assert precision_at_k(["A"], ["A", "B"], 2) == 0.5
    assert recall_at_k(["A", "A", "B"], ["A", "B"], 2) == 1
    assert mrr(["X", "A"], ["A"]) == 0.5
    assert ndcg(["A", "B"], ["A", "B"], 2) == 1
    with pytest.raises(ValueError):
        recall_at_k([], [], 0)


def test_macro_query_aggregation():
    truth = [{"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]},
             {"id": "q2", "relevant_docs": ["d2", "d3"], "relevant_chunks": ["c2", "c3"]}]
    prediction = [truth[0], {"id": "q2", "relevant_docs": [], "relevant_chunks": []}]
    result = evaluate_submission(prediction, truth)
    assert result["macro"]["relevant_chunks"]["f2"] == 0.5
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_submission(prediction[:1], truth)
