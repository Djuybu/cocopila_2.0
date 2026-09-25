"""Macro per-query evaluation for both document and chunk branches."""
from src.data.schema import unique_ids
from src.evaluation.fbeta import classification_metrics
from src.evaluation.retrieval_metrics import hit_at_k, mrr, ndcg, precision_at_k, recall_at_k


def evaluate_submission(predictions, ground_truths, metrics=None, *, zero_division=0.0):
    prediction_ids = unique_ids(predictions, "id")
    truth_ids = unique_ids(ground_truths, "id")
    if prediction_ids != truth_ids:
        raise ValueError("Prediction and ground-truth query IDs must match exactly")
    metrics = metrics or ["precision", "recall", "f1", "f2", "recall@10", "precision@10"]
    predictions = {row["id"]: row for row in predictions}
    ranking = {"hit": hit_at_k, "recall": recall_at_k, "precision": precision_at_k, "ndcg": ndcg}
    by_query = {}
    for truth in ground_truths:
        prediction = predictions[truth["id"]]
        branches = {}
        for field in ("relevant_docs", "relevant_chunks"):
            scores = classification_metrics(truth[field], prediction[field], zero_division)
            values = {}
            for metric in metrics:
                if metric in scores:
                    values[metric] = scores[metric]
                elif metric == "mrr":
                    values[metric] = mrr(prediction[field], truth[field])
                elif "@" in metric and metric.split("@")[0] in ranking:
                    name, k = metric.split("@")
                    values[metric] = ranking[name](prediction[field], truth[field], int(k))
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            branches[field] = values
        by_query[truth["id"]] = branches
    macro = {
        field: {metric: sum(row[field][metric] for row in by_query.values()) / len(by_query) if by_query else 0.0
                for metric in metrics}
        for field in ("relevant_docs", "relevant_chunks")
    }
    return {"aggregation": "macro_by_query", "zero_division": zero_division,
            "query_count": len(by_query), "macro": macro, "per_query": by_query}
