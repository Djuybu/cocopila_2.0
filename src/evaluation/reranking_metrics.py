"""Before/after ranking comparison using the labels supplied by P1, not new qrels."""
from src.data.schema import unique_ids
from src.evaluation.candidate_recall import evaluate_candidate_recall
from src.evaluation.retrieval_metrics import mrr, ndcg, precision_at_k


def evaluate_reranked_candidates(records, labels, ks, internal_to_official=None):
    """Binary chunk relevance; doc recall uses the same K-chunk budget as P1."""
    report = evaluate_candidate_recall(records, labels, ks, internal_to_official)
    report["stage"] = "after_reranking"
    report["relevance"] = "binary_from_supplied_labels"
    mapping, truth = internal_to_official or {}, {row["id"]: row for row in labels}
    unique_ids(records, "id")
    for row in records:
        ordered = list(dict.fromkeys(mapping.get(c["chunk_id"], c["chunk_id"]) for c in row["candidates"]))
        relevant = truth[row["id"]]["relevant_chunks"]
        chunk_metrics = report["per_query"][row["id"]]["chunks"]
        for k in report["cutoffs"]:
            chunk_metrics.update({f"precision@{k}": precision_at_k(ordered, relevant, k),
                                  f"mrr@{k}": mrr(ordered[:k], relevant),
                                  f"ndcg@{k}": ndcg(ordered, relevant, k)})
    count = len(records)
    for metric in (f"{name}@{k}" for k in report["cutoffs"] for name in ("precision", "mrr", "ndcg")):
        report["macro"]["chunks"][metric] = sum(row["chunks"][metric] for row in report["per_query"].values()) / count if count else 0.0
    return report
