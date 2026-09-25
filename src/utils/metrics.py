"""Legacy evaluation imports."""
from src.evaluation.evaluator import evaluate_submission
from src.evaluation.retrieval_metrics import hit_at_k, mrr, ndcg, recall_at_k, precision_at_k

__all__ = ["evaluate_submission", "hit_at_k", "mrr", "ndcg", "recall_at_k", "precision_at_k"]
