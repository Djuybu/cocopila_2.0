"""Ranking metrics; Precision@K uses K as denominator even for short lists."""
import math


def _top(predicted, k):
    if k < 1:
        raise ValueError("k must be positive")
    return list(dict.fromkeys(predicted))[:k]


def recall_at_k(predicted, ground_truth, k=10):
    truth = set(ground_truth)
    hits = set(_top(predicted, k)) & truth
    return len(hits) / len(truth) if truth else 0.0


def precision_at_k(predicted, ground_truth, k=10):
    return len(set(_top(predicted, k)) & set(ground_truth)) / k


def hit_at_k(predicted, ground_truth, k=10):
    return float(bool(set(_top(predicted, k)) & set(ground_truth)))


def mrr(predicted, ground_truth):
    truth = set(ground_truth)
    return next((1 / rank for rank, item in enumerate(dict.fromkeys(predicted), 1) if item in truth), 0.0)


def ndcg(predicted, ground_truth, k=10):
    truth = set(ground_truth)
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(_top(predicted, k), 1) if item in truth)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(truth)) + 1))
    return dcg / ideal if ideal else 0.0
