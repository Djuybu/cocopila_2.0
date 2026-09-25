"""Set-based metrics. Empty denominators use explicit zero_division (default 0)."""


def precision_recall(y_true, y_pred, zero_division=0.0):
    truth, predicted = set(y_true), set(y_pred)
    hits = len(truth & predicted)
    return (hits / len(predicted) if predicted else float(zero_division),
            hits / len(truth) if truth else float(zero_division))


def fbeta_score(y_true, y_pred, beta=2.0, zero_division=0.0):
    if beta <= 0:
        raise ValueError("beta must be positive")
    truth, predicted = set(y_true), set(y_pred)
    denominator = beta * beta * len(truth) + len(predicted)
    return ((1 + beta * beta) * len(truth & predicted) / denominator
            if denominator else float(zero_division))


def classification_metrics(y_true, y_pred, zero_division=0.0):
    precision, recall = precision_recall(y_true, y_pred, zero_division)
    return {"precision": precision, "recall": recall,
            "f1": fbeta_score(y_true, y_pred, 1, zero_division),
            "f2": fbeta_score(y_true, y_pred, 2, zero_division)}
