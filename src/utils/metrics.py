"""Evaluation metrics module for retrieval ranking evaluation in Medical RAG."""

__author__ = "Dev C (MLOps Lead)"

from typing import Any
import math


def hit_at_k(predicted: list[str], ground_truth: list[str], k: int = 10) -> float:
    """Tính Hit@K: 1.0 nếu có ít nhất 1 predicted item nằm trong ground_truth ở top K positions, 0.0 nếu không.

    Args:
        predicted: Danh sách ID các items được dự đoán (đã xếp hạng theo độ liên quan).
        ground_truth: Danh sách ID các items đúng theo nhãn chuẩn.
        k: Số lượng vị trí hàng đầu cần xét (mặc định là 10).

    Returns:
        float: Giá trị 1.0 nếu trúng ít nhất một nhãn, ngược lại là 0.0.
    """
    raise NotImplementedError("hit_at_k chưa được triển khai.")


def mrr(predicted: list[str], ground_truth: list[str]) -> float:
    """Tính Mean Reciprocal Rank: 1/rank của relevant item đầu tiên trong predicted list.

    Args:
        predicted: Danh sách ID các items được dự đoán theo thứ tự xếp hạng.
        ground_truth: Danh sách ID các items ground truth.

    Returns:
        float: Giá trị Reciprocal Rank (1/rank) của item đúng đầu tiên, 0.0 nếu không có item nào.
    """
    raise NotImplementedError("mrr chưa được triển khai.")


def ndcg(predicted: list[str], ground_truth: list[str], k: int = 10) -> float:
    """Tính Normalized Discounted Cumulative Gain tại vị trí K.

    Args:
        predicted: Danh sách ID các items được xếp hạng theo mô hình retrieval.
        ground_truth: Danh sách ID các items ground truth.
        k: Ngưỡng cutoff vị trí xếp hạng k (mặc định là 10).

    Returns:
        float: Điểm NDCG@K nằm trong đoạn [0.0, 1.0].
    """
    raise NotImplementedError("ndcg chưa được triển khai.")


def recall_at_k(predicted: list[str], ground_truth: list[str], k: int = 10) -> float:
    """Tính Recall@K: tỷ lệ ground_truth items được tìm thấy trong top K predictions.

    Args:
        predicted: Danh sách ID các items được dự đoán theo thứ bậc.
        ground_truth: Danh sách ID các items ground truth.
        k: Số lượng dự đoán top-k được xem xét (mặc định là 10).

    Returns:
        float: Tỷ lệ phần trăm ground truth items được tìm thấy trong top K [0.0, 1.0].
    """
    raise NotImplementedError("recall_at_k chưa được triển khai.")


def precision_at_k(predicted: list[str], ground_truth: list[str], k: int = 10) -> float:
    """Tính Precision@K: tỷ lệ items liên quan trong top K dự đoán.

    Args:
        predicted: Danh sách ID các items được dự đoán theo thứ tự xếp hạng.
        ground_truth: Danh sách ID các items ground truth.
        k: Số lượng vị trí hàng đầu cần xét (mặc định là 10).

    Returns:
        float: Giá trị Precision@K trong khoảng [0.0, 1.0].
    """
    raise NotImplementedError("precision_at_k chưa được triển khai.")


def evaluate_submission(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Đánh giá toàn bộ file submission.

    Tính metrics cho cả chunk-level và doc-level.
    Trả về dict tổng hợp với mean scores cho từng chỉ số đánh giá.

    Args:
        predictions: Danh sách các bản ghi dự đoán (mỗi bản ghi gồm id, relevant_docs, relevant_chunks).
        ground_truths: Danh sách các bản ghi ground truth đối chứng.
        metrics: Danh sách tên chỉ số cần tính (ví dụ: ["hit@10", "mrr", "ndcg@10", "recall@10"]).
            Nếu None, mặc định tính toán toàn bộ các chỉ số tiêu chuẩn.

    Returns:
        dict[str, Any]: Từ điển chứa kết quả chi tiết và điểm số trung bình (mean scores)
            cho cả doc-level và chunk-level.
    """
    raise NotImplementedError("evaluate_submission chưa được triển khai.")
