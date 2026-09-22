"""Document and chunk aggregation module for retrieval results."""

__author__ = "Dev B (AI Pipeline & Model Specialist)"

from typing import Any


class DocumentAggregator:
    """Aggregates chunk-level retrieval scores to document-level rankings."""

    def __init__(self, method: str = "max_p", top_docs: int = 10) -> None:
        """Khởi tạo Document Aggregator.

        method: 'max_p' (Maximum Passage) hoặc 'top_k_mean'.
        top_docs: số lượng tài liệu trả về.

        Args:
            method: Chiến lược gom cụm điểm số ('max_p' hoặc 'top_k_mean').
            top_docs: Số lượng tài liệu liên quan hàng đầu cần trả về.
        """
        raise NotImplementedError

    def max_p_aggregation(self, chunks: list[dict]) -> list[dict]:
        """Tính điểm tài liệu bằng điểm chunk cao nhất thuộc tài liệu đó: Score(D) = max_{c in D} Score(c).

        Phù hợp đặc thù y khoa: tài liệu dài chỉ cần 1 đoạn quyết định là đủ để đánh giá liên quan.
        Trả về list dict với keys: doc_id, score, best_chunk_id.

        Args:
            chunks: Danh sách các chunk có chứa điểm số và doc_id.

        Returns:
            Danh sách dict kết quả gom cụm cấp tài liệu với doc_id, score, best_chunk_id.
        """
        raise NotImplementedError

    def top_k_mean_aggregation(self, chunks: list[dict], k: int = 3) -> list[dict]:
        """Tính điểm tài liệu bằng trung bình K chunk điểm cao nhất thuộc tài liệu.

        Phương pháp thay thế cho max_p khi cần đánh giá tổng thể hơn.

        Args:
            chunks: Danh sách các chunk có chứa điểm số và doc_id.
            k: Số lượng chunk điểm cao nhất lấy trung bình (mặc định 3).

        Returns:
            Danh sách dict kết quả gom cụm cấp tài liệu kèm điểm trung bình.
        """
        raise NotImplementedError

    def deduplicate(self, doc_ids: list[str]) -> list[str]:
        """Khử trùng lặp danh sách doc_id, giữ nguyên thứ tự xuất hiện đầu tiên.

        Đảm bảo tuân thủ quy định nộp bài: không có ID lặp.

        Args:
            doc_ids: Danh sách mã định danh tài liệu có thể trùng lặp.

        Returns:
            Danh sách mã định danh tài liệu duy nhất bảo toàn thứ tự ban đầu.
        """
        raise NotImplementedError

    def aggregate(
        self,
        reranked_chunks: list[dict],
    ) -> tuple[list[str], list[str]]:
        """Entry point chính.

        Trả về tuple (relevant_docs, relevant_chunks) đã deduplicate và sắp xếp.

        Args:
            reranked_chunks: Danh sách các chunk đã được tái xếp hạng.

        Returns:
            Tuple chứa (relevant_docs, relevant_chunks) đã được khử trùng lặp và sắp xếp theo độ liên quan.
        """
        raise NotImplementedError
