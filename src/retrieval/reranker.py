"""Reranking module using Cross-Encoder models."""

__author__ = "Dev B (AI Pipeline & Model Specialist)"

from typing import Any


class CrossEncoderReranker:
    """Cross-Encoder reranker providing fine-grained semantic relevance scoring."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
    ) -> None:
        """Khởi tạo Cross-Encoder Reranker.

        Cross-Encoder nối trực tiếp query và passage thành chuỗi duy nhất, cho phép
        self-attention đan chéo giữa mọi token -> phân biệt chính xác false negatives
        và loại bỏ surface-level keyword matches sai bối cảnh y học. VRAM: ~1.5GB.

        Args:
            model_name: Tên hoặc đường dẫn model Cross-Encoder (mặc định BAAI/bge-reranker-v2-m3).
            device: Thiết bị tính toán ('cuda', 'cpu', ...).
        """
        raise NotImplementedError

    def load_model(self) -> None:
        """Tải model CrossEncoder vào GPU.

        Sử dụng sentence_transformers.CrossEncoder hoặc FlagEmbedding.
        """
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 20,
    ) -> list[dict]:
        """Tái xếp hạng Top candidates.

        Input: query + list of candidate dicts (phải có key 'text'). Tạo pairs
        [(query, candidate['text']), ...], chạy Cross-Encoder predict, sắp xếp theo score
        giảm dần, trả về Top-K. Output dicts có thêm key 'rerank_score'.

        Args:
            query: Chuỗi truy vấn.
            candidates: Danh sách các ứng viên (mỗi ứng viên phải có key 'text').
            top_k: Số lượng ứng viên điểm cao nhất trả về.

        Returns:
            Danh sách Top-K ứng viên kèm key 'rerank_score' được sắp xếp giảm dần.
        """
        raise NotImplementedError

    def batch_rerank(
        self,
        queries: list[str],
        candidates_list: list[list[dict]],
        top_k: int = 20,
    ) -> list[list[dict]]:
        """Batch reranking cho nhiều queries.

        Gộp tất cả pairs, predict 1 lần, rồi tách kết quả.

        Args:
            queries: Danh sách các câu truy vấn.
            candidates_list: Danh sách các danh sách ứng viên tương ứng với từng query.
            top_k: Số lượng ứng viên điểm cao nhất cho mỗi truy vấn.

        Returns:
            Danh sách kết quả reranked cho từng truy vấn tương ứng.
        """
        raise NotImplementedError
