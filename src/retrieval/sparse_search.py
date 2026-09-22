"""Sparse retrieval module using BM25 for lexical matching."""

__author__ = "Dev B (AI Pipeline & Model Specialist)"

from typing import Any


class SparseRetriever:
    """Sparse retriever utilizing BM25 algorithm for exact and keyword matching."""

    def __init__(self, index_path: str | None = None) -> None:
        """Khởi tạo Sparse Retriever dựa trên BM25.

        Nạp pre-built index từ file pickle nếu có, hoặc xây dựng index mới từ corpus.

        Args:
            index_path: Đường dẫn tới file pickle chứa chỉ mục BM25 đã được lưu trước đó.
        """
        raise NotImplementedError

    def build_index(
        self,
        corpus: list[dict],
        text_key: str = "segmented_text",
    ) -> None:
        """Xây dựng BM25 index từ corpus đã tách từ.

        Mỗi document trong corpus cần có key 'segmented_text' (đã qua PyVi tokenizer) và 'chunk_id'.

        Args:
            corpus: Danh sách các document chunk chứa text đã qua tiền xử lý.
            text_key: Tên trường khóa chứa nội dung văn bản đã tách từ trong dict.
        """
        raise NotImplementedError

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        """Tìm kiếm BM25 cho truy vấn đã tách từ.

        Trả về danh sách dict với keys: chunk_id, doc_id, text, score.
        BM25 hiệu quả cho đối khớp chính xác: số hiệu gen, liều lượng thuốc, định danh phân tử.

        Args:
            query: Câu truy vấn đã được tách từ (word-segmented).
            top_k: Số lượng kết quả có điểm cao nhất cần trả về.

        Returns:
            Danh sách kết quả tìm kiếm với chunk_id, doc_id, text, score.
        """
        raise NotImplementedError

    def tokenize_for_bm25(self, text: str) -> list[str]:
        """Tách text thành danh sách tokens phù hợp cho BM25.

        Loại bỏ stopwords, giữ lại thuật ngữ y khoa.

        Args:
            text: Chuỗi văn bản đầu vào cần tách token.

        Returns:
            Danh sách các token từ vựng phục vụ tính toán tần suất BM25.
        """
        raise NotImplementedError

    def save_index(self, path: str) -> None:
        """Lưu BM25 index ra file pickle.

        Args:
            path: Đường dẫn file lưu trữ chỉ mục (.pkl / .pickle).
        """
        raise NotImplementedError

    def load_index(self, path: str) -> None:
        """Nạp BM25 index từ file pickle.

        Args:
            path: Đường dẫn file chỉ mục đã được lưu trước đó.
        """
        raise NotImplementedError
