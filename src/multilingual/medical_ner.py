"""Trích xuất thực thể y khoa (Medical NER) và mở rộng truy vấn dựa trên ontology UMLS."""

from __future__ import annotations

__author__ = "Dev B (AI Pipeline & Model Specialist)"

from typing import Any, Dict, List, Optional

from ..generation.llm_loader import LLMManager


class MedicalNER:
    """Nhận diện thực thể đặt tên y sinh và chuẩn hóa khái niệm với UMLS."""

    def __init__(self, llm_manager: Optional[LLMManager] = None) -> None:
        """Khởi tạo Medical Named Entity Recognition.

        Sử dụng LLM (Qwen2.5) hoặc rule-based patterns để nhận diện thực thể y khoa
        từ truy vấn tiếng Việt.

        Args:
            llm_manager: Đối tượng LLMManager tùy chọn dùng cho việc trích xuất thực thể
                bằng prompt chuyên biệt.
        """
        self.llm_manager = llm_manager
        raise NotImplementedError("MedicalNER.__init__ chưa được triển khai.")

    def extract_entities(self, query: str) -> list[dict[str, Any]]:
        """Trích xuất thực thể y khoa.

        Các loại thực thể:
        - Tên bệnh (DISEASE)
        - Thuốc (DRUG)
        - Triệu chứng (SYMPTOM)
        - Xét nghiệm (TEST)
        - Chỉ số (MEASUREMENT)
        - Cơ quan (ANATOMY)

        Args:
            query: Câu truy vấn hoặc đoạn văn bản y khoa bằng tiếng Việt.

        Returns:
            list[dict]: Danh sách thực thể trích xuất được với cấu trúc:
                {'text': str, 'type': str, 'start': int, 'end': int}.
        """
        raise NotImplementedError("MedicalNER.extract_entities chưa được triển khai.")

    def map_to_umls(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ánh xạ thực thể sang mã UMLS (Unified Medical Language System) CUI.

        Bổ sung synonyms đa ngôn ngữ từ UMLS ontology. Trả về entities enriched
        với key 'umls_cui' và 'synonyms'.

        Args:
            entities: Danh sách các thực thể từ hàm extract_entities.

        Returns:
            list[dict]: Danh sách thực thể được bổ sung thông tin khái niệm:
                - 'umls_cui': Mã định danh khái niệm y sinh UMLS (ví dụ: 'C0006826').
                - 'synonyms': Danh sách các tên gọi đồng nghĩa bằng tiếng Việt, tiếng Anh, tiếng Trung.
        """
        raise NotImplementedError("MedicalNER.map_to_umls chưa được triển khai.")

    def expand_query(
        self,
        query: str,
        entities: list[dict[str, Any]] | None = None,
    ) -> str:
        """Mở rộng truy vấn y khoa: bổ sung biến thể từ vựng đồng nghĩa, thuật ngữ viết đầy đủ, và tên quốc tế.

        Ví dụ: 'ung thư phổi' -> 'ung thư phổi lung cancer 肺癌 NSCLC non-small cell'.
        Giúp mạng lưới tìm kiếm bắt nhiều khía cạnh của văn bản gốc.

        Args:
            query: Câu truy vấn gốc của người dùng.
            entities: Danh sách thực thể y khoa đã trích xuất và ánh xạ UMLS (tùy chọn).
                Nếu không cung cấp, hàm sẽ tự động gọi trích xuất và ánh xạ.

        Returns:
            str: Chuỗi truy vấn mở rộng kết hợp các từ khóa đồng nghĩa và thuật ngữ chuyên ngành.
        """
        raise NotImplementedError("MedicalNER.expand_query chưa được triển khai.")
