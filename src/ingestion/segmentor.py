"""Vietnamese text segmentation and preprocessing module for medical domain."""

__author__ = "Dev A (Data & Infrastructure Engineer)"

from __future__ import annotations

from typing import Any


class VietnameseSegmentor:
    """Bộ tách từ và tiền xử lý văn bản tiếng Việt cho tài liệu y khoa."""

    def __init__(self, backend: str = "pyvi") -> None:
        """Khởi tạo bộ tách từ tiếng Việt. Hỗ trợ backend 'pyvi' hoặc 'vncorenlp'. PyVi phù hợp cho tách từ nhanh, VnCoreNLP cho độ chính xác cao hơn với từ ghép y khoa.

        Args:
            backend: Backend sử dụng để tách từ ('pyvi' hoặc 'vncorenlp').
        """
        raise NotImplementedError

    def segment(self, text: str) -> str:
        """Tách từ tiếng Việt, bảo toàn ranh giới từ ghép chuyên ngành y khoa (vd: 'ung_thư biểu_mô tế_bào vảy', 'nhồi_máu cơ_tim cấp'). Trả về chuỗi đã tách từ với dấu gạch dưới nối các từ ghép.

        Args:
            text: Văn bản tiếng Việt cần tách từ.

        Returns:
            Chuỗi đã tách từ với dấu gạch dưới nối các từ ghép.
        """
        raise NotImplementedError

    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa bảng mã Unicode (NFC), loại bỏ ký tự ẩn, đồng nhất hóa dấu thanh tiếng Việt.

        Args:
            text: Văn bản thô cần chuẩn hóa bảng mã và dấu thanh.

        Returns:
            Văn bản đã được chuẩn hóa Unicode NFC và dấu thanh tiếng Việt.
        """
        raise NotImplementedError

    def expand_abbreviations(
        self,
        text: str,
        abbreviation_dict: dict[str, str] | None = None,
    ) -> str:
        """Mở rộng các viết tắt y khoa phổ biến (VD: 'HA' -> 'huyết áp', 'XN' -> 'xét nghiệm'). Sử dụng từ điển tùy chỉnh hoặc từ điển mặc định.

        Args:
            text: Văn bản cần mở rộng từ viết tắt.
            abbreviation_dict: Từ điển ánh xạ từ viết tắt sang từ đầy đủ. Nếu None, sử dụng từ điển mặc định.

        Returns:
            Văn bản với các từ viết tắt đã được thay thế thành dạng đầy đủ.
        """
        raise NotImplementedError

    def preprocess(self, text: str) -> str:
        """Pipeline đầy đủ: normalize_unicode -> expand_abbreviations -> segment. Đây là entry point chính cho tiền xử lý văn bản.

        Args:
            text: Văn bản thô đầu vào.

        Returns:
            Văn bản sau khi đã qua toàn bộ các bước tiền xử lý.
        """
        raise NotImplementedError
