"""VRAM monitoring utility module for GPU memory management in Medical RAG."""

__author__ = "Dev C (MLOps Lead)"

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class VRAMMonitor:
    """Giám sát bộ nhớ VRAM GPU theo thời gian thực để ngăn ngừa tràn bộ nhớ (OOM)."""

    def __init__(self, budget_gb: float = 15.0) -> None:
        """Khởi tạo VRAM Monitor.

        Args:
            budget_gb: Giới hạn VRAM của GPU (ví dụ: T4=15GB, P100=16GB).
                Sử dụng pynvml để giám sát real-time.
        """
        self.budget_gb = budget_gb
        raise NotImplementedError("VRAMMonitor.__init__ chưa được triển khai.")

    def get_usage(self) -> dict[str, float]:
        """Trả về thông tin sử dụng VRAM hiện tại.

        Returns:
            dict[str, float]: Từ điển chứa các thông số:
                - 'total_gb': Tổng dung lượng VRAM (GB)
                - 'used_gb': Dung lượng VRAM đang sử dụng (GB)
                - 'free_gb': Dung lượng VRAM còn trống (GB)
                - 'utilization_pct': Tỷ lệ sử dụng VRAM (%)
        """
        raise NotImplementedError("get_usage chưa được triển khai.")

    def check_budget(self, required_gb: float) -> bool:
        """Kiểm tra xem còn đủ VRAM cho operation tiếp theo hay không.

        Args:
            required_gb: Dung lượng VRAM ước tính cần thiết (GB).

        Returns:
            bool: True nếu dung lượng còn trống (free_gb) >= required_gb, ngược lại False.
        """
        raise NotImplementedError("check_budget chưa được triển khai.")

    def log_usage(self, label: str = "") -> None:
        """In VRAM usage hiện tại ra console với label mô tả.

        Sử dụng rich formatting để hiển thị trực quan mức độ tiêu thụ tài nguyên.

        Args:
            label: Nhãn hoặc thông điệp mô tả ngữ cảnh kiểm tra (ví dụ: 'Sau khi nạp mô hình').
        """
        raise NotImplementedError("log_usage chưa được triển khai.")

    @contextmanager
    def context_monitor(self, label: str = "") -> Generator[None, None, None]:
        """Context manager để đo VRAM delta trước/sau một operation.

        Usage:
            with monitor.context_monitor('Loading BGE-M3'):
                model = load_model(...)

        Args:
            label: Nhãn tên tiến trình hoặc tác vụ đang đo lường.

        Yields:
            None: Trao quyền thực thi cho khối lệnh bên trong context.
        """
        raise NotImplementedError("context_monitor chưa được triển khai.")
        yield
