"""Script đóng gói tệp submission thành file ZIP chuẩn nộp bài thi."""

__author__ = "Dev C (MLOps Lead)"

import argparse
from pathlib import Path
import zipfile

from rich.console import Console

# Sử dụng validator từ utils module
from src.utils.validator import SubmissionValidator

console = Console()


def validate_json(file_path: str) -> bool:
    """Kiểm tra tính hợp lệ của tệp JSON dự đoán bằng SubmissionValidator.

    Args:
        file_path: Đường dẫn tới tệp JSON submission cần kiểm tra.

    Returns:
        bool: True nếu tệp JSON hợp lệ về cấu trúc schema và không chứa duplicate ID.
    """
    # TODO: Khởi tạo SubmissionValidator và gọi validate_file(file_path)
    raise NotImplementedError("validate_json chưa được triển khai.")


def pack_submission(json_path: str, output_zip_path: str) -> str:
    """Xác thực và nén tệp JSON trực tiếp vào file ZIP ở cấp gốc (không lồng thư mục con).

    Quy trình:
    1. Gọi validate_json để đảm bảo file JSON đạt chuẩn.
    2. Nén tệp JSON vào tệp ZIP với arcname là tên file gốc (flat structure, không có thư mục cha).
    3. Kiểm tra tính toàn vẹn (CRC / testzip) của tệp ZIP vừa tạo.
    4. Trả về đường dẫn tệp ZIP hoàn chỉnh.

    Args:
        json_path: Đường dẫn tới tệp JSON nguồn.
        output_zip_path: Đường dẫn tới tệp ZIP đầu ra cần tạo.

    Returns:
        str: Đường dẫn tuyệt đối của tệp ZIP đã đóng gói thành công.
    """
    # TODO:
    # 1. Gọi validate_json(json_path), nếu không hợp lệ thì raise ValueError
    # 2. Sử dụng zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED)
    # 3. zip_file.write(json_path, arcname=Path(json_path).name)
    # 4. Kiểm tra zip_file.testzip() is None
    raise NotImplementedError("pack_submission chưa được triển khai.")


def parse_arguments() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh phục vụ đóng gói submission.

    Returns:
        argparse.Namespace: Danh sách các tham số gồm input json và output zip.
    """
    parser = argparse.ArgumentParser(
        description="Đóng gói tệp JSON kết quả thành tệp ZIP chuẩn để nộp bài đánh giá."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Đường dẫn tới tệp JSON submission cần đóng gói.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Đường dẫn tới tệp ZIP đầu ra (ví dụ: data/submissions/submission.zip).",
    )
    return parser.parse_args()


def main() -> None:
    """Điểm khởi chạy chính của script đóng gói submission."""
    args = parse_arguments()
    console.print(f"[bold cyan]Bắt đầu quy trình đóng gói submission từ: {args.input}[/bold cyan]")

    # TODO:
    # 1. Kiểm tra sự tồn tại của args.input
    # 2. Gọi pack_submission(args.input, args.output)
    # 3. In thông báo thành công và kích thước tệp nén bằng rich console
    raise NotImplementedError("pack_submission.py main() chưa được triển khai.")


if __name__ == "__main__":
    main()
