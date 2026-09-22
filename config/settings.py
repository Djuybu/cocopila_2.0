"""Settings module for Medical RAG System."""

__author__ = "Dev C (MLOps Lead)"

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Union

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


_DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Lớp cấu hình trung tâm cho toàn bộ hệ thống Medical RAG.

    Chứa các tham số cấu hình đường dẫn dữ liệu, mô hình ngôn ngữ lớn (LLM),
    mô hình embedding đa ngữ, mô hình reranker, tham số truy xuất lai (hybrid search)
    và ngân sách tài nguyên phần cứng.
    """

    # --- Đường dẫn hệ thống ---
    BASE_DIR: Path = field(
        default_factory=lambda: _DEFAULT_BASE_DIR,
        metadata={"description": "Thư mục gốc của dự án."},
    )
    DATA_RAW_DIR: Path = field(
        default_factory=lambda: _DEFAULT_BASE_DIR / "data" / "raw",
        metadata={"description": "Thư mục lưu trữ dữ liệu y khoa thô."},
    )
    DATA_PROCESSED_DIR: Path = field(
        default_factory=lambda: _DEFAULT_BASE_DIR / "data" / "processed",
        metadata={"description": "Thư mục lưu trữ dữ liệu sau khi tiền xử lý và chunking."},
    )
    QDRANT_DB_PATH: Path = field(
        default_factory=lambda: _DEFAULT_BASE_DIR / "data" / "qdrant_db",
        metadata={"description": "Thư mục lưu trữ cơ sở dữ liệu vector Qdrant cục bộ."},
    )
    BM25_INDEX_PATH: Path = field(
        default_factory=lambda: _DEFAULT_BASE_DIR / "data" / "bm25_index",
        metadata={"description": "Thư mục lưu trữ chỉ mục tìm kiếm từ khóa BM25."},
    )

    # --- Cấu hình LLM ---
    LLM_MODEL_REPO: str = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    LLM_MODEL_FILE: str = "qwen2.5-7b-instruct-q4_k_m.gguf"
    LLM_N_GPU_LAYERS: int = -1  # Offload toàn bộ các layer lên GPU (-1)
    LLM_N_CTX: int = 4096  # Kích thước context window tối đa của LLM

    # --- Cấu hình Embedding & Vector Store ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_MAX_SEQ_LEN: int = 8192
    QDRANT_COLLECTION_NAME: str = "medical_chunks"

    # --- Cấu hình Reranker ---
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # --- Tham số Retrieval & Hybrid Fusion ---
    TOP_K_DENSE: int = 100  # Số lượng kết quả lấy từ dense vector search
    TOP_K_SPARSE: int = 100  # Số lượng kết quả lấy từ sparse BM25 search
    TOP_K_CANDIDATES: int = 200  # Số lượng ứng viên sau khi hợp nhất (fusion) đưa vào reranker
    RERANK_TOP_K: int = 20  # Số lượng chunk điểm cao nhất được giữ lại sau reranking
    ALPHA_CC: float = 0.7  # Trọng số Convex Combination cho dense retrieval (0.7 dense + 0.3 sparse)

    # --- Tài nguyên & Môi trường ---
    VRAM_BUDGET_GB: float = 15.0  # Ngân sách bộ nhớ VRAM tối đa cho phép sử dụng (GB)
    DEVICE: str = "cuda"  # Thiết bị tính toán chính ("cuda" hoặc "cpu")

    # --- Đa ngữ & Ngữ cảnh mở rộng ---
    SUPPORTED_LANGUAGES: list[str] = field(
        default_factory=lambda: ["vi", "en", "zh"],
        metadata={"description": "Các ngôn ngữ được hệ thống hỗ trợ truy vấn và trích xuất."},
    )
    SLIDING_WINDOW_SIZE: int = 1  # Số chunk liền kề (trước và sau) được gộp vào ngữ cảnh
    MAX_P_TOP_DOCS: int = 10  # Số lượng tài liệu cấp document tối đa được chọn cho kết quả cuối

    def ensure_directories(self) -> None:
        """Tạo các thư mục lưu trữ dữ liệu nếu chưa tồn tại trên hệ thống."""
        self.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.QDRANT_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.BM25_INDEX_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, env_file: Optional[Union[str, Path]] = None) -> "Settings":
        """Khởi tạo đối tượng cấu hình Settings từ các biến môi trường hệ thống.

        Nếu tệp .env tồn tại và gói python-dotenv đã được cài đặt, phương thức sẽ
        tự động nạp các biến môi trường từ tệp đó trước khi phân tích cú pháp.

        Args:
            env_file: Đường dẫn tùy chọn tới tệp .env cần nạp. Nếu None, sẽ tìm tệp .env
                mặc định tại thư mục gốc của dự án.

        Returns:
            Settings: Đối tượng cấu hình với các giá trị được nạp từ biến môi trường
                hoặc giá trị mặc định nếu biến môi trường không tồn tại.
        """
        if load_dotenv is not None:
            if env_file is not None:
                load_dotenv(dotenv_path=str(env_file), override=True)
            else:
                default_env = _DEFAULT_BASE_DIR / ".env"
                if default_env.exists():
                    load_dotenv(dotenv_path=str(default_env), override=True)

        base_dir_str = os.getenv("BASE_DIR")
        base_dir = Path(base_dir_str).resolve() if base_dir_str else _DEFAULT_BASE_DIR

        raw_dir_str = os.getenv("DATA_RAW_DIR")
        data_raw_dir = Path(raw_dir_str).resolve() if raw_dir_str else base_dir / "data" / "raw"

        proc_dir_str = os.getenv("DATA_PROCESSED_DIR")
        data_processed_dir = (
            Path(proc_dir_str).resolve() if proc_dir_str else base_dir / "data" / "processed"
        )

        qdrant_str = os.getenv("QDRANT_DB_PATH")
        qdrant_db_path = (
            Path(qdrant_str).resolve() if qdrant_str else base_dir / "data" / "qdrant_db"
        )

        bm25_str = os.getenv("BM25_INDEX_PATH")
        bm25_index_path = (
            Path(bm25_str).resolve() if bm25_str else base_dir / "data" / "bm25_index"
        )

        supported_langs_str = os.getenv("SUPPORTED_LANGUAGES")
        if supported_langs_str:
            supported_languages = [
                lang.strip() for lang in supported_langs_str.split(",") if lang.strip()
            ]
        else:
            supported_languages = ["vi", "en", "zh"]

        return cls(
            BASE_DIR=base_dir,
            DATA_RAW_DIR=data_raw_dir,
            DATA_PROCESSED_DIR=data_processed_dir,
            QDRANT_DB_PATH=qdrant_db_path,
            BM25_INDEX_PATH=bm25_index_path,
            LLM_MODEL_REPO=os.getenv("LLM_MODEL_REPO", "Qwen/Qwen2.5-7B-Instruct-GGUF"),
            LLM_MODEL_FILE=os.getenv("LLM_MODEL_FILE", "qwen2.5-7b-instruct-q4_k_m.gguf"),
            LLM_N_GPU_LAYERS=int(os.getenv("LLM_N_GPU_LAYERS", "-1")),
            LLM_N_CTX=int(os.getenv("LLM_N_CTX", "4096")),
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            EMBEDDING_DIM=int(os.getenv("EMBEDDING_DIM", "1024")),
            EMBEDDING_MAX_SEQ_LEN=int(os.getenv("EMBEDDING_MAX_SEQ_LEN", "8192")),
            RERANKER_MODEL=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME", "medical_chunks"),
            TOP_K_DENSE=int(os.getenv("TOP_K_DENSE", "100")),
            TOP_K_SPARSE=int(os.getenv("TOP_K_SPARSE", "100")),
            TOP_K_CANDIDATES=int(os.getenv("TOP_K_CANDIDATES", "200")),
            RERANK_TOP_K=int(os.getenv("RERANK_TOP_K", "20")),
            ALPHA_CC=float(os.getenv("ALPHA_CC", "0.7")),
            VRAM_BUDGET_GB=float(os.getenv("VRAM_BUDGET_GB", "15.0")),
            DEVICE=os.getenv("DEVICE", "cuda"),
            SUPPORTED_LANGUAGES=supported_languages,
            SLIDING_WINDOW_SIZE=int(os.getenv("SLIDING_WINDOW_SIZE", "1")),
            MAX_P_TOP_DOCS=int(os.getenv("MAX_P_TOP_DOCS", "10")),
        )
