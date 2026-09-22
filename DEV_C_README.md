# ⚙️ Hướng dẫn công việc cho DEV C (MLOps Lead)

Chào **Dev C**, bạn là người "thổi hồn" vào việc vận hành và biến mã nguồn thành một hệ thống thực thụ. Bạn sẽ quản lý cấu hình hệ thống, kiểm thử, giám sát tài nguyên GPU và viết kịch bản để tự động hóa toàn bộ quy trình chạy (Pipeline) và đóng gói kết quả nộp bài. *(Lưu ý: Phần API REST và UI Web đã được gỡ bỏ khỏi phiên bản MVP này để tập trung vào hiệu năng lõi).*

Dưới đây là danh sách các công việc và các file bạn đang "làm chủ" (Owner):

## 📂 Các file do bạn phụ trách:
**Module Config & Utils:**
- `config/settings.py`
- `config/prompt_templates.py`
- `src/utils/validator.py`
- `src/utils/metrics.py`
- `src/utils/vram_monitor.py`

**Scripts & Tests:**
- `scripts/run_pipeline.py`
- `scripts/pack_submission.py`
- `tests/*`

---

## 🎯 Chi tiết nhiệm vụ cần triển khai (Implementation Tasks)

### 1. Quản trị Cấu hình và Prompt (Config)
- **Settings (`settings.py`):** 
  - Khai báo tất cả các tham số tĩnh của dự án (đường dẫn DB, model ID, số layer offload, VRAM budget, batch size) sử dụng `dataclass`.
  - Hỗ trợ load từ biến môi trường (Environment Variables) hoặc file `.env`.
- **Prompt Templates (`prompt_templates.py`):** 
  - Quản lý các system prompts y khoa. Thiết kế prompt ép LLM phải (1) trả lời tiếng Việt, (2) trích dẫn nguồn, (3) từ chối nếu không có bằng chứng.
  - Viết các templates cho Query Expansion và Reranking instruction.

### 2. Validation & Đánh giá (Utils)
- **Validation (`validator.py`):** 
  - Dùng thư viện `jsonschema-rs` (viết bằng Rust, tốc độ siêu nhanh) để kiểm tra tính hợp lệ của file JSON kết quả trước khi nộp.
  - Đảm bảo output có đủ các trường `id`, `relevant_docs`, `relevant_chunks`. 
  - Đảm bảo các ID không bị lặp lại trong mỗi mảng. Xử lý tốt streaming parsing bằng `ijson` để không làm tràn RAM khi file submission quá lớn.
- **Metrics (`metrics.py`):** 
  - Code các hàm đo lường chuẩn hóa (Hit@K, MRR, NDCG, Recall@K, Precision@K) để benchmark mô hình nội bộ.
- **VRAM Monitor (`vram_monitor.py`):** 
  - Dùng `pynvml` (Nvidia Management Library) để giám sát VRAM GPU.
  - Viết 1 decorator hoặc context manager để in ra mức tiêu thụ VRAM trước/sau mỗi bước (như load BGE-M3, Reranker, LLM), cảnh báo hoặc ngắt luồng nếu mức dùng vọt qua 15GB.

### 3. Tự động hóa Automation (Scripts)
- **Run Pipeline (`run_pipeline.py`):**
  - Kịch bản chạy End-to-End hệ thống (Dành cho Kaggle/Colab).
  - Kết nối luồng: Đọc câu hỏi -> Dev B retrieval -> Dev B generation -> Lưu JSON output. Sử dụng thư viện `rich` để hiển thị progress bar và console log đẹp mắt.
- **Pack Submission (`pack_submission.py`):**
  - Kiểm tra file kết quả (bằng `validator.py`).
  - Tự động nén file `submission.json` thành `submission.zip` (không lồng trong thư mục con) theo đúng chuẩn nộp bài của BTC.

### 4. Kiểm thử (Tests)
- Triển khai unit test cho các luồng xử lý độc lập sử dụng `pytest`.

---

## 🤝 Phối hợp
- Nhận luồng pipeline lõi từ **Dev A** và **Dev B** để lắp ráp thành luồng chạy cuối cùng trong `run_pipeline.py`.
- Theo sát Dev B trong quá trình load model để giám sát và tinh chỉnh cấu hình trong `settings.py` sao cho không bao giờ OOM.
- Là chốt chặn cuối cùng kiểm tra file JSON trước khi đẩy lên Leaderboard.

Chúc bạn vận hành êm ái và không bao giờ gặp lỗi Runtime! 🚀
