# 👨‍💻 Hướng dẫn công việc cho DEV A (Data & Infrastructure Engineer)

Chào **Dev A**, vai trò của bạn là nền móng của toàn bộ hệ thống. Nhiệm vụ chính của bạn là đảm bảo dữ liệu văn bản y khoa đa ngôn ngữ được tiền xử lý sạch sẽ, tách từ chuẩn xác và được nạp vào cơ sở dữ liệu Vector (Qdrant) cũng như chỉ mục từ khóa (BM25) một cách tối ưu nhất.

Dưới đây là danh sách các công việc và các file bạn đang "làm chủ" (Owner):

## 📂 Các file do bạn phụ trách:
- `src/ingestion/segmentor.py`
- `src/ingestion/chunker.py`
- `src/ingestion/indexer.py`

---

## 🎯 Chi tiết nhiệm vụ cần triển khai (Implementation Tasks)

### 1. Tách từ tiếng Việt chuyên biệt Y khoa (`segmentor.py`)
- **Vấn đề:** Các từ ghép y khoa như "nhồi máu cơ tim", "ung thư biểu mô" rất dễ bị các tokenizer thông thường cắt vụn, làm mất ngữ nghĩa.
- **Yêu cầu:** 
  - Triển khai class `VietnameseSegmentor`.
  - Bạn có thể dùng `PyVi` (tốc độ cao) hoặc `VnCoreNLP` (chính xác cao).
  - Triển khai hàm `expand_abbreviations` để tự động nội suy các từ viết tắt phổ biến trong y bạ (HA -> huyết áp, XN -> xét nghiệm).
  - Kết quả đầu ra của hàm `segment()` phải nối các âm tiết của từ ghép bằng dấu gạch dưới `_`.

### 2. Xử lý ngữ cảnh bằng Cửa sổ trượt (`chunker.py`)
- **Vấn đề:** Khi chunking tài liệu dài, các câu bị cắt ngang sẽ mất đi ngữ cảnh y khoa quan trọng.
- **Yêu cầu:**
  - Triển khai class `SlidingWindowChunker`.
  - Viết logic `apply_context_window(window_size=1)`: Với mỗi `chunk` hiện tại, hãy ghép thêm phần text của chunk ngay trước nó và ngay sau nó (cùng `doc_id`).
  - Dùng đoạn text đã ghép (expanded) để đưa cho Dev B chạy Embedding, nhưng khi lưu trữ và trả về vẫn phải giữ đúng `chunk_id` gốc để nộp bài hợp lệ.

### 3. Khởi tạo và Tối ưu Cơ sở dữ liệu Vector (`indexer.py`)
- **Yêu cầu Qdrant (Dense Index):**
  - Khởi tạo kết nối tới Qdrant lưu trữ cục bộ (disk-backed) để chạy được trên Kaggle/Colab mà không tốn RAM.
  - Cấu hình bắt buộc: Sử dụng thuật toán HNSW kết hợp với **Scalar Quantization (SQ8)** (Quantization về INT8). Điều này giúp giảm 75% RAM tiêu thụ, cực kỳ quan trọng để không bị Out-Of-Memory (OOM) trên GPU T4 15GB.
  - Viết logic batch upsert vào Qdrant để nạp hàng triệu vector nhanh chóng.
- **Yêu cầu BM25 (Sparse Index):**
  - Viết logic để khởi tạo chỉ mục BM25 (dùng `fastbm25` hoặc `rank_bm25`) dựa trên danh sách các tokens đã được `segmentor.py` xử lý.
  - Cung cấp hàm `save_index` và `load_index` (sử dụng `pickle`) để lưu BM25 index ra ổ cứng.

---

## 🤝 Phối hợp
- Sau khi bạn hoàn thiện luồng nạp dữ liệu, bạn cần cung cấp bộ dữ liệu (đã index) cho **Dev B** để họ có thể tiến hành truy vấn thử nghiệm (Hybrid Search).
- Đảm bảo các script của bạn xử lý tốt các file lớn (tránh đọc toàn bộ file JSON khổng lồ vào RAM cùng một lúc).

Chúc bạn code thuận lợi và không dính bug bộ nhớ! 🚀
