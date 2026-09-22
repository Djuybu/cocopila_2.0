# 🧠 Hướng dẫn công việc cho DEV B (AI Pipeline & Model Specialist)

Chào **Dev B**, bạn nắm giữ "trái tim" AI của hệ thống. Bạn chịu trách nhiệm toàn bộ luồng tìm kiếm lai (Hybrid Search), tái xếp hạng (Reranking), cơ chế LLM sinh văn bản (Generation) và xử lý rào cản ngôn ngữ (Multilingual).

Dưới đây là danh sách các công việc và các file bạn đang "làm chủ" (Owner):

## 📂 Các file do bạn phụ trách:
**Module Retrieval (Tìm kiếm):**
- `src/retrieval/dense_search.py`
- `src/retrieval/sparse_search.py`
- `src/retrieval/hybrid_fusion.py`
- `src/retrieval/reranker.py`
- `src/retrieval/aggregator.py`

**Module Generation (Sinh văn bản):**
- `src/generation/llm_loader.py`
- `src/generation/rag_chain.py`

**Module Multilingual (Đa ngôn ngữ):**
- `src/multilingual/shift.py`
- `src/multilingual/translator.py`
- `src/multilingual/medical_ner.py`

---

## 🎯 Chi tiết nhiệm vụ cần triển khai (Implementation Tasks)

### 1. Hybrid Search & Reranking (Luồng truy xuất)
- **Dense Search (`dense_search.py`):** Viết logic sử dụng mô hình `BAAI/bge-m3` để mã hóa truy vấn thành vector (1024-dim) và query vào Qdrant DB do Dev A chuẩn bị. Chú ý thêm instruction "Represent this sentence for searching relevant passages:" khi encode truy vấn.
- **Sparse Search (`sparse_search.py`):** Viết hàm load BM25 index do Dev A xây dựng và truy vấn ra Top K kết quả.
- **Fusion (`hybrid_fusion.py`):** Implement 2 thuật toán dung hợp điểm số: 
  - *Reciprocal Rank Fusion (RRF)*
  - *Convex Combination (CC)* (Ưu tiên dùng CC với hệ số alpha điều chỉnh giữa Dense và Sparse). Nhớ chuẩn hóa điểm số (Min-Max normalization) trước khi fuse.
- **Cross-Encoder (`reranker.py`):** Dùng mô hình `bge-reranker-v2-m3` (chạy trên GPU) để tái chấm điểm (rerank) Top 200 ứng viên từ Hybrid Search. Trả về Top 10-20 chunk chất lượng nhất (`relevant_chunks`).
- **Aggregation (`aggregator.py`):** Áp dụng chiến lược `Max-P` để quy đổi điểm của các chunks về điểm của tài liệu (doc), từ đó lấy ra danh sách tài liệu (`relevant_docs`).

### 2. LLM Engine & RAG Chain (Luồng sinh)
- **LLM Loader (`llm_loader.py`):** 
  - Dùng `llama-cpp-python` để nạp mô hình Qwen2.5-7B-Instruct định dạng GGUF (Q4_K_M). 
  - Cấu hình offload toàn bộ các layer lên GPU (`n_gpu_layers=-1`). Đảm bảo quản lý VRAM tốt (VRAM cho LLM chiếm khoảng 4.5GB).
- **LangGraph Orchestration (`rag_chain.py`):** 
  - Xây dựng luồng `StateGraph` với các node: Retrieve -> Rerank -> Aggregate -> Generate -> Reflect.
  - Implement cơ chế tự sửa lỗi (Reflect) nếu LLM nhận thấy context không đủ để trả lời truy vấn.

### 3. Xử lý Đa ngôn ngữ chuyên sâu
- **Medical NER & Translation (`medical_ner.py`, `translator.py`):** 
  - Nhận diện thực thể lâm sàng trong câu hỏi (disease, drug, symptom).
  - Mở rộng câu truy vấn đa ngữ (mapping sang mã UMLS).
  - (Tùy chọn) Triển khai mô hình dịch NLLB-200.
- **SHIFT Calibration (`shift.py`):** 
  - Thuật toán loại bỏ thiên kiến ngôn ngữ. Tính toán "Vector tương đối" giữa tiếng Việt và tiếng Anh/Trung, sau đó tịnh tiến (shift) vector không gian để hệ thống chỉ so khớp ý nghĩa y khoa chứ không bị lệch do khác biệt ngôn ngữ.

---

## 🤝 Phối hợp
- Nhận Index từ **Dev A** để test retrieval pipeline.
- Làm việc chặt chẽ với **Dev C** để cung cấp VRAM metrics thực tế của các mô hình, đảm bảo hệ thống không vượt quá 15GB VRAM.
- Tuân thủ cấu trúc dữ liệu JSON đầu ra mà hệ thống yêu cầu.

Cố lên! Bạn đang xây dựng bộ não thông minh nhất cho dự án! 🚀
