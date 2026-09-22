# 🏥 Medical RAG System — Hệ thống RAG Đa Ngôn ngữ Y Sinh

<!-- Badges placeholder -->
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

---

## 📖 1. Giới thiệu tổng quan (Overview)

**Medical RAG System** là hệ thống Hỏi - Đáp và Khai phá Tri thức Y sinh học Đa ngôn ngữ (**Multilingual Biomedical Retrieval-Augmented Generation**) được thiết kế chuyên sâu cho các truy vấn y khoa phức tạp. Hệ thống giải quyết bài toán rào cản ngôn ngữ và tính phân tán của tài liệu học thuật y khoa thông qua khả năng liên kết tri thức xuyên suốt giữa ba ngôn ngữ: **Tiếng Việt, Tiếng Anh và Tiếng Trung** (`vi`, `en`, `zh`).

### Các tính năng trọng tâm:
- **Truy xuất lai đa tầng (Hybrid Multistage Retrieval):** Kết hợp tìm kiếm từ khóa (BM25 với công cụ tách từ tiếng Việt chuyên biệt) và tìm kiếm ngữ nghĩa dày đặc (BGE-M3 Dense Vector) nhằm hạn chế tối đa hiện tượng bỏ sót thuật ngữ y khoa đặc thù.
- **Tái xếp hạng chéo đa ngôn ngữ (Cross-Encoder Reranking):** Ứng dụng `bge-reranker-v2-m3` để chuẩn hóa và đánh giá tương đồng sâu sắc giữa câu hỏi và từng phân đoạn văn bản ở cấp độ token.
- **Chiến lược tính điểm Max-P Document Scoring:** Cho phép xác định đồng thời các văn bản cấp độ tài liệu liên quan (`relevant_docs`) và các trích đoạn văn bản mang tính quyết định (`relevant_chunks`).
- **Sinh phản hồi chuẩn xác & có bằng chứng y khoa:** Kết hợp mô hình ngôn ngữ lớn `Qwen2.5-7B-Instruct` đã lượng tử hóa (GGUF Q4_K_M) để suy luận an toàn, trích dẫn tài liệu tham chiếu và phản hồi người dùng bằng tiếng Việt tự nhiên, chính xác.

---

## 🏛️ 2. Kiến trúc hệ thống (System Architecture)

Quy trình xử lý truy vấn và suy luận của hệ thống được minh họa qua sơ đồ kiến trúc sau:

```
[User Query (Việt)] ──► [Query Expansion / NER] ──► [Hybrid Retrieval: BM25 + BGE-M3 Dense]
                                                                  │
[Max-P Doc Score] ◄── [Top 200 Candidates] ◄── [Convex Combination / RRF]
        │                         │
[relevant_docs]         [Cross-Encoder Reranker (Top 20)] ──► [relevant_chunks]
                                  │
                        [Prompt + LLM Generator] ──► [Phản hồi Y khoa]
```

### Luồng xử lý chi tiết:
1. **Query Expansion & Medical NER:** Phân tích câu hỏi đầu vào, nhận diện thực thể y khoa (bệnh học, hoạt chất, triệu chứng) và mở rộng truy vấn đồng nghĩa đa ngữ.
2. **Hybrid Retrieval:** Tìm kiếm song song trên kho dữ liệu qua:
   - *BM25 Index:* Bắt chính xác tên thuốc, mã bệnh ICD, thuật ngữ y sinh hiếm.
   - *BGE-M3 Dense Vector:* Bắt ngữ nghĩa tổng quan và quan hệ ngữ nghĩa xuyên ngôn ngữ.
3. **Convex Combination / RRF:** Dung hợp kết quả từ BM25 và Vector Search để chọn lọc ra Top 200 ứng viên tiềm năng nhất.
4. **Max-P Document Score:** Tổng hợp điểm số từ các chunks để xếp hạng và xuất ra danh sách văn bản liên quan (`relevant_docs`).
5. **Cross-Encoder Reranker:** Đưa Top chunks qua mô hình `bge-reranker-v2-m3` để chọn lọc Top 10–20 chunks chuẩn xác nhất (`relevant_chunks`).
6. **Prompt Assembly & LLM Generation:** Lắp ráp ngữ cảnh giàu thông tin cùng prompt y khoa nghiêm ngặt đưa vào `Qwen2.5-7B-Instruct` sinh câu trả lời hoàn chỉnh kèm chú dẫn.

---

## 🛠️ 3. Công nghệ sử dụng (Tech Stack)

| Thành phần | Công nghệ / Thư viện | Vai trò |
| :--- | :--- | :--- |
| **Large Language Model** | `Qwen2.5-7B-Instruct-GGUF` | Sinh phản hồi y khoa đa ngữ chất lượng cao, chạy tối ưu phần cứng qua `llama-cpp-python` |
| **Dense Embedding** | `BAAI/bge-m3` | Vector hóa đa ngôn ngữ (1024 chiều), hỗ trợ văn bản dài lên tới 8192 tokens |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | Tái xếp hạng chính xác cao cho các ứng viên hàng đầu |
| **Vector Database** | `Qdrant` | Lưu trữ vector nhúng, hỗ trợ bộ lọc metadata y khoa và tìm kiếm tương đồng tốc độ cao |
| **Sparse / Lexical Search** | `rank-bm25` / `fastbm25` | Truy xuất từ khóa theo tần suất thuật ngữ y khoa |
| **Xử lý tiếng Việt** | `pyvi`, `underthesea` | Tách từ, chuẩn hóa tiếng Việt cho pipeline chỉ mục lexical |
| **Dịch thuật đa ngữ** | `NLLB` (`ctranslate2` / `transformers`) | Hỗ trợ đối chiếu thuật ngữ qua các ngôn ngữ vi - en - zh |
| **Orchestration** | `LangGraph`, `LangChain` | Quản lý luồng trạng thái, kiểm soát tiến trình RAG dạng đồ thị (State Graph) |

---

## 💾 4. Phân bổ tài nguyên VRAM (VRAM Budget)

Hệ thống được thiết kế tối ưu hóa bộ nhớ GPU để vận hành ổn định trên các dòng GPU phổ thông có VRAM từ **12 GB đến 16 GB** (ví dụ RTX 3060/4060Ti 16GB, RTX 3090/4080/4090, Tesla T4):

| Thành phần | Định dạng & Cấu hình | VRAM tiêu thụ ước tính | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-7B-Instruct** | GGUF Q4_K_M (Context 4k–8k) | **~4.5 GB** | Offload toàn bộ layers lên GPU qua `llama-cpp-python` |
| **BGE-M3 Embedding** | FP16 / INT8 | **~2.2 GB** | Xử lý batch embed truy vấn & tài liệu |
| **bge-reranker-v2-m3** | FP16 | **~1.5 GB** | Cross-Encoder tái chấm điểm Top 20 chunks |
| **CUDA Runtime & Cache** | PyTorch Memory Overhead | **~0.5 GB** | Bộ đệm tính toán động trong quá trình suy luận |
| **TỔNG CỘNG** | | **~8.7 GB / 15.0 GB** | **An toàn (~58% ngân sách VRAM 15GB)** |

---

## 📂 5. Cấu trúc thư mục dự án (Project Structure)

```
cocopila_2.0/
├── .gitignore                     # File cấu hình loại trừ cho Git
├── requirements.txt               # Danh mục thư viện phụ thuộc của dự án
├── README.md                      # Tài liệu hướng dẫn sử dụng và kiến trúc
├── config/                        # Cấu hình tập trung của dự án
│   ├── __init__.py
│   └── settings.py                # Quản lý tham số mô hình, đường dẫn, cổng dịch vụ
├── data/                          # Kho dữ liệu
│   ├── raw/                       # Dữ liệu y khoa thô đầu vào (.json, .jsonl, .pdf)
│   ├── processed/                 # Dữ liệu sau tiền xử lý, phân đoạn (chunks)
│   ├── qdrant_db/                 # Không gian lưu trữ cơ sở dữ liệu vector Qdrant
│   ├── bm25_index/                # Chỉ mục đảo BM25 đã được tuần tự hóa
│   └── sample/                    # Tập dữ liệu kiểm thử nhỏ
├── src/                           # Mã nguồn lõi (Core Modules)
│   ├── __init__.py
│   ├── ingestion/                 # Pipeline xử lý dữ liệu đầu vào và chunking
│   ├── retrieval/                 # Hybrid Search, Dense Search, BM25, Reranker, Max-P
│   ├── generation/                # Prompt Templates, LLM Wrapper, Pipeline Generator
│   ├── multilingual/              # Dịch thuật, phát hiện ngôn ngữ, từ điển y khoa
│   └── utils/                     # Tiện ích logging, GPU monitoring, metric helpers
├── scripts/                       # Các kịch bản thực thi tác vụ dòng lệnh
│   ├── ingest.py                  # Chạy quá trình nạp và lập chỉ mục dữ liệu
│   ├── evaluate.py                # Đánh giá độ chính xác (Precision, Recall, MRR, NDCG)
│   └── download_models.py         # Kịch bản tải tự động các mô hình trọng số
├── notebooks/                     # Thư mục thí nghiệm và phân tích dữ liệu
└── tests/                         # Bộ kiểm thử tự động (Unit test, Integration test)
```

---

## 🚀 6. Hướng dẫn cài đặt (Installation)

### 6.1. Yêu cầu hệ thống
- Hệ điều hành: Windows 10/11, Linux (Ubuntu 20.04+)
- Python: `>= 3.10`
- GPU: Khuyến nghị NVIDIA GPU (>= 12GB VRAM), CUDA Toolkit 12.1 trở lên.

### 6.2. Các bước cài đặt

1. **Khởi tạo môi trường ảo:**
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   .\venv\Scripts\activate
   # Hoặc trên Linux/macOS:
   source venv/bin/activate
   ```

2. **Cài đặt PyTorch tương thích CUDA 12.1:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Cài đặt các gói phụ thuộc dự án:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Cài đặt `llama-cpp-python` hỗ trợ tăng tốc phần cứng CUDA:**
   ```bash
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```

5. **Cấu hình môi trường:**
   Tạo file `.env` tại thư mục gốc dựa trên các thiết lập mặc định trong `config/settings.py`.

---

## ⚡ 7. Hướng dẫn sử dụng nhanh (Quick Start)

### Bước 1: Tải các mô hình trọng số
```bash
python scripts/download_models.py
```

### Bước 2: Nạp dữ liệu và xây dựng chỉ mục (Ingestion & Indexing)
Đặt các file văn bản y tế thô vào thư mục `data/raw/` rồi chạy:
```bash
python scripts/ingest.py --input-dir data/raw --chunk-size 512 --overlap 64
```

### Bước 3: Chạy đánh giá chất lượng hệ thống (Evaluation)
```bash
python scripts/evaluate.py --test-set data/sample/test_questions.json
```

---

## 👥 8. Phân công vai trò dự án (Team Roles)

Dự án được phân chia nhiệm vụ chuyên môn hóa theo 3 vai trò chính:

- **Dev A — Data Ingestion, Text Processing & Multilingual NLP:**
  - Xây dựng pipeline đọc, làm sạch và chunking tài liệu (`src/ingestion/`).
  - Xây dựng và tối ưu bộ chỉ mục tìm kiếm từ khóa BM25 (`src/retrieval/bm25_search.py`).
  - Tích hợp công cụ tách từ tiếng Việt (`pyvi`, `underthesea`) và module dịch thuật / mapping thuật ngữ y khoa (`src/multilingual/`).
  - Phụ trách kịch bản `scripts/ingest.py`.

- **Dev B — Vector Retrieval, Reranking & Document Scoring:**
  - Xây dựng và quản lý cơ sở dữ liệu vector Qdrant (`src/retrieval/vector_store.py`).
  - Tích hợp mô hình nhúng `BAAI/bge-m3` và kỹ thuật trích xuất Dense Vector.
  - Xây dựng thuật toán kết hợp Hybrid Search (Convex Combination / RRF).
  - Triển khai Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`) và thuật toán Max-P Document Scoring để xác định `relevant_docs` và `relevant_chunks`.

- **Dev C — LLM Generation, Orchestration, MLOps Lead:**
  - Tích hợp mô hình `Qwen2.5-7B-Instruct-GGUF` qua `llama-cpp-python` và xây dựng bộ Prompt Templates y khoa (`src/generation/`).
  - Điều phối toàn bộ luồng RAG bằng `LangGraph` State Graph.
  - Xây dựng REST API bằng `FastAPI` (`app/api.py`) và ứng dụng Web UI bằng `Streamlit` (`app/streamlit_app.py`).
  - Xây dựng pipeline đo lường, đánh giá hệ thống (`scripts/evaluate.py`).

---

## 📄 9. Giấy phép (License)

Dự án được phân phối dưới giấy phép **MIT License**. Chi tiết xem tại file `LICENSE` (placeholder). Mọi đóng góp học thuật và phát triển vì cộng đồng y tế đều được hoan nghênh.
#   c o c o p i l a _ 2 . 0  
 