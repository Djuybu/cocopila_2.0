# Quy Chuẩn Commit (Commit Convention) - Dự Án Cocopila 2.0

> **Dự án**: Medical Retrieval & RAG Competition  
> **Áp dụng cho**: Toàn bộ thành viên đội ngũ (Data, Pipeline, Reranking, Graph, DevOps)  
> **Tài liệu tham chiếu**: [Checklist công việc](Checklist.csv), [Kiến trúc hệ thống](architecture.md), [Ghi chú thực nghiệm](experiment_notes.md)

---

## 1. Mục Tiêu & Nguyên Tắc Cốt Lõi

Trong một dự án AI/RAG thi đấu với pipeline đa tầng phức tạp (*Data Adapter → Hybrid Retrieval → Cross-Encoder Reranking → Scoring & Threshold Tuning → Submission*), lịch sử Git không chỉ dùng để lưu trữ code mà còn là **hệ thống truy vết khoa học (experiment provenance & traceability)**.

Mỗi commit cần đảm bảo 4 nguyên tắc:
1. **Traceability (Khả năng truy vết)**: Mỗi commit phải liên kết được với mã công việc trong `Checklist.csv` (ví dụ: `P1-04`, `P2-01`, `P3-06`, `S-02`) hoặc mã thực nghiệm (`exp001`, `exp002`).
2. **Atomic Commits (Tính nguyên tử)**: Mỗi commit chỉ giải quyết **một** nhiệm vụ logic duy nhất. Tuyệt đối không gom nhiều tính năng không liên quan hoặc vừa sửa bug vừa format code vào chung một commit.
3. **Reproducibility (Tính tái lập)**: Code và config đi cùng commit phải luôn ở trạng thái chạy được, không làm gãy pipeline hoặc test suite (`pytest` phải pass).
4. **Clean Git Hygiene (Không rác dữ liệu)**: Tuyệt đối không commit dữ liệu lớn, model weights, cache, outputs hay file nén submission.

---

## 2. Cấu Trúc Commit Message Chuẩn

Dự án tuân thủ chuẩn **Conventional Commits 1.0.0** với định dạng mở rộng cho Machine Learning / Data Science:

```text
<type>(<scope>): <subject>

[optional body - giải thích lý do, bối cảnh và tác động kỹ thuật]

[optional footer - mã task Checklist, Run ID, breaking change, issue reference]
```

### Chi tiết các thành phần:

```text
feat(reranking): add Qwen3-Reranker-0.6B integration and evaluation suite

- Implement QwenReranker class wrapping AutoModelForCausalLM.
- Support prompt format with custom instruction for medical passage ranking.
- Add FP16 inference option to optimize VRAM on Kaggle 2xT4 GPUs.

Task: P2-01
Run-ID: exp004
```

---

## 3. Quy Định Chi Tiết Về Các Trường

### 3.1. Type (Bắt buộc)

| Type | Ý nghĩa | Khi nào sử dụng |
| :--- | :--- | :--- |
| `feat` | Tính năng mới | Thêm retriever mới, thêm reranker, thêm logic chọn candidate, thêm validator submission. |
| `fix` | Sửa lỗi | Sửa bug logic điểm, sửa lỗi mapping chunk-doc, sửa lỗi rò rỉ dữ liệu (leakage), sửa syntax error. |
| `refactor` | Tái cấu trúc code | Tối ưu hóa cấu trúc code, tách module mà không thay đổi hành vi hoặc kết quả tính toán. |
| `perf` | Tối ưu hiệu năng | Giảm VRAM, tăng throughput (QPS), tối ưu hóa batch inference, tăng tốc độ BM25 / vector search. |
| `exp` | Thực nghiệm RAG/ML | Thay đổi hyperparameters, tuning threshold, chạy ablation study hoặc cấu hình pipeline mới. |
| `test` | Kiểm thử | Thêm unit test, smoke test cho retriever/reranker, cập nhật test fixtures hoặc mock data. |
| `docs` | Tài liệu | Cập nhật README, Checklist.csv, kiến trúc, sơ đồ Neo4j schema, hướng dẫn commit. |
| `config` | Cấu hình | Thay đổi file YAML cấu hình trong `configs/` (dataset paths, top_k, model checkpoints). |
| `chore` | Việc phụ trợ | Cập nhật `.gitignore`, dependencies trong `pyproject.toml`, dọn dẹp scripts phụ trợ. |

---

### 3.2. Scope (Khuyến khích sử dụng)

Scope chỉ rõ module bị ảnh hưởng, map trực tiếp với cấu trúc thư mục của dự án:

| Scope | Thư mục / Thành phần tương ứng | Ví dụ nội dung |
| :--- | :--- | :--- |
| `data` | `src/data/`, `data/` | Loader, adapter, schema, chunking, data audit, split validation. |
| `retrieval` | `src/retrieval/` | BM25, dense BGE-M3, hybrid fusion, RRF, candidate pooling. |
| `reranking` | `src/reranking/` | Cross-encoder BGE, Qwen reranker, scoring pairs, batch rerank. |
| `scoring` | `src/scoring/` | Chunk threshold selector, fallback Top-N, doc max-aggregation. |
| `eval` | `src/evaluation/` | F2-score chunk/doc, NDCG@K, MRR@K, Recall@K, ECE calibration. |
| `submission` | `src/submission/` | Format submission JSON, official validator, zip packing. |
| `graph` | `docs/graph_schema.md`, Neo4j | Knowledge graph schema, entity extraction, relation traversal. |
| `pipeline` | `src/pipeline/` | Orchestrator, CLI runner, multi-stage pipeline flow. |
| `configs` | `configs/` | File cấu hình `configs/**/*.yaml`. |
| `notebooks` | `notebooks/` | Jupyter notebooks phân tích, benchmark, prototype. |
| `scripts` | `scripts/` | Script entrypoint độc lập (build testcases, sweep threshold). |

---

### 3.3. Subject (Tiêu đề - Bắt buộc)

- **Độ dài**: Tối đa **72 ký tự** (giúp hiển thị trọn vẹn trên GitHub / GitLab / CLI `git log --oneline`).
- **Thì & Giọng văn**: Sử dụng **thể mệnh lệnh thời hiện tại** (Imperative mood, ví dụ: `add`, `fix`, `implement`, `update`, `optimize`; không dùng `added`, `fixing`, `adds`).
- **Chữ hoa/thường**: Bắt đầu bằng chữ thường sau dấu hai chấm (ví dụ: `feat(retrieval): add reciprocal rank fusion`).
- **Dấu kết thúc**: **Không** đặt dấu chấm (`.`) ở cuối dòng subject.
- **Ngôn ngữ**: Khuyến khích sử dụng **Tiếng Anh** để chuẩn hóa và dễ làm việc với các thư viện tự động. Nếu dùng Tiếng Việt, phải diễn đạt ngắn gọn, gãy gọn và nhất quán.

---

### 3.4. Body (Thân commit - Tùy chọn nhưng khuyến khích)

Cần thiết khi thay đổi mang tính phức tạp, thay đổi thuật toán, hoặc ảnh hưởng đến metric:
- Giải thích **tại sao (Why)** thay đổi này được thực hiện, không chỉ nhắc lại code làm gì.
- Nêu rõ **sự khác biệt** so với hành vi trước đó.
- Nêu rõ các lưu ý kỹ thuật (VRAM, dependency mới, yêu cầu GPU).

---

### 3.5. Footer (Chân commit - Tùy chọn)

- **Liên kết Task ID**: Dùng cú pháp `Task: <ID>` (ví dụ: `Task: P1-05`, `Task: P2-01`, `Task: P3-04`).
- **Mã thực nghiệm (Run ID)**: Dùng `Run-ID: <run_name>` (ví dụ: `Run-ID: exp003`).
- **Breaking Changes**: Nếu thay đổi schema JSON hoặc interface làm gãy code của người khác, bắt đầu bằng `BREAKING CHANGE: <mô tả chi tiết>`.

---

## 4. Bảng Quy Ước Nhánh (Branching Convention)

Để tránh xung đột khi nhiều thành viên cùng phát triển trên repo:

### 4.1. Quy tắc đặt tên nhánh

Cú pháp: `<type>/<task-id>-<short-description>`

| Mục đích | Cú pháp nhánh | Ví dụ |
| :--- | :--- | :--- |
| Tính năng mới theo Checklist | `feat/<task-id>-<slug>` | `feat/p2-01-bge-reranker`<br>`feat/p1-06-dense-retriever` |
| Sửa lỗi kỹ thuật / bug | `fix/<task-id>-<slug>` | `fix/p1-04-duplicate-chunk-id`<br>`fix/p3-07-zip-validator` |
| Chạy thực nghiệm / Tuning | `exp/<run-id>-<slug>` | `exp/exp004-qwen-reranker`<br>`exp/p2-03-threshold-sweep` |
| Refactor / Tối ưu code | `refactor/<slug>` | `refactor/modular-retrieval`<br>`refactor/split-scoring-utils` |
| Tài liệu / Checklist | `docs/<slug>` | `docs/commit-convention`<br>`docs/update-checklist-status` |

### 4.2. Quy tắc Merge vào `main`
- Nhánh `main` là nhánh phát hành production/submission, luôn phải chạy được.
- Chỉ tạo Pull Request / Merge vào `main` sau khi:
  1. Toàn bộ unit test vượt qua: `python -m pytest -q`.
  2. Code đã được test khói (smoke test) với pipeline.
  3. Có sự đồng thuận hoặc review từ người phụ trách module liên quan (P1: Mai Ngọc Duy, P2: Mạc Duy, P3: Quế).

---

## 5. Quy Tắc Vàng "Git Hygiene" Cho Dự Án AI/ML

> [!CAUTION]
> **TUYỆT ĐỐI KHÔNG COMMIT CÁC TỆP SAU LÊN GIT:**
> 1. **Dữ liệu lớn**: Không commit bất kỳ file nào trong `data/raw/`, `data/interim/`, `data/processed/`, file `.json` hàng trăm MB, file `.npy`, `.parquet`.
> 2. **Model Weights / Checkpoints**: Không commit file `.pt`, `.pth`, `.bin`, `.safetensors`, `.gguf`, `.onnx`.
> 3. **Outputs đồ sộ**: Không commit toàn bộ thư mục `outputs/<run_name>/` (chỉ commit file log tóm tắt như `experiments/experiment_log.csv` hoặc config YAML).
> 4. **Submission ZIP**: Không commit file `.zip` nộp bài vào git (chỉ commit script sinh zip và file `submissions/submission_log.csv`).
> 5. **Môi trường & Cache**: Không commit `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.env` (chứa API key).

### Cách xử lý nếu lỡ `git add` nhầm file nặng:
```bash
# Bỏ stage file nhầm trước khi commit
git reset HEAD path/to/large_file.npy

# Kiểm tra lại dung lượng và trạng thái
git status
```

---

## 6. Ví Dụ Cụ Thể (Good vs. Bad Commits)

### 6.1. Ví dụ chuẩn (Good Commits)

#### Ví dụ 1: Tính năng Reranker (P2-01)
```text
feat(reranking): implement Qwen3-Reranker model wrapper

Add QwenReranker class using Hugging Face AutoModelForCausalLM.
Supports customized medical instruction prompts and automatic
sigmoid score scaling.

Task: P2-01
```

#### Ví dụ 2: Sửa lỗi Mapping ID (P1-04)
```text
fix(data): ensure 100% chunks map to valid parent document ID

Resolve orphaned chunk IDs when adapting prototype articles.
Filter out corrupted metadata rows during schema normalization.

Task: P1-04
```

#### Ví dụ 3: Thực nghiệm quét Threshold F2 (P2-03)
```text
exp(scoring): tune chunk acceptance threshold on validation set

Sweep threshold from 0.2 to 0.8 with step 0.05.
Identified plateau between 0.45 and 0.55 yielding peak F2=0.684.

Task: P2-03
Run-ID: exp008
```

#### Ví dụ 4: Sửa lỗi chính tả & cú pháp Notebook
```text
fix(notebooks): resolve syntax errors in cell headers and fix typos

- Add missing comment prefix '#' to cell titles (Cell 1 to Cell 8).
- Correct docstring grammar from 'Numerical safe' to 'Numerically safe'.
- Update overview from 4 to 5 evaluation pillars to match benchmark suite.

Ref: notebooks/P2_01_reranker_evaluation.ipynb
```

---

### 6.2. Ví dụ xấu cần tránh (Bad Commits)

| Commit Message Xấu | Tại sao xấu? | Cách sửa đúng |
| :--- | :--- | :--- |
| `update` | Quá mơ hồ, không rõ làm gì, ở đâu. | `fix(submission): validate non-empty prediction array` |
| `fix bug` | Không rõ bug gì, nguyên nhân hay tác động. | `fix(retrieval): prevent division by zero in BM25 score calculation` |
| `done task P2` | Không mang thông tin kỹ thuật, không thể tra cứu. | `feat(reranking): add cross-encoder inference batching for BGE` |
| `add code and data` | Vi phạm Git Hygiene (commit cả code và data). | Tách riêng: chỉ commit code với `feat(data): add jsonl chunk parser` |
| `final_v2_fix_last_hope` | Đặt tên cảm tính, phá vỡ cấu trúc commit lịch sử. | `exp(scoring): apply fallback top-3 when threshold yields no chunks` |

---

## 7. Checklist Tự Kiểm Tra Trước Khi Push (Pre-push Checklist)

Trước khi thực hiện `git push origin <branch-name>`, hãy tự kiểm tra 5 câu hỏi:

- [ ] 1. **Kiểm tra status**: Đã chạy `git status` để chắc chắn không có file rác (`.npy`, `.pt`, `large json`, `__pycache__`) bị add nhầm chưa?
- [ ] 2. **Kiểm tra test suite**: Đã chạy `python -m pytest -q` và tất cả test cases đều `PASSED` chưa?
- [ ] 3. **Kiểm tra commit format**: Commit message đã có dạng `<type>(<scope>): <subject>` chưa?
- [ ] 4. **Kiểm tra liên kết Task**: Đã gắn mã Task Checklist (`P1-xx`, `P2-xx`, `P3-xx`, `S-xx`) vào footer chưa?
- [ ] 5. **Kiểm tra tính độc lập**: Thay đổi này có làm hỏng luồng chạy của thành viên khác không?
