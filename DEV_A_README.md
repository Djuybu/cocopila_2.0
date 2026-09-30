# 👨‍💻 DEV A — Người 1 (Mai Ngọc Duy): Data & Retrieval

Bạn phụ trách **P1-01 → P1-17** trong `Chia việc.xlsx` (sheet `Checklist`): chuẩn hóa
dữ liệu prototype, dựng corpus/index, retrieval (BM25 + dense + fusion), harness
Candidate Recall và chuẩn hóa output bàn giao cho Người 2.

- **Phạm vi code**: `src/data/**`, `src/retrieval/**`, `src/pipeline/{retrieve,benchmark,handoff}.py`, `configs/data/**`, `configs/retrieval/**`, `scripts/{prepare_data,download_mmedc,build_bm25_index,build_dense_index,run_retrieval,benchmark_retrieval,evaluate_retrieval,export_reranking_input,validate_reranking_input}.py`.
- **Bàn giao**: `outputs/p1_p2_handoff_qwen3/` + `.zip` (schema `medical-rag-candidates-v1`), xem `docs/p1_p2_handoff.md`.
- **Phụ thuộc P3**: P1-16 nhận `GraphRetriever` từ P3-12 để đưa vào fusion.

## 📋 Bảng công việc P1 (nội dung theo `Chia việc.xlsx`)

| ID | Giai đoạn | Module | Đầu việc | Deliverable | Ưu tiên | Phụ thuộc | Trạng thái (xlsx) | Trạng thái repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1-01 | Trước 01/10 | Data adapter | Chuẩn hóa schema prototype | Module data_adapter + schema JSON thống nhất | P0 | — | Hoàn thành | Hoàn thành |
| P1-02 | Trước 01/10 | Data preparation | Deduplicate dữ liệu prototype | Dataset sạch + báo cáo số lượng trước/sau dedup | P0 | P1-01 | Hoàn thành | Hoàn thành |
| P1-03 | Trước 01/10 | Evaluation split | Split theo document/title | File split + seed cố định | P0 | P1-02 | Hoàn thành | Hoàn thành |
| P1-04 | Trước 01/10 | Indexing | Dựng corpus chunk-level | chunk_corpus.jsonl / parquet | P0 | P1-01 | Hoàn thành | Hoàn thành |
| P1-05 | Trước 01/10 | BM25 | Baseline BM25 chunk retrieval | BM25Retriever + benchmark | P0 | P1-04 | Hoàn thành | Hoàn thành |
| P1-06 | Trước 01/10 | Dense retrieval | Baseline dense bằng BGE-M3 | DenseRetriever(BGE-M3) + index | P0 | P1-04 | Hoàn thành | Hoàn thành |
| P1-07 | Trước 01/10 | Dense retrieval | So sánh multilingual-e5 | Bảng so sánh BGE-M3 vs E5 | P1 | P1-06 | Hoàn thành | Hoàn thành |
| P1-08 | Trước 01/10 | Hybrid retrieval | Union BM25 + Dense | CandidatePool v1 | P0 | P1-05,P1-06 | Hoàn thành | Hoàn thành |
| P1-09 | Trước 01/10 | Fusion | RRF fusion | Fusion module + top-K fused candidates | P1 | P1-08 | Hoàn thành | Hoàn thành |
| P1-10 | Trước 01/10 | Metrics | Candidate Recall harness | Script evaluate_retrieval.py | P0 | P1-05,P1-06 | Hoàn thành | Hoàn thành |
| P1-11 | Trước 01/10 | Interface | Chuẩn hóa output cho MMD | candidates.jsonl schema cố định | P0 | P1-09 | Hoàn thành | Hoàn thành |
| P1-12 | Ngày 01/10 | Data audit | Phân tích dataset chính thức | data_audit.md + bảng thống kê | P0 | — | Chưa làm | Chưa làm (chờ dữ liệu BTC) |
| P1-13 | Sau 01/10 | Official indexing | Dựng index theo ID BTC | official BM25+dense indexes | P0 | P1-12 | Chưa làm | Chưa làm (chờ ID BTC) |
| P1-14 | Sau 01/10 | Top-K tuning | Tune K cho từng retriever | Bảng ablation K | P0 | P1-13 | Chưa làm | Chưa làm (chờ dữ liệu) |
| P1-15 | Sau 01/10 | Query expansion | Thử query expansion | QueryExpansionRetriever | P2 | P1-14 | Chưa làm | Chưa làm |
| P1-16 | Sau 01/10 | Graph integration | Nhận GraphRetriever từ Quế | Hybrid BM25+Dense+Graph | P1 | P3-12,P1-14 | Chưa làm | Chờ P3-12 trên dữ liệu thật |
| P1-17 | Tối ưu | Error analysis | Phân tích retrieval miss | retrieval_error_analysis.csv | P1 | P1-14 | Chưa làm | Chưa làm |


## 🎯 Chi tiết thực hiện & tiêu chí hoàn thành

- **P1-01 Chuẩn hóa schema prototype** — Định nghĩa `Query/Document/Chunk`; map `anchor→query`, `positive→chunk`, `meta.title/article_id→doc`; giữ mapping ngược về ID nguồn. *Tiêu chí*: load 1 batch và xuất đúng `query_id/doc_id/chunk_id`; không trùng ID.
- **P1-02 Deduplicate dữ liệu prototype** — Gom row cùng anchor/positive, loại duplicate chunk, gom nhiều negative cho cùng query. *Tiêu chí*: không còn duplicate query-positive gây leakage; thống kê được số query/doc/chunk.
- **P1-03 Split theo document/title** — Chia train/val/test theo `meta.title`/`article_id`, không random theo row. *Tiêu chí*: một document chỉ nằm trong đúng 1 split; tái lập bằng seed.
- **P1-04 Dựng corpus chunk-level** — Index chính thức từng chunk: `chunk_id, parent doc_id, title, text, metadata`. *Tiêu chí*: 100% chunk có parent doc hợp lệ; tra ngược chunk→doc được.
- **P1-05 Baseline BM25** — Index corpus bằng BM25, hỗ trợ top-k cấu hình, cache retrieval. *Tiêu chí*: chạy được toàn val; xuất Top-K kèm score và ID chính xác.
- **P1-06 Baseline dense BGE-M3** — Encode query/chunk, tạo vector index, thử cosine/dot-product theo khuyến nghị model. *Tiêu chí*: chạy được toàn val; Recall@K ghi vào experiment log.
- **P1-07 So sánh multilingual-e5** — Cùng split/corpus, benchmark E5 với cùng K và metric. *Tiêu chí*: có Recall@20/50/100, latency, RAM/VRAM; chọn baseline rõ ràng.
- **P1-08 Union BM25 + Dense** — Lấy top-k từ hai nguồn, union theo `chunk_id`, giữ score/rank nguồn. *Tiêu chí*: không duplicate chunk; có provenance `bm25_rank/dense_rank`.
- **P1-09 RRF fusion** — Cài Reciprocal Rank Fusion cho BM25+dense; thử vài giá trị RRF k. *Tiêu chí*: Candidate Recall không giảm đáng kể so với union; ranking ổn định.
- **P1-10 Candidate Recall harness** — Tính Recall@20/50/100/200 ở chunk-level và doc-level trước rerank. *Tiêu chí*: báo cáo theo query và aggregate; xác định query bị miss hoàn toàn.
- **P1-11 Chuẩn hóa output cho Người 2** — Mỗi candidate gồm `query_id, chunk_id, doc_id, text, rank nguồn, score nguồn, fused_score`. *Tiêu chí*: Người 2 đọc trực tiếp, không cần xử lý lại ID/text.
- **P1-12 Phân tích dataset chính thức** — Thống kê query/doc/chunk, độ dài text, số chunk/doc, language, duplicate, missing, mapping chunk→doc. *Tiêu chí*: phát hiện bất thường dữ liệu và chốt schema adapter trong ngày.
- **P1-13 Dựng index theo ID BTC** — BM25 + dense trên đúng doc/chunk chính thức; mọi kết quả map được về ID BTC. *Tiêu chí*: 100% candidate trả về ID hợp lệ; không dùng ID chunk tự tạo khi submit.
- **P1-14 Tune K cho từng retriever** — Sweep `K_BM25`, `K_dense`, `K_fusion`; đo Candidate Recall và latency. *Tiêu chí*: chọn K tại điểm recall tốt nhưng chi phí rerank chấp nhận được.
- **P1-15 Thử query expansion** — Sinh synonym/medical alias/multilingual variants; retrieve riêng rồi union/fusion. *Tiêu chí*: chỉ giữ nếu Candidate Recall/F2 tăng ổn định trên CV.
- **P1-16 Nhận GraphRetriever từ Quế** — Chuẩn hóa output GraphRetriever vào candidate schema và thêm vào RRF/fusion. *Tiêu chí*: có ablation có/không graph và Candidate Recall tương ứng.
- **P1-17 Phân tích retrieval miss** — Phân loại miss: exact term, synonym, multilingual, long query, rare entity, chunk boundary, graph relation. *Tiêu chí*: top nhóm lỗi có tần suất và hướng xử lý cụ thể.

## 🤝 Phối hợp
- Bàn giao `candidates.jsonl` + `queries.json` + `labels.json` + `registry.json` cho **Người 2** (P2-01).
- Nhận `GraphRetriever` (P3-12) từ **Người 3** cho P1-16.
- Giữ nguyên fingerprint corpus/query/label để đối chiếu khi export/benchmark.

