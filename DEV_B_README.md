# 🧠 DEV B — Người 2 (Mạc Duy): Reranking, Scoring & Evaluation

Bạn phụ trách **P2-01 → P2-15** trong `Chia việc.xlsx` (sheet `Checklist`): nhận
candidates từ Người 1, rerank, đánh giá F2 chunk-level, tuning threshold/fallback/max,
tạo dữ liệu huấn luyện + hard negatives, fine-tune, cross-validation và bàn giao gói P2→P3.

- **Phạm vi code**: `src/reranking/**`, `src/scoring/**`, `src/training/**`, `src/evaluation/{evaluate_f2,fbeta,reranking_metrics}.py`, `src/pipeline/{rerank,reranker_benchmark,predict,p2_pipeline}.py`, `configs/reranker/**`, `configs/training/**`, `scripts/{run_reranking,benchmark_reranking,evaluate_f2,sweep_threshold,sweep_fallback,sweep_max_output,cv_threshold,generate_training_data,mine_hard_negatives,finetune_reranker}.py`.
- **Bàn giao**: `p2_to_p3_handoff/` + `.zip` (`reranked.jsonl`, `best_chunk_selector.yaml`, `p2_selected_chunks.json`, `threshold_sweep.csv`, `fn_analysis.csv`, `manifest.json`) — dùng trực tiếp cho P3-02/04/05/06/15.
- **Tài liệu**: `docs/p2_02_manual_calculation_guide.md` (công thức F2 tính tay).

## 📋 Bảng công việc P2 (nội dung theo `Chia việc.xlsx`)

| ID | Giai đoạn | Module | Đầu việc | Deliverable | Ưu tiên | Phụ thuộc | Trạng thái (xlsx) | Trạng thái repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2-01 | Trước 01/10 | Reranking | Dựng BGE reranker baseline | Reranker inference module | P0 | P1-11 | Hoàn thành | Hoàn thành |
| P2-02 | Trước 01/10 | Metrics | Evaluation F2 chunk-level | evaluate_f2.py | P0 | — | Đang làm | Hoàn thành |
| P2-03 | Trước 01/10 | Threshold | Sweep threshold reranker | threshold_sweep.csv | P0 | P2-01,P2-02 | Chưa làm | Hoàn thành |
| P2-04 | Trước 01/10 | Fallback | Thử minimum fallback Top-N | Fallback selector | P0 | P2-03 | Chưa làm | Hoàn thành |
| P2-05 | Trước 01/10 | Output cap | Thử max-N | Max-output selector | P1 | P2-03 | Chưa làm | Hoàn thành |
| P2-06 | Trước 01/10 | Training data | Tạo pair positive/negative | reranker_train.jsonl | P0 | P1-03 | Chưa làm | Hoàn thành |
| P2-07 | Trước 01/10 | Hard negatives | Mine hard negatives | hard_negatives.jsonl | P1 | P1-08,P2-06 | Chưa làm | Hoàn thành |
| P2-08 | Trước 01/10 | Fine-tuning | Fine-tune reranker | Reranker checkpoint v1 | P1 | P2-07 | Chưa làm | Hoàn thành code (chờ checkpoint thật) |
| P2-09 | Trước 01/10 | CV | Cross-validation threshold | CV threshold report | P1 | P2-03 | Chưa làm | Hoàn thành |
| P2-10 | Sau 01/10 | Metric replication | Mô phỏng metric BTC | official_metric_local.py | P0 | P1-12 | Chưa làm | Chưa làm (chờ định nghĩa metric BTC) |
| P2-11 | Sau 01/10 | Reranker benchmark | Benchmark reranker trên data thật | reranker_benchmark.csv | P0 | P2-10,P1-13 | Chưa làm | Hoàn thành code (chạy thật cần GPU/weights) |
| P2-12 | Sau 01/10 | F2 tuning | Joint sweep selector | best_chunk_selector.yaml | P0 | P2-11,P1-14 | Chưa làm | Hoàn thành |
| P2-13 | Sau 01/10 | Score calibration | Thử calibration nếu cần | Calibration ablation | P2 | P2-12 | Chưa làm | Chưa làm (chỉ làm nếu CV tăng) |
| P2-14 | Tối ưu | False negative analysis | Phân tích FN sau rerank | fn_analysis.csv | P1 | P2-12 | Chưa làm | Hoàn thành |
| P2-15 | Tối ưu | Inference optimization | Tối ưu batch reranking | rerank pipeline tối ưu | P2 | P2-12 | Chưa làm | Chưa làm (tùy chọn latency) |

## 🎯 Chi tiết thực hiện & tiêu chí hoàn thành

- **P2-01 Dựng BGE reranker baseline** — Nhận candidates từ Người 1; score từng cặp query-chunk bằng BGE reranker v2 m3 (và Qwen3 qua factory). *Tiêu chí*: xuất score cho toàn bộ candidates; giữ nguyên query/chunk/doc ID.
- **P2-02 Evaluation F2 chunk-level** — Tính precision, recall, F1, F2 theo query và macro aggregate cho `relevant_chunks`. *Tiêu chí*: khớp công thức F2; có unit test trên ví dụ nhỏ tính tay.
- **P2-03 Sweep threshold reranker** — Quét threshold trên validation; lưu P/R/F2 và số chunk output/query. *Tiêu chí*: xác định được vùng threshold plateau, không chỉ 1 điểm tốt nhất.
- **P2-04 Thử minimum fallback Top-N** — Nếu số chunk qua threshold < N_min, lấy Top-N theo reranker score; sweep N_min. *Tiêu chí*: có ablation threshold-only vs fallback; F2 không giảm trên CV.
- **P2-05 Thử max-N** — Giới hạn số chunk output/query để tránh precision sụt khi quá nhiều candidate qua threshold. *Tiêu chí*: có sweep max-N và phân tích tác động Precision/Recall/F2.
- **P2-06 Tạo pair positive/negative** — Từ anchor-positive-negative tạo cặp query/chunk có label; ưu tiên giữ metadata và split đúng. *Tiêu chí*: không leakage giữa train/val; tỷ lệ pos/neg được báo cáo.
- **P2-07 Mine hard negatives** — Dùng BM25/dense top results sai label làm hard negative; loại false negative nghi ngờ. *Tiêu chí*: hard negative có similarity cao hơn random negative và không chứa ground-truth positive.
- **P2-08 Fine-tune reranker** — Fine-tune cross-encoder/reranker trên positive + hard negatives; lưu config, checkpoint, seed. *Tiêu chí*: F2/Recall trên validation ≥ baseline hoặc có lý do loại bỏ.
- **P2-09 Cross-validation threshold** — Tune threshold/fallback qua nhiều fold hoặc bootstrap query; tránh overfit một validation split. *Tiêu chí*: chọn threshold ổn định; báo mean/std F2.
- **P2-10 Mô phỏng metric BTC** — Đọc định nghĩa metric chính thức; xác định cách combine doc/chunk score nếu BTC có; viết evaluator giống BTC nhất có thể. *Tiêu chí*: chạy được submission JSON và trả score local; unit test các edge case.
- **P2-11 Benchmark reranker trên data thật** — So sánh off-the-shelf BGE với fine-tuned; thử batch size/max length hợp lý. *Tiêu chí*: có F2_chunk, latency/query, VRAM và lựa chọn model.
- **P2-12 Joint sweep selector** — Tune `threshold_chunk, fallback_chunk, max_chunk, candidate_K`; tối ưu trực tiếp F2_chunk. *Tiêu chí*: cấu hình tái lập được; score xác nhận lại trên holdout.
- **P2-13 Thử calibration nếu cần** — Nếu raw score phân phối khác mạnh giữa query, thử Platt/isotonic hoặc query-relative threshold. *Tiêu chí*: chỉ giữ nếu CV F2 tăng ổn định; tránh thêm complexity không cần thiết.
- **P2-14 Phân tích FN sau rerank** — Tách FN do candidate miss và FN do reranker/threshold; gửi candidate-miss lại Người 1. *Tiêu chí*: mỗi FN có nguyên nhân; loop cải tiến giữa P1/P2 được ghi nhận.
- **P2-15 Tối ưu batch reranking** — Batch inference, truncate hợp lý, cache query, mixed precision nếu an toàn. *Tiêu chí*: không đổi prediction so với bản chuẩn; latency giảm hoặc throughput tăng rõ.

## 🤝 Phối hợp
- Nhận index/candidates từ **Người 1** (P1-11) để test retrieval/rerank pipeline.
- Bàn giao gói P2→P3 cho **Người 3**; gói dùng `input_manifest.json` giữ provenance P1.
- Không coi điểm tuning trên `best_chunk_selector.yaml` là điểm generalisation; dùng out-of-fold trong `selector_cv_report.json`.

