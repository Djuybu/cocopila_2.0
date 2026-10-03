# ⚙️ DEV C — Người 3 (Quế): Graph, Doc Aggregation & Submission

Bạn phụ trách **P3-01 → P3-17** trong `Chia việc.xlsx` (sheet `Checklist`): mapping
chunk→doc, doc aggregation/selector, consistency, submission generator/validator/ZIP,
medical graph (schema → NER → normalization → store → GraphRetriever), label audit,
official aggregation/submission, submission log và working notes.

- **Phạm vi code**: `src/p3/**` (hierarchy, doc aggregation/selector/pipeline, consistency, predictions, submission, official submission, submission log, label audit, working notes), `src/graph/**` (schema, ner, normalize, store, retriever), `configs/p3/**`, `configs/graph/**`, `scripts/{build_chunk_doc_mapping,tune_doc_selector,check_doc_chunk_consistency,audit_label_hierarchy,generate_submission,validate_submission,zip_submission,extract_entities,normalize_entities,build_graph,graph_retrieve,tune_doc_pipeline,generate_official_submission,log_submission,export_working_notes}.py`.
- **Nguyên tắc**: chỉ import P1/P2 read-only; output ghi kiểu exclusive, không ghi đè.
- **Tuỳ chọn cài**: graph cần `pip install -e ".[graph]"` (neo4j) + server Neo4j; XLSX log cần `.[xlsx]` (openpyxl).

## 📋 Bảng công việc P3 (nội dung theo `Chia việc.xlsx`)

| ID | Giai đoạn | Module | Đầu việc | Deliverable | Ưu tiên | Phụ thuộc | Trạng thái (xlsx) | Trạng thái repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P3-01 | Trước 01/10 | Hierarchy | Dựng mapping chunk→doc | chunk_to_doc mapping | P0 | P1-01 | Chưa làm | Hoàn thành |
| P3-02 | Trước 01/10 | Doc aggregation | Baseline max-chunk aggregation | DocAggregator(max) | P0 | P2-01,P3-01 | Chưa làm | Hoàn thành |
| P3-03 | Trước 01/10 | Doc aggregation | Thử Top-K mean | doc_aggregation_ablation.csv | P1 | P3-02 | Chưa làm | Hoàn thành |
| P3-04 | Trước 01/10 | Doc selector | Tune threshold/fallback doc | best_doc_selector.yaml | P0 | P3-02 | Chưa làm | Hoàn thành |
| P3-05 | Trước 01/10 | Consistency | Kiểm tra consistency doc/chunk | consistency_report.csv | P1 | P3-04 | Chưa làm | Hoàn thành |
| P3-06 | Trước 01/10 | Submission | Tạo submission generator | make_submission.py | P0 | P3-04 | Chưa làm | Hoàn thành |
| P3-07 | Trước 01/10 | Submission | Tạo validator + ZIP | validate_submission.py + zip_submission.py | P0 | P3-06 | Chưa làm | Hoàn thành |
| P3-08 | Trước 01/10 | Graph schema | Thiết kế medical graph schema | graph_schema.md | P1 | — | Chưa làm | Hoàn thành |
| P3-09 | Trước 01/10 | Entity extraction | Prototype medical NER/extraction | entities.jsonl | P1 | P3-08 | Chưa làm | Hoàn thành |
| P3-10 | Trước 01/10 | Normalization | Entity normalization | normalized_entities.jsonl | P1 | P3-09 | Chưa làm | Hoàn thành |
| P3-11 | Trước 01/10 | Graph indexing | Build graph prototype | Graph store + query API | P1 | P3-10 | Chưa làm | Hoàn thành (Neo4j + in-memory) |
| P3-12 | Sau 01/10 | Graph retrieval | GraphRetriever | GraphRetriever API | P1 | P3-11,P1-12 | Chưa làm | Hoàn thành code (chờ P1-12) |
| P3-13 | Ngày 01/10 | Label audit | Kiểm tra quan hệ ground-truth doc/chunk | label_hierarchy_report.md | P0 | P1-12 | Chưa làm | Hoàn thành code (chờ P1-12) |
| P3-14 | Sau 01/10 | Official aggregation | Doc aggregation trên data thật | best_doc_pipeline.yaml | P0 | P3-13,P2-11 | Chưa làm | Hoàn thành code (chờ data thật) |
| P3-15 | Sau 01/10 | Official submission | Generate official JSON/ZIP | submission_<run_id>.zip | P0 | P3-07,P2-12,P3-14 | Chưa làm | Hoàn thành code (chờ data thật) |
| P3-16 | Leaderboard | Submission log | Ghi mọi lần submit | submission_log.xlsx/csv | P0 | P3-15 | Chưa làm | Hoàn thành |
| P3-17 | Cuối cuộc thi | Working notes | Chuẩn bị working notes paper | working_notes_outline.md + figures/tables | P1 | P1-17,P2-14,P3-16 | Chưa làm | Hoàn thành skeleton + generator |


## 🎯 Chi tiết thực hiện & tiêu chí hoàn thành

- **P3-01 Dựng mapping chunk→doc** — Tạo parent mapping ổn định từ prototype; validate mỗi chunk có đúng một doc. *Tiêu chí*: không chunk mồ côi; không một chunk thuộc nhiều doc. → `src/p3/hierarchy.py`, `scripts/build_chunk_doc_mapping.py`.
- **P3-02 Baseline max-chunk aggregation** — Tính doc score = max chunk reranker score theo parent doc. *Tiêu chí*: từ scored chunks sinh đúng danh sách doc score; deterministic. → `src/p3/doc_aggregation.py`.
- **P3-03 Thử Top-K mean** — So sánh max, mean(top2), mean(top3), weighted max+mean. *Tiêu chí*: có F2_doc cho từng phương pháp; chọn baseline. → `doc_aggregation_ablation.csv`.
- **P3-04 Tune threshold/fallback doc** — Tune `threshold_doc`, `fallback_doc`, `max_doc` độc lập với chunk selector. *Tiêu chí*: tối ưu F2_doc; không dùng chung threshold chunk một cách mặc định. → `best_doc_selector.yaml`.
- **P3-05 Kiểm tra consistency doc/chunk** — Báo trường hợp chunk được chọn nhưng parent doc không được chọn và ngược lại; chưa hard-code khi chưa biết label rule. *Tiêu chí*: mọi inconsistency được định lượng; có policy cấu hình được (`none`/`prune_chunks`/`add_parent_docs`/`prune_docs`). → `consistency_report.csv`.
- **P3-06 Tạo submission generator** — Sinh JSON đúng schema BTC: `id`, `relevant_docs`, `relevant_chunks`; sort/dedup ID; đủ toàn bộ query. *Tiêu chí*: JSON pass validator; không thiếu query/trường; ID không lặp. → `src/p3/submission.py`, `scripts/generate_submission.py`.
- **P3-07 Tạo validator + ZIP** — Kiểm tra ID tồn tại, duplicate, thiếu query, thiếu field; ZIP chỉ chứa đúng 1 JSON ở root. *Tiêu chí*: có test invalid case; file ZIP đạt toàn bộ rule BTC. → `scripts/validate_submission.py`, `scripts/zip_submission.py`.
- **P3-08 Thiết kế medical graph schema** — Node tối thiểu: Document, Chunk, Disease, Drug, Symptom, Treatment; edge: HAS_CHUNK, MENTIONS, TREATS/HAS_SYMPTOM khi đủ bằng chứng. *Tiêu chí*: schema rõ node/edge/property/provenance; không tạo relation chỉ vì co-occurrence. → `docs/graph_schema.md`, `src/graph/schema.py`.
- **P3-09 Prototype medical NER/extraction** — Extract disease/drug/symptom/treatment từ chunk; lưu surface, type, canonical candidate, confidence. *Tiêu chí*: có sample audit thủ công; giữ provenance `chunk_id`. → `entities.jsonl`, `src/graph/ner.py`.
- **P3-10 Entity normalization** — Gom alias Việt/Anh và biến thể viết tắt về canonical entity; ưu tiên dictionary/ontology khi có. *Tiêu chí*: giảm node duplicate; giữ alias gốc và canonical name. → `normalized_entities.jsonl`, `src/graph/normalize.py`.
- **P3-11 Build graph prototype** — Nạp Document-Chunk-Entity và provenance; hỗ trợ query entity→chunk/doc. *Tiêu chí*: từ entity truy ra được candidate chunk/doc và evidence. → `src/graph/store.py` (Neo4j + in-memory).
- **P3-12 GraphRetriever** — Parse entity từ query, normalize, retrieve chunk/doc qua node/edge; trả candidate theo schema Người 1. *Tiêu chí*: output tương thích P1-11; có Recall@K độc lập và ablation. → `src/graph/retriever.py`.
- **P3-13 Kiểm tra quan hệ ground-truth doc/chunk** — Đo xem relevant_chunk có luôn suy ra parent relevant_doc không; và doc relevant có bắt buộc có labeled chunk không. *Tiêu chí*: chốt được có/không dùng hierarchical constraint. → `label_hierarchy_report.md`, `scripts/audit_label_hierarchy.py`.
- **P3-14 Doc aggregation trên data thật** — Re-tune max/top-k/direct-doc fusion nếu có doc retriever; tune F2_doc. *Tiêu chí*: có holdout score và ablation rõ ràng. → `best_doc_pipeline.yaml`, `scripts/tune_doc_pipeline.py`.
- **P3-15 Generate official JSON/ZIP** — Sinh file cho toàn test; validate ID; bảo đảm đúng 1 JSON trong ZIP. *Tiêu chí*: pass validator 100%; lưu config/commit tương ứng. → `submission_<run_id>.zip`, `src/p3/official_submission.py`.
- **P3-16 Ghi mọi lần submit** — Lưu run_id, git commit, config, model, local score, public/private score, ghi chú; đặc biệt private chỉ 5 lượt. *Tiêu chí*: mỗi submission truy ngược được toàn bộ cấu hình; không submit mù. → `submissions/submission_log.csv` (+`.xlsx`), `scripts/log_submission.py`.
- **P3-17 Chuẩn bị working notes paper** — Tổng hợp architecture, data prep, ablation, metric, kết quả, error analysis và contribution của từng thành viên. *Tiêu chí*: đủ thông tin để viết paper theo yêu cầu BTC; số liệu khớp experiment log. → `docs/working_notes_outline.md`, `scripts/export_working_notes.py`.

## 🧪 Kiểm thử
- P3 có bộ test riêng: `tests/test_p3_{hierarchy,doc_aggregation,doc_selector,consistency,label_audit,submission,graph,graph_store,doc_pipeline,official_submission,submission_log,working_notes}.py`.
- Chạy: `python -m pip install -e ".[test]"` rồi `python -m pytest -q` (graph dùng fake driver; Neo4j thật cần server).

## 🤝 Phối hợp
- Nhận gói P2→P3 (`reranked.jsonl`, `best_chunk_selector.yaml`, `p2_selected_chunks.json`, `registry.json`, `labels.json`) để chạy doc aggregation/selection/submission.
- Bàn giao `GraphRetriever` (P3-12) cho **Người 1** dùng ở P1-16; `best_doc_pipeline.yaml` cho **Người 2** đối chiếu P2-12.
- Dữ liệu/label chính thức (P1-12) và định nghĩa metric BTC là điều kiện để chốt P3-12/13/14/15; trước đó mọi số liệu phải ghi rõ là prototype/weak-label.

