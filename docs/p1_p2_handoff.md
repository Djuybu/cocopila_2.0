# P1-11 — bàn giao candidates cho Người 2

Mai Ngọc Duy (Người 1) bàn giao candidate generation cho Mạc Duy (Người 2),
để bắt đầu P2-01. Không đổi thuật toán retrieval, không chạy baseline reranker,
không chọn threshold trong P1-11.

## Tạo và kiểm tra gói

Cài package editable trước khi chạy scripts. Người xuất cần benchmark và dữ
liệu processed/mappings tương ứng còn nguyên để đối chiếu fingerprint và text:

```bash
python -m pip install -e ".[test]"
python scripts/export_reranking_input.py --config configs/handoff/p1_to_p2.yaml
python scripts/validate_reranking_input.py --run-dir outputs/p1_p2_handoff
```

Config chọn `outputs/p1_bge_m3_fourlang/rrf_k60`: BM25 unicode_cjk + BGE-M3
cosine, validation 7 queries, tối đa 200 candidates/query, tổng 1.400 candidates.
Không retrieval lại, không thay ranking/score. Những lần xuất tiếp theo phải dùng
`output_dir` và `archive_path` mới; thư mục/ZIP cũ không bị ghi đè.

```text
outputs/p1_p2_handoff/
├── candidates.jsonl   # Một record/query, mỗi candidate chứa đủ text và ID
├── queries.json       # Query text: [{"id": "...", "text": "...", ...}]
├── labels.json        # Weak labels cho đúng split này
├── registry.json      # Query coverage + chunk/document và internal/official mappings
├── config.yaml        # Reranker config, không phụ thuộc đường dẫn corpus/index máy gửi
└── manifest.json      # Schema version, provenance, counts, SHA256 và giới hạn
outputs/p1_p2_handoff.zip
```

ZIP chứa đúng sáu file trên ở root, không có folder con. Đây là **gói input
reranking**, không phải ZIP submission (submission chỉ chứa một predictions JSON).
Output/ZIP bị Git ignore; cần chuyển ZIP riêng cùng code đã commit. Người nhận
không cần corpus MMedC ~20 GB, Qdrant/BM25 index hoặc retrieval model weights.

## Candidate contract v1

Schema version: `medical-rag-candidates-v1` trong manifest.
Schema máy đọc: [candidate_record.schema.json](schemas/candidate_record.schema.json).
Định nghĩa và kiểm tra runtime: `src/retrieval/schema.py`.
Giữ format JSONL một record/query tương thích pipeline hiện tại:

```json
{"id":"q_001","candidates":[{"query_id":"q_001","chunk_id":"d_01_c000","doc_id":"d_01","text":"Candidate text","rank":1,"score":0.03278688524590164,"source":"rrf","bm25_rank":1,"bm25_score":12.8,"dense_rank":1,"dense_score":0.81,"fused_score":0.03278688524590164,"source_ranks":{"bm25":1,"dense":1},"source_scores":{"bm25":12.8,"dense":0.81}}]}
```

Các ID trong ví dụ chỉ minh họa, không phải ID corpus thật.

- `query_id`, `chunk_id`, `doc_id`, `text` có sẵn trên từng candidate; không cần
  remap ID, tìm parent document hay join corpus để lấy text.
- `rank` là vị trí một-based trong candidate list cuối. `score` là score ranking
  hiện có; với RRF/CC nó bằng `fused_score`. Union giữ score cũ nhưng không coi
  score khác nguồn là cùng scale.
- `bm25_rank/score`, `dense_rank/score` luôn có field; nguồn không trả chunk này
  có giá trị `null`. `source_ranks/source_scores` chứa bằng chứng nguồn thực có,
  với cùng tập keys. Raw score không được normalize hay giả tạo khi export.
- `fused_score` là `null` nếu không có fusion score (BM25/dense riêng hoặc union).
- Metadata bổ sung được giữ nguyên. Query không có candidates vẫn có record với
  `candidates: []`; không được bỏ query.
- Query text lấy trực tiếp từ `queries.json` bằng `id`, như `run_reranking` đã
  thực hiện. Reranker thêm `rerank_score`; không truncate hay threshold candidates.

Exporter đối chiếu corpus/query/label fingerprints, parent và text với dữ liệu
nguồn. Validator kiểm tra toàn bộ query, duplicate IDs, parent mappings, field
bắt buộc, rank/score hợp lệ, source provenance và SHA256. Khi có manifest,
`run_reranking` tự validate trước khi nạp model hoặc tạo output.

## Người 2 chạy P2-01

Sau khi nhận code và ZIP, giải nén vào thư mục run mới; có thể đổi vị trí/thư mục.
Cài môi trường phù hợp rồi chạy (ví dụ giải nén vào `outputs/p1_p2_handoff`):

```bash
python -m pip install -e ".[dense,test]"
python scripts/validate_reranking_input.py --run-dir outputs/p1_p2_handoff
python scripts/run_reranking.py --run-dir outputs/p1_p2_handoff
```

Output: `reranked.jsonl`, giữ query/chunk/doc IDs, text và retrieval evidence,
thêm `rerank_score` cho mọi candidate. Query/candidate sidecars không bị ghi lại.
Có thể đánh giá candidate recall hiện tại, **không phải reranker F2**, bằng:

```bash
python scripts/run_evaluation.py \
  --run-dir outputs/p1_p2_handoff --stage candidates --k 200
```

Labels path được resolve tương đối với config trong gói nên không phụ thuộc máy
gửi. Không dùng `run_prediction`/`make_submission` ngay trên gói này: chưa có
selector/submission config và chưa calibrate threshold (các việc P2 tiếp theo).

Reranker mặc định `BAAI/bge-reranker-v2-m3`, CUDA, batch size 1. Đây là config
khởi đầu, **không phải cam kết vừa mọi GPU**; cần tải model và môi trường PyTorch
phù hợp. P2-01 chịu trách nhiệm đo inference thực và cấu hình dtype/token limit
nếu cần. Không sửa file input đã có checksum; muốn config khác, xuất gói mới từ
YAML handoff khác (hoặc copy config và dùng API `run_reranking(..., reranker=...)`
để tiêm backend, ghi rõ thay đổi trong báo cáo thí nghiệm).

## Giới hạn và kiểm thử

- MMedC không có retrieval qrels; labels là `weak_adjacent_span`. 7 queries chỉ là
  smoke/prototype, không đủ kết luận chất lượng model hay chọn threshold tin cậy.
- ID hiện tại là namespace `prototype`, **không phải ID BTC**. Export giữ nguyên
  ID; mappings không biến prototype thành official IDs.
- SHA256 phát hiện artifact bị thay đổi, không phải chữ ký xác thực người gửi.
- Tests kiểm tra giữ ID/text, nullable evidence, coverage, duplicate/invalid
  fields, stale data, checksum/counts, không ghi đè, CLI ngoài repository và
  reranking sau khi mất đường dẫn dữ liệu gốc.
- Smoke trên 7 queries/1.400 candidates dùng backend giả đã kiểm tra giải nén,
  validation và bảo toàn IDs/text; **chưa benchmark model BGE reranker thật**.

P1-11 hoàn tất contract và input bàn giao. P2-01 và các bước training/threshold
của Người 2 vẫn là công việc tiếp theo; không cập nhật trạng thái Google Sheet.
