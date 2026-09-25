# Medical Retrieval / RAG Competition

Python pipeline cho truy xuất tài liệu y khoa:

**data → retrieval → reranking → scoring → evaluation → submission**

Baseline BM25 chạy trên CPU. Dense dùng BGE-M3 + Qdrant; reranking dùng
bge-reranker-v2-m3. Model được nạp khi cần. Generation bằng Qwen, multilingual và
các interface preprocessing cũ được giữ lại nhưng chưa triển khai.

Repo chưa chứa dataset hoặc model weights. Các lệnh dưới đây cần dữ liệu đầu vào
theo schema mô tả; tests có corpus nhỏ riêng để kiểm tra pipeline.

## Repository structure

```text
configs/
  data/                  # Raw preparation + prepared dataset paths
  retrieval/             # BM25, BGE-M3, fusion
  reranker/              # BGE cross-encoder
  experiments/           # exp000 BM25, exp001 dense, exp002 hybrid, exp003 full
data/
  raw/{prototype,competition}/
  interim/{normalized,deduplicated,entity_extracted}/
  processed/             # Canonical data; generated per dataset
  mappings/              # Official chunk/document and internal/official mappings
src/
  data/                  # Loader, adapter, schema, split checks, indexing
  retrieval/             # Interface, BM25, dense, fusion, candidate generator
  reranking/             # Model scoring only
  scoring/               # Chunk selection and document aggregation
  evaluation/            # Retrieval metrics, F-beta, macro evaluator
  submission/            # Generator, validator, ZIP
  pipeline/              # Stage orchestration and CLI parsing
  utils/                 # Config, I/O, logging, compatibility exports
  ingestion/             # Old import aliases
  generation/            # Preserved unfinished optional RAG components
  multilingual/          # Preserved unfinished NER/translation/SHIFT components
scripts/                  # Thin entrypoints
artifacts/{indexes,models,cache}/
outputs/<run_name>/        # Config, registry, candidates, scores, predictions, metrics
experiments/              # Manual log and ablation notes
submissions/{public,private}/
tests/
docs/                     # Architecture, migration, experiment notes, graph TODO
config/                   # Preserved Settings and prompt templates
notebooks/                # Preserved; no fake notebooks generated
```

Mapping từng file: [docs/migration.md](docs/migration.md).
Kiến trúc và quy ước scoring: [docs/architecture.md](docs/architecture.md).
Tài liệu DEV_A/B/C và notes/ được giữ như ghi chú lịch sử; kiến trúc mới ở docs/.

## Setup environment

Python >= 3.10:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

Cho dense retrieval và reranking:

```bash
python -m pip install -e ".[dense]"
```

Đọc Parquet: `python -m pip install -e ".[parquet]"`.
requirements.txt giữ các dependencies đầy đủ của thiết kế RAG cũ, gồm generation;
không cần cài toàn bộ để chạy BM25 competition baseline.

Cài editable package là bước bắt buộc trước khi gọi scripts. Không sửa sys.path.
Các lệnh ví dụ chạy tại repository root; dùng đường dẫn tuyệt đối cho --config
hoặc --run-dir nếu gọi từ nơi khác.

## Prepare data

Đặt dữ liệu gốc tại data/raw/prototype/:

- documents.json: array record có `doc_id` là string.
- chunks.json: array record có `chunk_id`, `doc_id`, `text`.
- queries.json: array record có `id`, `text`.
- labels.json (nếu có): schema giống submission, dùng ID chính thức.

Ví dụ record:

```json
{"chunk_id": "d_12345678_c000", "doc_id": "d_12345678", "text": "Nội dung tài liệu"}
```

Nếu rechunk nội bộ, bắt buộc cung cấp `official_chunk_id`:

```json
{"chunk_id": "internal_001", "official_chunk_id": "d_12345678_c000", "doc_id": "d_12345678", "text": "Nội dung"}
```

ID chính thức phải lấy từ corpus BTC. Không suy đoán hoặc tạo ID thay thế.
Tên field khác có thể map qua `data.fields` trong YAML. Loader hỗ trợ JSON, JSONL,
Parquet; sửa đường dẫn tương ứng trong config.

```bash
python scripts/prepare_data.py --config configs/data/prototype.yaml
```

Lệnh kiểm tra ID trùng, parent document và ghi:

```text
data/processed/prototype/{documents,chunks,queries}.json
data/mappings/prototype/chunk_to_doc.json
data/mappings/prototype/internal_to_official_id.json
```

Không thay text, không tự segment/deduplicate/rechunk. Raw được giữ nguyên.
Output đã tồn tại sẽ báo lỗi: dùng thư mục dataset version mới để chạy lại.
Có thể thêm `data.splits.train.data_path`, `data.splits.val.data_path` trỏ tới
records chứa doc_id để kiểm tra leakage theo document.

Competition preparation dùng configs/data/competition.yaml. Sau đó tạo experiment
config trỏ đến processed/competition và mappings/competition; không tự dùng
prototype index cho corpus mới.

## Run BM25 baseline

```bash
python scripts/build_bm25_index.py --config configs/retrieval/bm25.yaml
python scripts/run_full_pipeline.py --config configs/experiments/exp000_bm25.yaml
```

Output ở outputs/exp000/. BM25 dùng rank-bm25 BM25Okapi. Tokenizer baseline
tách whitespace, giữ nguyên hoa/thường theo config. Có thể dùng text_key:
segmented_text khi corpus đã được tách từ. Snapshot index là JSON dưới artifacts/,
không phải pickle. BM25-only không tải model.

## Run dense baseline

Cài extra dense và có model/network cache cùng thiết bị phù hợp. Đổi device trong
YAML nếu muốn chạy CPU; mặc định giữ cuda như cấu hình cũ.

```bash
python scripts/build_dense_index.py --config configs/retrieval/dense_bge_m3.yaml
python scripts/run_full_pipeline.py --config configs/experiments/exp001_dense.yaml
```

Index Qdrant mới được ghi tại artifacts/indexes/dense/prototype_bge_m3/.
Model, dimension, batch size, text field, query prefix và max sequence length
được lấy từ YAML. Không tự xóa index cũ.

## Run hybrid retrieval

Sau khi đã build cả BM25 và dense index:

```bash
python scripts/run_full_pipeline.py --config configs/experiments/exp002_bm25_dense.yaml
```

Hybrid dùng RRF theo rank từng nguồn. CC cũng có sẵn qua fusion.method: cc,
với đúng hai nguồn bm25 và dense. CandidateGenerator chỉ retrieval/fusion,
không chứa reranker.

## Run reranking

Chạy từng stage của exp003 (chưa chạy full exp003 trước đó):

```bash
python scripts/run_retrieval.py --config configs/experiments/exp003_full.yaml
python scripts/run_reranking.py --run-dir outputs/exp003
python scripts/run_prediction.py --run-dir outputs/exp003
```

Hoặc chạy ba stage trong một lệnh:

```bash
python scripts/run_full_pipeline.py --config configs/experiments/exp003_full.yaml
```

Hai cách trên là lựa chọn thay thế: một run_name chỉ được tạo một lần.
Reranker thêm rerank_score, không áp threshold. Scoring chọn chunks/docs theo
threshold, fallback tối thiểu và max output; docs hỗ trợ max, mean_top_k, weighted.
Nhiều internal chunks cùng một official chunk được gộp trước selection.

Threshold mặc định null vì điểm RRF/BM25/cross-encoder khác thang đo. Hãy chọn
threshold trên validation set; không coi raw reranker scores là xác suất đã calibrate.

## Evaluate

Ground truth phải đủ đúng tập query của run và dùng official IDs:

```bash
python scripts/run_evaluation.py --run-dir outputs/exp000 \
  --ground-truth data/raw/prototype/labels.json

python scripts/run_evaluation.py --run-dir outputs/exp000 \
  --ground-truth data/raw/prototype/labels.json --stage candidates --k 100
```

Kết quả lần lượt là metrics_predictions.json và metrics_candidates.json.
Nếu evaluation.labels_path được cấu hình, full pipeline tự đánh giá predictions;
không gọi lại lệnh ghi cùng metrics file.

Metric chính là macro F2 theo query, tách doc/chunk branch. Có Precision, Recall,
F1, F2, Recall@K, Precision@K; metric helpers cũng hỗ trợ Hit@K, MRR, NDCG.
Mặc định empty denominator = 0; Precision@K chia cho K. Rule cuộc thi chưa có
trong repo, cần đối chiếu trước khi coi đây là điểm chính thức.

## Generate submission

```bash
python scripts/make_submission.py --run exp003
```

Hoặc cho baseline CPU:

```bash
python scripts/make_submission.py --run-dir outputs/exp000
```

Tạo submissions/public/<run_name>.zip, chứa đúng predictions.json ở root.
Đổi đích bằng --submission-dir hoặc submission.output_dir trong YAML.
Validator kiểm tra đủ query, query ID trùng, đủ hai list (được rỗng), ID trùng
trong list, doc/chunk tồn tại và không có internal chunk ID.
Packaging sử dụng registry.json đã snapshot cùng run; không bỏ qua corpus validation.

Schema:

```json
[
  {
    "id": "q_0001",
    "relevant_docs": ["d_12345678"],
    "relevant_chunks": ["d_12345678_c000"]
  }
]
```

## Experiment convention

Mỗi experiment có run_name riêng, ví dụ exp014. Copy config, đổi run_name và
hyperparameters; index/data paths phải khớp corpus/model. YAML hỗ trợ extends,
deep merge; mọi *_path và *_dir tính tương đối từ file YAML khai báo giá trị đó.

```text
outputs/exp014/
  config.yaml
  registry.json
  queries.json
  run.log
  candidates.jsonl
  reranked.jsonl
  predictions.json
  metrics_predictions.json    # Khi có ground truth
  metrics_candidates.json     # Khi chạy candidate evaluation
```

Config snapshot/log chứa run ID, dataset split, model, top-k, thresholds, output
paths; evaluation ghi metrics. Không ghi đè run/index/dataset/ZIP cũ.
Nếu stage lỗi, giữ artifact để chẩn đoán và dùng run_name mới khi chạy lại.
CSV logs trong experiments/ và submissions/ dành cho ghi chú thủ công.

Dataset, index, weights, cache và generated outputs đã được ignore.
Không tạo giả các file Parquet, notebook, model config hoặc graph/training scripts.

## Compatibility and pending work

Import cũ vẫn được giữ qua alias, ví dụ:

```python
from src.retrieval.sparse_search import SparseRetriever  # alias BM25Retriever
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.aggregator import DocumentAggregator
from src.utils.validator import SubmissionValidator
```

Code mới dùng src.retrieval.bm25, src.retrieval.dense, src.reranking.bge,
src.scoring.doc_aggregation và src.submission.validator.

Legacy run_pipeline.py nhận --config / --config-path, chạy full pipeline mới.
Các cờ cũ --data-path, --query-path, --output-path, --mode chuyển sang YAML và các
entrypoint theo stage; chúng trước đây chỉ parse arguments rồi raise NotImplementedError.
Legacy pack_submission.py vẫn nhận --input/-i, --output/-o và nay cần --registry
trỏ tới outputs/<run_name>/registry.json để xác thực corpus.

Settings.from_env và prompt templates trong config/ giữ nguyên. YAML điều khiển
pipeline mới; không tự thay các giá trị YAML bằng environment settings.
Legacy data/bm25_index và data/qdrant_db được giữ, nhưng run mới dùng artifacts/.

TODO: GraphRetriever, graph schema/building, fine-tuning/hard negatives,
VietnameseSegmentor/SlidingWindowChunker, multilingual NER/translation/SHIFT,
generation và VRAM monitor chưa triển khai. Không gọi các interface này trong baseline.
Xem [graph status](docs/graph_schema.md) và [experiment notes](docs/experiment_notes.md).
