# Mai Ngọc Duy — kết quả P1-01 đến P1-10

Mapping công việc: [p1\_retrieval\_plan.md](p1_retrieval_plan.md).

## Dữ liệu và giới hạn bắt buộc

Nguồn [Henrychur/MMedC](https://huggingface.co/datasets/Henrychur/MMedC), revision
`6629f0a54ae73b6f32d42476b2b9e90ff96035fd`. Đây là corpus TXT pretraining,
**không có query/qrels retrieval thật**. Người dùng đã chấp thuận prototype với
weak labels; không diễn giải điểm dưới đây thành điểm cuộc thi.

- Chỉ tải Chinese, English, Japanese, French; raw ZIP không bị sửa/giải nén hàng loạt.
Cả bốn ZIP đã tải xong và xác minh size/SHA256; manifest ở
`data/raw/mmedc/download_manifest.json` (19.55 GiB tổng cộng).
- Lấy mẫu 16 documents/ngôn ngữ, seed 42; đọc prefix giới hạn 8.000 ký tự/document.
- Chỉ lấy member TXT, bỏ `__MACOSX` và `cultural_filtered_data_used` khi lấy mẫu;
các ZIP tải đầy đủ vẫn giữ nguyên mọi member.
- Query: 180 ký tự đầu. Weak positive: chunk liền kề đầu tiên sau query.
- Chunk: 600 ký tự, overlap 100; mọi chunk bắt đầu sau query span.
- 64 documents, 1.024 chunks, 64 queries; text query/chunk đều không trùng trên mẫu này.
- Train/val/test = 51/7/6 documents và queries; không leakage theo document.
- Sampling seed và split seed tách riêng. Seed split 1 được chọn **chỉ theo coverage
ngôn ngữ**, không theo recall: val có Chinese 1, English 3, Japanese 1, French 2.
- Seed split 42 ban đầu thiếu Chinese/French trong val. Dataset/run đó được giữ
nguyên; kết quả chính dùng version `mmedc_p1_fourlang`.
- Corpus 1.024 chunks gồm tất cả document splits làm không gian retrieval chung.
Không model nào được fine-tune trên các query này.

Positive liền kề không chắc relevant; chunk khác cùng document có thể relevant
nhưng bị tính là false negative. Prefix query không mô phỏng câu hỏi của bác sĩ;
overlap và cùng-topic làm bài toán proxy dễ. 7 queries quá ít cho model selection,
không có significance test hay ước lượng chất lượng ngoài tập mẫu.
Leakage checks dùng source-file document IDs và exact normalized text; chưa
bảo đảm loại near-duplicates, các phần khác của cùng sách hay bài dịch song ngữ.
IDs `prototype_*` là IDs nội bộ của **prototype**, không phải ID chính thức BTC.
Mapping identity trong prototype không biến chúng thành competition IDs.

## Deliverables


| Task  | Kết quả                                                                                                                  |
| ----- | ------------------------------------------------------------------------------------------------------------------------ |
| P1-01 | Typed schemas + `docs/schemas/prototype_records.json`; adapter anchor/positive/meta, reversible source IDs               |
| P1-02 | Dedup query-positive theo NFC/whitespace identity, giữ text gốc; merge negatives, bỏ contradictions; before/after report |
| P1-03 | Seeded connected-component split: document/title/content/shared positive/query; lọc negative cross-split và đếm          |
| P1-04 | Canonical documents/chunks/queries/labels JSON + `chunk_corpus.jsonl`, split files và mappings                           |
| P1-05 | BM25 Top-K configurable, opt-in CJK tokenizer, cache gắn corpus/config/query/K; benchmark tất cả val queries             |
| P1-06 | BGE-M3 normalized dense vectors, Qdrant cosine/dot config, model revision/dtype/index metadata                           |
| P1-07 | E5 config với query/passage prefixes, cùng corpus/split/K/token limit; runner ghi recall/latency/RAM/VRAM                |
| P1-08 | Union dedup chunk ID, bảo toàn parent và bm25/dense rank/score provenance                                                |
| P1-09 | RRF k=20/60/100 dựa trên rank, tie-break theo chunk ID; delta recall so với union cùng K                                 |
| P1-10 | `evaluate_retrieval.py`: chunk/doc Recall@20/50/100/200, macro/per-query, missing IDs và fully missed queries            |


Triplet adapter đã được test bằng triplets có nhãn rõ ràng trong fixtures; MMedC
không được giả vờ là dataset triplet có supervision. Cùng text nhưng khác parent
document vẫn giữ cả parent và nối component split để không mất provenance.

## Benchmark đã chạy

GPU RTX 3050 Laptop 4GB; Python 3.14.4; torch 2.14.0; transformers 5.17.0;
sentence-transformers 6.1.0; qdrant-client 1.19.1; numpy 2.5.3; rank-bm25 0.2.2.
Hai dense models dùng FP16, batch 1, 1.024 dimensions, max 512 tokens,
normalized embeddings. Đây là giới hạn prototype cho GPU, không phải benchmark
long-context 8.192 tokens của BGE-M3.

[BGE-M3](https://huggingface.co/BAAI/bge-m3) revision
`5617a9f61b028005a4858fdac845db406aefb181`, không thêm instruction prefix.
[Multilingual E5 large](https://huggingface.co/intfloat/multilingual-e5-large)
revision `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`, dùng `query:` / `passage:`
kể cả query không phải English. Config baseline cũ ngoài P1 được giữ nguyên.

Kết quả hiện có trên split fourlang (weak macro recall):


| Method            | Chunk R@20 | R@50   | R@100  | R@200  | Doc R@20/50/100/200   | Mean latency/query |
| ----------------- | ----------: | ------: | ------: | ------: | --------------------- | ------------------: |
| BM25 unicode\_cjk | 1.0000     | 1.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 1         | 22.31 ms           |
| BGE-M3 cosine     | 0.8571     | 0.8571 | 1.0000 | 1.0000 | .8571 / .8571 / 1 / 1 | 69.43 ms           |
| BGE-M3 dot        | 0.8571     | 0.8571 | 1.0000 | 1.0000 | .8571 / .8571 / 1 / 1 | 73.67 ms           |
| E5 large cosine   | 0.8571     | 1.0000 | 1.0000 | 1.0000 | .8571 / 1 / 1 / 1     | 91.68 ms           |
| BM25+BGE union    | 1.0000     | 1.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 1         | 93.12 ms           |
| BM25+BGE RRF k=60 | 1.0000     | 1.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 1         | 93.48 ms           |


BGE dot đã chạy xong; recall ở cả bốn cutoffs bằng cosine trong lần đo này.
Không suy rộng thành mọi raw score/ranking đều bằng nhau, nhất là với FP16.


| Retrieval phase | Peak RSS  | Peak torch CUDA allocated |
| --------------- | ---------: | -------------------------: |
| BM25-only       | 86.52 MiB | N/A (CPU-only)            |
| BM25+BGE cosine | 2.526 GiB | 1.070 GiB                 |
| BM25+BGE dot    | 2.647 GiB | 1.070 GiB                 |
| BM25+E5 cosine  | 2.636 GiB | 1.056 GiB                 |


Lựa chọn baseline tạm cho candidate budget 100/200: giữ BGE-M3 cosine trong
P1 base config và BM25+BGE RRF k=60 làm hybrid mặc định. BGE và E5 cùng proxy
R@20/100/200; BGE nhanh hơn trong số đo này, RSS thấp hơn, và giữ model hiện có.
**E5 có R@50 tốt hơn (7/7 so với 6/7)**; nếu ưu tiên budget 50 thì đây là tín hiệu
cần kiểm tra lại bằng nhãn thật. Không đổi model ở legacy pipeline hay coi lựa
chọn này là kết luận chất lượng. k=60 được giữ vì các k sweep ngang recall trên
proxy, không phải vì đã tìm ra hyperparameter tối ưu.

RRF k=20/60/100 của BGE cosine/dot và E5 đều đạt 1.0 ở cả bốn cutoffs. Delta so với union là
0.0 cho chunk/doc. Đây là quan sát trên proxy, **không phải bảo đảm toán học RRF
không giảm recall**. Union giữ first appearance BM25 trước, không có fused score
ranking, và có thể chứa tối đa 400 pooled chunks; so sánh tại cùng K chunks.
Document Recall@K lấy parent docs của K unique chunks đầu, không lấy K unique docs.
BGE miss weak positive của query Japanese `prototype_query_153f937569d4ce1a8e8c286e`
tại K=20/50 nhưng tìm thấy tại K=100. Không method nào miss hoàn toàn tại K=200
trong các run trên. E5 cũng miss query Japanese đó tại K=20, nhưng tìm được tại
K=50, sớm hơn BGE trên weak label này.

Latency uncached, CUDA synchronized, sau warmup, không gồm startup/indexing;
results vẫn được lưu cache. Không so cache hit với model inference. Mỗi timing
chỉ có 7 samples và chạy trên máy đang tải dữ liệu: dùng làm số đo vận hành,
không phải microbenchmark cô lập. RAM là sampled process RSS; VRAM là torch
allocated/reserved, không phải toàn bộ GPU. Hybrid phase gồm cả BM25 + dense.
Startup/index peaks được ghi riêng trong `benchmark.json`.

Fingerprints dùng chung cho các run chính:

```text
corpus  ad0b3cc77011c53b51ea6090f6c143536eae2d2c21f366b7f14898605dcbe8ac
queries 763a4e18a0a5f2006cf56f8016842e5a7048ef409aa16004d8451358df614d7f
labels  9ac23a70805560d5e0df4b31e27b0cb9bef45ba3ef78258cd0cee2f0932b3999
```

Run artifacts nằm trong `outputs/p1_*_fourlang/`: config snapshot, registry,
benchmark JSON, comparison CSV, từng method có candidates JSONL và recall JSON.
Harness CLI riêng đã chạy ở `outputs/p1_candidate_recall/`.
CSV nhỏ có thể track: `experiments/retrieval_benchmark_log.csv`.
Bản tổng hợp số đo nhỏ, không chứa corpus/weights/candidates:
[p1\_retrieval\_summary.json](../experiments/ablations/p1_retrieval_summary.json).
Git HEAD ghi trong runs là `34d6aae` (base commit); code P1 chưa commit tại thời
điểm đo, nên HEAD không phải bằng chứng source benchmark đã nằm trong commit đó.
Implementation P1 sau đó được lưu trong commit `c59dfa2`.

## Chạy lại

Xem [README](../README.md#p1-01p1-10-mmedc-weak-label-prototype) cho download,
preparation, BM25/BGE/E5/dot và harness commands. Cài editable package với extras
`.[dense,benchmark,test]` trước khi gọi scripts. Run/index/dataset outputs không
bị ghi đè: dùng `--run-name` mới; preparation/index retry cần đổi output paths.

## Kiểm chứng và phần còn thiếu

- `python -m pytest -q`: 104 passed, 5 skipped; skips thuộc segmentor cũ chưa implement.
- `python -m compileall -q src scripts config`: pass.
- `git diff --check`: pass.
- BM25/BGE cosine/BGE dot/E5 benchmark thực tế + standalone recall harness: pass.
- Schema, document leakage và query-span exclusion checks trên prototype thực tế: pass.
- Cả 4 raw ZIP size/SHA256 verified.
- Dataset/model/index/cache/output lớn đều bị Git ignore; không đưa vào commit.
- TODO: thay weak qrels bằng retrieval labels thật trước quyết định model/hyperparameters.
- TODO: mapping organizer IDs khi có corpus BTC; không nộp prototype IDs.
- P1-11 đã bổ sung sau batch benchmark: [gói bàn giao Người 2](p1_p2_handoff.md).
  Export không thay số đo retrieval ở trên; giữ nguyên weak labels và prototype IDs.
- TODO ngoài scope: P1-12+, training/reranking/threshold; GraphRetriever chưa implement.
