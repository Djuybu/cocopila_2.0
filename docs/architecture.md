# Architecture

Pipeline: **data → retrieval → reranking → scoring → evaluation → submission**.

## Boundaries

| Package | Responsibility | Must not do |
| --- | --- | --- |
| src/data | Canonical loading, source field adaptation, identity checks, mappings, index writing | Modify raw input, invent official IDs |
| src/retrieval | BM25, BGE/Qdrant, candidate union, RRF/CC, optional query variants | Rerank pairs or choose final thresholds |
| src/reranking | Score query/text pairs with BGE cross-encoder | Threshold or fallback |
| src/scoring | Independent chunk/document selection | Load models |
| src/evaluation | Set metrics, ranking metrics, macro per query | Infer competition scoring rules |
| src/submission | Official-ID conversion, corpus validation, single-file ZIP | Accept unverified internal IDs |
| src/pipeline | Connect stages, config/run artifacts, CLI dispatch | Implement retrieval or metric algorithms |
| scripts | Import and invoke CLI | Business logic |

The original implementation consisted of interfaces raising NotImplementedError.
The new runnable baseline adds BM25Okapi (rank-bm25), Qdrant adapters and
cross-encoder adapters using the originally selected BGE models. No working
algorithm or preprocessing implementation was replaced.

## Data contracts

Documents require string doc_id. Chunks require chunk_id, doc_id, text.
Queries require id, text. Additional metadata is retained. When chunk_id is an
internal rechunk ID, official_chunk_id must identify its official parent chunk.
An official chunk may have several internal chunks but only one document.
The authoritative official IDs must come from the organizer's corpus; no regex
can establish their provenance.

Preparation preserves text without normalization, validates unique IDs
and parent documents, then writes canonical JSON and two mandatory mappings.
JSON, JSONL and optionally Parquet inputs are supported. Preparation does not
silently segment, normalize, deduplicate, rechunk or split the dataset.
Optional data.splits.<name>.data_path inputs are checked for document leakage.

The opt-in triplet path in `src/data/prototype.py` deduplicates normalized
query/positive pairs (NFC and whitespace as identity keys, original text kept),
merges negatives and removes contradictory positives-as-negatives. It records
every source row reference. Same text in different parent documents retains both
parents and joins their split component rather than discarding provenance.
Shared titles, duplicate content, duplicate queries and multi-positive queries
must stay in one split component. Seeded assignment happens on these components;
cross-split negative references are removed and counted.

The separately approved MMedC path streams bounded TXT prefixes out of language
ZIP archives. It creates explicitly namespaced prototype IDs and weak
adjacent-span labels, excluding query characters from candidate chunks. Local
ZIPs are size/SHA256 verified; remote sampling records archive revision/hash,
member name/CRC and sampled byte-prefix hash. No archive-wide extraction,
medical normalization or invented organizer IDs is performed.

A retriever returns dictionaries containing chunk_id, doc_id, score (float),
rank (one-based integer), source; text and metadata are retained. BM25 index
snapshots store corpus plus tokenizer/algorithm parameters in JSON, rebuilding
the in-memory BM25 index on load. No historical pickle implementation existed.

Dense indexes store UUID Qdrant point IDs only as storage keys; official/internal
chunk IDs remain intact in payload. Metadata binds an index to its model,
dimension, text field, maximum sequence length and corpus fingerprint.

## Candidate generation

CandidateGenerator receives named BaseRetriever instances and independent top_k
limits. An optional query_expander callable supplies extra query strings. Query
variants fuse within a source, then sources fuse together. No model-based query
expander is implemented or enabled in the shipped YAML configs.

RRF sums 1 / (rrf_k + rank) for each source; raw BM25 and cosine scores are never
compared. Repeated chunk IDs in one source count once. Conflicting document IDs
for the same chunk are rejected. CC min-max normalizes each source before applying
alpha * dense + (1-alpha) * BM25. Constant-score lists normalize to zero.
Union mode retains first appearance and is order-based, not score-comparable.
Named union/RRF outputs additionally retain `source_ranks`, `source_scores` and
flat `bm25_rank`, `bm25_score`, `dense_rank`, `dense_score` when present.

Persisted candidate artifacts use `medical-rag-candidates-v1`: one JSONL record
per query, with a nested `candidates` list. Each candidate contains query/chunk/doc
IDs, text, final rank/score, source, explicit nullable BM25/dense evidence and
fused_score. `src/retrieval/schema.py` standardizes only artifact fields; retriever
APIs and fusion algorithms remain unchanged. Missing sources are null, not fake
zero scores. Runtime validation checks identity, coverage and source provenance.

`src/pipeline/handoff.py` exports an existing benchmark method as a portable P2
input bundle: candidates, queries, labels, registry, reranker config and a
versioned SHA256 manifest. Original text/IDs and source fingerprints are checked
before export. No original dataset/index paths are needed for received-bundle
reranking; checksum/schema checks run before model loading. Weak-label/prototype
warnings travel with the bundle. See [p1_p2_handoff.md](p1_p2_handoff.md).

## Reranking and selection

The reranker retains retrieval score and adds rerank_score. The pipeline sends
all candidates to reranking and does not truncate there. The legacy top_k argument
remains available for old API callers, while the canonical default returns all.
Scores are the cross-encoder backend's output, not guaranteed calibrated
probabilities. Thresholds are null in baseline configs until calibrated on val.

The pipeline instantiates adapters through `create_reranker`; Qwen3 is selected
in exp003 and new P1 handoffs, while old configs without a type still select BGE.
Qwen uses Sentence Transformers >=6.1's native prompt/chat template and accepts
max_length, dtype and revision from YAML. Original query text is not prefixed
with the instruction. Reranked candidate coverage, IDs/text/evidence and finite
descending scores are checked before writing each query record.

Received handoffs can use a reranker YAML override only with a separate output
run. Input sidecars remain untouched, effective config is snapshotted and the
original manifest is saved as input_manifest.json, not re-used as checksums for
the new config. `benchmark_reranking` evaluates the same candidate pool and labels
before/after, with binary chunk ranking metrics and shared-K chunk/doc recall.
It does not regenerate a corpus, queries or qrels. P2's earlier controlled
heuristic benchmark is retained under notebooks/legacy rather than removed.

Before selection, internal rechunks collapse to official chunks using their
highest score. This prevents duplicate mappings from consuming the output limit
or reducing fallback coverage.

Chunk branch: threshold → fill to fallback minimum → cap at chunk_max.
Document branch independently sees all scored official chunks:
group → aggregate → document threshold → fallback → doc_max.

Aggregations:
- max (legacy alias max_p): maximum chunk score.
- mean_top_k (legacy alias top_k_mean): mean of up to K best chunks.
- weighted: weight * max + (1-weight) * mean_top_k.

No requirement is imposed that every selected document has a selected chunk:
the branches have independent thresholds as specified.

## Evaluation

Precision/Recall/F1/F2 use ID sets per query. F2 = 5*TP/(4*|truth|+|prediction|).
Macro is the arithmetic mean of per-query metrics, computed separately for
documents and chunks; it is not F2 of averaged precision and recall.
Empty denominators use configurable zero_division, default 0.
Precision@K divides by K (including short result lists). Recall@K divides by the
number of relevant IDs. Ranking duplicates are removed while preserving order.
Evaluation requires identical query coverage in predictions and ground truth.

These are explicit local conventions. Confirm them against the actual
competition rules before treating the metric as an official leaderboard score.

`candidate_recall.py` is a separate before-reranking harness: both chunk and
document Recall@K use the first K unique official chunks as the common budget.
It saves per-query recall, missing IDs and fully missed query lists. This is
distinct from document ranking metrics on already aggregated document lists.
The benchmark runner compares independent source rankings, first-appearance
union and RRF k sweeps on exactly one corpus/query/label fingerprint. It measures
uncached retrieval after warmup and separately records index/startup/inference
RAM/VRAM. Caches bind to corpus, config, query and K; cache lookup times must not
be presented as model inference speed. Weak labels propagate into every report.

## State and reproducibility

Paths ending in _path or _dir resolve relative to the YAML file declaring them.
extends supports recursive deep merge and rejects cycles. Run snapshots contain
absolute resolved paths so later stage commands do not depend on working directory.
Each run records config.yaml, registry.json, queries.json, run.log and stage outputs.
The registry snapshot lets packaging validate without re-reading a changed corpus.
Preparation, indexes, runs, result files and ZIP archives use exclusive creation.
A failed stage may leave diagnostic partial artifacts; preserve it and retry with
a new run_name/index path. No automatic cleanup or overwrite is performed.

All model imports are lazy. Pure CPU BM25, metrics and submission do not load
torch, Qdrant or generation dependencies. Install the package in editable mode;
there is no sys.path manipulation in scripts.

## Retained extensions

config/Settings and prompts retain their original values and environment behavior.
Legacy data/bm25_index and data/qdrant_db remain untouched; YAML uses artifacts/.
src/generation, src/multilingual, preprocessing/context-window interfaces and GPU
monitoring remain unfinished. See migration.md and graph_schema.md.
