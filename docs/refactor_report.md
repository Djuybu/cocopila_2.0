# Refactor completion report

## Outcome

The repository now has a runnable retrieval competition baseline split into data,
retrieval, reranking, scoring, evaluation and submission. No tracked file was
deleted, no raw dataset was changed, and no model weights or indexes were added
to Git. Old import paths forward to canonical modules. Original README content
is archived in working_notes/README_original.md.

The pre-edit file-by-file plan and reasons are in [migration.md](migration.md).
All source algorithms previously raised NotImplementedError; the implementation
work here adds the explicitly requested baseline/interfaces/metrics/validation
rather than replacing an existing working retrieval algorithm.

## Important moves

| Old | Canonical implementation |
| --- | --- |
| src/ingestion/segmentor.py | src/data/preprocess.py |
| src/ingestion/chunker.py | src/data/chunker.py |
| src/ingestion/indexer.py | src/data/indexer.py |
| src/retrieval/sparse_search.py | src/retrieval/bm25.py |
| src/retrieval/dense_search.py | src/retrieval/dense.py |
| src/retrieval/hybrid_fusion.py | src/retrieval/fusion.py |
| src/retrieval/reranker.py | src/reranking/bge.py |
| src/retrieval/aggregator.py | src/scoring/doc_aggregation.py |
| src/utils/metrics.py | src/evaluation/retrieval_metrics.py + evaluator.py |
| src/utils/validator.py | src/submission/validator.py |
| scripts/run_pipeline.py | src/pipeline/full_pipeline.py + cli.py |
| scripts/pack_submission.py | src/submission/zipper.py + pipeline/cli.py |

## New-file responsibilities

- src/data: canonical record loading, explicit source field adaptation, identity
  validation, document-disjoint split checks, mappings and index creation.
  preprocess/chunker preserve their original unfinished interfaces.
- src/retrieval: shared candidate schema, BM25 and dense adapters, rank-based fusion,
  union/deduplication and multi-retriever candidate orchestration.
- src/reranking: common interface and BGE pair scoring without threshold selection.
- src/scoring: max/mean_top_k/weighted document aggregation and independent branch
  threshold/fallback/maximum selection.
- src/evaluation: Precision/Recall/F1/F2, ranking metrics and macro per-query reports.
- src/submission: official-ID conversion, strict corpus/query checks and ZIP validation.
- src/pipeline: index/retrieval/rerank/predict/evaluate/submission orchestration;
  scripts remain thin entrypoints.
- src/utils: YAML inheritance and file-relative paths, exclusive artifact writes,
  per-run logging.
- configs: preparation, BM25/BGE indexes, fusion, reranker, four runnable experiment
  configurations. No unsupported graph/E5/fine-tuning configurations.
- tests: contracts, edge cases, actual BM25 workflow, Qdrant integration, subprocess
  CLI tests, mapping/ZIP checks and import compatibility.
- pyproject.toml: editable packaging and optional test/dense/Parquet dependencies.
- docs: migration, architecture, experiment conventions, graph status.
- .gitkeep and package __init__.py files: only required package/storage boundaries.
- experiment/submission CSV logs: headers for manual notes, not fake experiment data.

### New files (excluding package markers and .gitkeep)

- configs/data/competition.yaml
- configs/data/prepared_prototype.yaml
- configs/data/prototype.yaml
- configs/experiments/base.yaml
- configs/experiments/exp000_bm25.yaml
- configs/experiments/exp001_dense.yaml
- configs/experiments/exp002_bm25_dense.yaml
- configs/experiments/exp003_full.yaml
- configs/reranker/bge_reranker.yaml
- configs/retrieval/bm25.yaml
- configs/retrieval/dense_bge_m3.yaml
- configs/retrieval/hybrid_rrf.yaml
- docs/architecture.md
- docs/experiment_notes.md
- docs/graph_schema.md
- docs/migration.md
- docs/working_notes/README_original.md
- experiments/experiment_log.csv
- pyproject.toml
- scripts/build_bm25_index.py
- scripts/build_dense_index.py
- scripts/make_submission.py
- scripts/prepare_data.py
- scripts/run_evaluation.py
- scripts/run_full_pipeline.py
- scripts/run_prediction.py
- scripts/run_reranking.py
- scripts/run_retrieval.py
- src/data/adapter.py
- src/data/chunker.py
- src/data/indexer.py
- src/data/loader.py
- src/data/preprocess.py
- src/data/schema.py
- src/data/split.py
- src/evaluation/evaluator.py
- src/evaluation/fbeta.py
- src/evaluation/retrieval_metrics.py
- src/pipeline/cli.py
- src/pipeline/full_pipeline.py
- src/pipeline/predict.py
- src/pipeline/reporting.py
- src/pipeline/rerank.py
- src/pipeline/retrieve.py
- src/reranking/base.py
- src/reranking/bge.py
- src/retrieval/base.py
- src/retrieval/bm25.py
- src/retrieval/candidate_generator.py
- src/retrieval/dense.py
- src/retrieval/fusion.py
- src/scoring/chunk_selector.py
- src/scoring/doc_aggregation.py
- src/submission/generator.py
- src/submission/validator.py
- src/submission/zipper.py
- src/utils/config.py
- src/utils/io.py
- src/utils/logging.py
- submissions/submission_log.csv
- tests/conftest.py
- tests/test_aggregation.py
- tests/test_config.py
- tests/test_data.py
- tests/test_dense_index.py
- tests/test_fbeta.py
- tests/test_pipeline.py
- tests/test_reranking.py
- tests/test_submission.py
- docs/refactor_report.md (this report)

## Important modified files

README.md now contains real setup/stage commands and limitations. .gitignore
excludes data, indexes, models, cache, outputs and ZIP artifacts. requirements.txt
adds PyYAML while retaining old dependencies. Old ingestion/retrieval/utils paths
are compatibility shims; retrieval/__init__.py exports canonical implementations.
scripts/run_pipeline.py and pack_submission.py now delegate to src.

The original generation/multilingual modules required moving future imports above
__author__ to make them importable. Unused eager GPU/validation imports were removed
or made lazy so BM25 does not require the full GPU stack. Their unfinished runtime
behavior is retained. Settings and prompt templates remain intact.

tests/test_retrieval.py and test_validator.py now test behavior rather than skip.
The five pending segmentation tests remain unchanged.

## Compatibility / commands

Install with python -m pip install -e ".[test]"; optional models use .[dense].
Use the commands in [README.md](../README.md), which cover prepare_data,
build_bm25_index, build_dense_index, run_retrieval, run_reranking, run_prediction,
run_evaluation, run_full_pipeline and make_submission.

Old run_pipeline --config-path is supported as an alias for --config.
Its old path/mode flags move into YAML and stage-specific commands.
pack_submission keeps --input/--output and requires --registry for corpus checks.
Python import aliases remain. Legacy Settings and old index directories are retained.

## Verification

Verified in the local .venv using Python 3.14.4:

- python -m pytest -q: **79 passed, 5 skipped**.
- All five skips are pre-existing unimplemented VietnameseSegmentor tests.
- Actual BM25 prepare/index/retrieve/select/evaluate/ZIP workflow passes.
- Every CLI --help runs, and the staged CLI workflow runs from another directory.
- Actual local Qdrant indexing, queries, filtering and dense pipeline pass using
  supplied small vectors (no model downloads).
- Reranker adapter tested with an injected scoring backend.
- Every src module imports; old and canonical class imports match.
- python -m compileall -q src config scripts tests passes.
- git diff --check passes.
- Generated data/index/model/output/ZIP paths are confirmed ignored.

Dense BGE encoding and BGE reranking with real weights/CUDA were not executed.
These still require model downloads, appropriate runtime dependencies and hardware.
No claim of competition score improvement or GPU memory performance is made.

## Remaining work

TODO: GraphRetriever not implemented yet. Graph entity/relation schema and builder
are not invented. No graph/training scripts or fake notebook/data files were added.

Existing preprocessing/context-window, generation, multilingual and VRAM monitor
interfaces remain unfinished. Fine-tuning/hard negatives, streaming large files
and actual model benchmarks are separate tasks. Empty-set/macro metric conventions
must be confirmed against the competition rules. The runnable baseline requires
real source corpus/queries; no competition dataset was present in this repository.
