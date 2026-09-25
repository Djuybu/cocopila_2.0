# Migration audit — 2026-09-25

Audit completed before source edits. The worktree was clean. All existing Python
modules, entrypoints, tests, settings, prompts and role documents were inspected.
There are no datasets, model weights or indexes: data directories contain only
`.gitkeep`. No notebooks exist beyond `notebooks/.gitkeep`.

The working implementation is Settings (environment loading and directory
creation), PromptTemplates and CLI argument parsing. All retrieval, ingestion,
generation, multilingual, metric, validation and packaging operations raise
NotImplementedError. All 15 existing tests are skipped. Therefore runnable
baselines and the explicitly requested validation/metrics require new logic;
there is no existing retrieval algorithm or preprocessing behavior to replace.

## Planned mapping (recorded before refactoring)

| Old file(s) | Destination | Reason / compatibility |
| --- | --- | --- |
| config/settings.py | retained | Preserve every environment option and default; new YAML config loader in src/utils/config.py |
| config/prompt_templates.py, config/__init__.py | retained | Existing generation prompts are functional; not part of competition retrieval |
| src/ingestion/segmentor.py | src/data/preprocess.py | Data preprocessing interface; preserve unfinished behavior and old import shim |
| src/ingestion/chunker.py | src/data/chunker.py | Context window interface; preserve unfinished behavior and old import shim |
| src/ingestion/indexer.py | src/data/indexer.py | Qdrant/BM25 indexing; old import shim |
| src/retrieval/sparse_search.py | src/retrieval/bm25.py | BM25 retriever; SparseRetriever alias retained |
| src/retrieval/dense_search.py | src/retrieval/dense.py | Dense retriever; old import shim |
| src/retrieval/hybrid_fusion.py | src/retrieval/fusion.py | Union, deduplication, RRF and CC; old class API retained |
| src/retrieval/reranker.py | src/reranking/bge.py | Separate model scoring from retrieval and selection; old import shim |
| src/retrieval/aggregator.py | src/scoring/doc_aggregation.py | Document score aggregation; old import shim |
| src/utils/metrics.py | src/evaluation/retrieval_metrics.py + evaluator.py | Ranking metrics and dataset evaluation; old exports retained |
| src/utils/validator.py | src/submission/validator.py | Submission schema and corpus validation; old import shim |
| scripts/run_pipeline.py | src/pipeline/full_pipeline.py + CLI entrypoints | Orchestration belongs in src; legacy config-path alias retained, changed CLI documented |
| scripts/pack_submission.py | src/submission/zipper.py + CLI entrypoint | ZIP and validation logic belongs in src; function aliases retained |
| src/generation/*.py | retained | Optional future answer generation; unfinished, outside retrieval competition scope |
| src/multilingual/*.py | retained | NER, translation and SHIFT interfaces; no working implementation to migrate into graph |
| src/utils/vram_monitor.py | retained | Existing GPU monitoring interface |
| src/**/__init__.py | retained / updated | Stable old and new imports; avoid eager optional dependency loading |
| tests/test_segmentor.py | retained | Existing pending segmentation tests |
| tests/test_retrieval.py | updated | Replace skipped retrieval tests with behavior checks |
| tests/test_validator.py | retained | Legacy pending tests; new submission tests exercise implementation |
| tests/__init__.py | retained | Test package |
| README.md | rewritten; original archived in docs/working_notes/README_original.md | Accurate runnable commands; preserve original design |
| DEV_A_README.md, DEV_B_README.md, DEV_C_README.md | retained | Historical ownership notes; superseded architecture documented |
| notes/22_09_26.txt | retained | Planning note mentioning Neo4j, not an implementation |
| requirements.txt, .gitignore | updated | Lightweight install metadata; exclude generated data/artifacts |
| data/raw, data/processed | retained | Never modify original raw data |
| data/bm25_index, data/qdrant_db | retained (legacy); new runs use artifacts/indexes | Preserve old Settings paths and any user indexes |
| notebooks/.gitkeep | retained | Preserve notebook area |

New modules will be limited to usable data adapters/schema/mappings, retriever
interfaces, candidate generation, selection, F-beta evaluation, submission,
configuration, pipeline entrypoints and tests. Graph and fine-tuning remain
explicit TODOs; no fake GraphRetriever or empty source placeholders.

Runtime paths in YAML are relative to the declaring YAML file, including inherited
configuration paths. Runs use outputs/<run_name>, created exclusively. Existing
runs and prepared datasets/index files must never be silently overwritten.

## Completion

The mapping above has been applied. See [refactor_report.md](refactor_report.md)
for the new-file inventory, compatibility details, verification results and TODOs.
The original 15 skipped tests became active retrieval/validator tests; only the
five pre-existing segmentation tests remain pending. No tracked files were deleted.
