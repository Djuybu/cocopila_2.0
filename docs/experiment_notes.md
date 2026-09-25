# Experiment conventions

- Use a unique run_name such as exp014. Copy an experiment YAML and update the name
  before another run; outputs/<run_name> cannot already exist.
- Paths belong to the YAML file that declares them, not the shell working directory.
- Dataset text and official ID registries must be versioned together. Preparation
  creates chunk_to_doc.json and internal_to_official_id.json.
- Build into new artifact paths when changing corpus, tokenization or dense model
  settings. Stale BM25/dense indexes are rejected.
- Record manual observations in experiments/experiment_log.csv and ablations/.
  Pipeline config and metrics are automatically recorded within each run; the
  CSV files are intentionally human-maintained indexes.
- Submission CSV is likewise manual; do not fabricate leaderboard scores.
- Keep train/val/test document-disjoint where required. Configure
  data.splits.<name>.data_path during preparation to validate document overlap.
  The adapter does not invent a split or infer query-document supervision.
- Fit thresholds and aggregation weights on val only. Baseline null thresholds
  return top ranked items up to the configured maxima.
- Dense/reranker smoke tests isolate adapters from model downloads. Actual model
  quality, CUDA execution and large-corpus memory use require separate validation.
- JSON/JSONL loaders and submission validation currently materialize records.
  Streaming for competition-scale corpora is a future task, not a claimed feature.

Pending: graph retrieval, fine-tuning/hard negatives, production Vietnamese
preprocessing, translation/NER/SHIFT, generation, VRAM monitoring, official scoring
rule confirmation, and model quality benchmarks.
