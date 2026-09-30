# %% [markdown]
# # P2-01: Qwen3 Reranker Comprehensive Benchmark & Evaluation (P1-11 Handoff)
# ### Đánh giá mô hình Reranker trên gói bàn giao Candidates P1-11
#
# Notebook này thực hiện quy trình đánh giá và benchmark chuyên sâu cho **Qwen/Qwen3-Reranker-0.6B** dựa trên gói candidates chuẩn hóa được bàn giao từ P1-11:
# 1. **Kiểm tra môi trường & GPU**: Xác thực CUDA và VRAM khả dụng trên Kaggle T4.
# 2. **Xác thực gói Handoff**: Kiểm tra tính toàn vẹn `manifest.json`, `candidates.jsonl`, `labels.json`, `registry.json`.
# 3. **Thực thi Reranking Benchmark**: Chạy inference chấm điểm từng cặp `(query, chunk)` qua `benchmark_reranking`.
# 4. **Phân tích Đa chiều**: So sánh macro metrics Before vs After (Recall@K, MRR@K, NDCG@K), phân tích độ trễ (latency mean, p50, p95), phân tích cải thiện theo từng query và trực quan hóa biểu đồ.

# %% Cell 1: Clone Repository from GitHub into Working Directory
import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Djuybu/cocopila_2.0.git"
WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
REPO_DIR = WORKING_DIR / "cocopila_2.0"

# Clone or pull repository on Kaggle / working directory
if Path("/kaggle").exists() or not (WORKING_DIR / "src").exists():
    if not REPO_DIR.exists():
        print(f"Cloning {REPO_URL} into {REPO_DIR}...")
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    else:
        print(f"Updating repository at {REPO_DIR}...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=False)

    # Copy src, config (Python package) and configs (YAMLs) into working directory
    for item in ["src", "config", "configs"]:
        src_path = REPO_DIR / item
        dest_path = WORKING_DIR / item
        if src_path.exists() and not dest_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)

    # Install editable package so both src and config modules are registered
    subprocess.run(["pip", "install", "-q", "-e", str(REPO_DIR)], check=False)
    os.chdir(WORKING_DIR)

print(f"Working directory ready: {os.getcwd()}")

# %% Cell 2: Environment Setup & GPU Verification
import logging
import os
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p2_reranker_eval")

# Check CUDA GPU Availability
num_gpus = torch.cuda.device_count()
logger.info(f"CUDA Available: {torch.cuda.is_available()}, Device Count: {num_gpus}")

for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    total_mem_gb = props.total_memory / (1024**3)
    logger.info(f"GPU {i}: {props.name} ({total_mem_gb:.2f} GB VRAM)")

# %% Cell 3: Resolve Handoff Input Directory
import zipfile

possible_locations = [
    Path(os.environ.get("P1_HANDOFF_DIR", "")),
    Path("/kaggle/input/datasets/duymcminh/r2ai-phase3-reranker-model-test-dataset"),
    Path("/kaggle/input/r2ai-phase3-reranker-model-test-dataset"),
    Path("outputs/p1_p2_handoff_qwen3"),
    Path("outputs/p1_p2_handoff"),
    Path("../outputs/p1_p2_handoff_qwen3"),
]

HANDOFF_DIR = None

# 1. Check explicitly listed paths (including nested folders and zip archives)
for loc in possible_locations:
    if loc and loc.exists():
        if loc.is_dir() and (loc / "manifest.json").exists():
            HANDOFF_DIR = loc
            break
        if loc.is_dir():
            sub_manifests = list(loc.rglob("manifest.json"))
            if sub_manifests:
                HANDOFF_DIR = sub_manifests[0].parent
                break
        zips = list(loc.glob("*.zip")) if loc.is_dir() else ([loc] if loc.suffix == ".zip" else [])
        if zips:
            extract_target = Path("/kaggle/working/p1_p2_handoff") if Path("/kaggle/working").exists() else Path("outputs/p1_p2_handoff")
            extract_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(extract_target)
            extracted_manifests = list(extract_target.rglob("manifest.json"))
            if extracted_manifests:
                HANDOFF_DIR = extracted_manifests[0].parent
                break

# 2. Dynamic discovery: search anywhere in /kaggle/input if not found yet
if HANDOFF_DIR is None and Path("/kaggle/input").exists():
    kaggle_manifests = list(Path("/kaggle/input").rglob("manifest.json"))
    if kaggle_manifests:
        HANDOFF_DIR = kaggle_manifests[0].parent
    else:
        kaggle_zips = list(Path("/kaggle/input").rglob("*.zip"))
        if kaggle_zips:
            extract_target = Path("/kaggle/working/p1_p2_handoff")
            extract_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(kaggle_zips[0], "r") as zf:
                zf.extractall(extract_target)
            extracted = list(extract_target.rglob("manifest.json"))
            if extracted:
                HANDOFF_DIR = extracted[0].parent

if HANDOFF_DIR is None:
    HANDOFF_DIR = Path("outputs/p1_p2_handoff_qwen3")

logger.info(f"Using handoff input directory: {HANDOFF_DIR.resolve()}")
if not HANDOFF_DIR.exists():
    kaggle_inputs = [str(p) for p in Path("/kaggle/input").iterdir()] if Path("/kaggle/input").exists() else []
    logger.warning(
        f"Handoff bundle not found at {HANDOFF_DIR.resolve()}. "
        f"Available items in /kaggle/input: {kaggle_inputs}. "
        "Please attach your dataset in Kaggle via '+ Add Input'."
    )

# %% Cell 4: Validate Handoff Input Manifest
from src.pipeline.handoff import validate_reranking_input
from src.utils.io import read_json

if HANDOFF_DIR.exists():
    manifest = validate_reranking_input(HANDOFF_DIR)
    manifest_summary = {
        "schema_version": manifest.get("schema_version"),
        "stage": manifest.get("stage"),
        "query_count": manifest.get("query_count"),
        "candidate_count": manifest.get("candidate_count"),
        "label_quality": manifest.get("label_quality"),
        "id_namespace": manifest.get("id_namespace"),
    }
    logger.info(f"Handoff Manifest Validated: {manifest_summary}")
else:
    manifest = None
    logger.warning(f"Handoff directory {HANDOFF_DIR} does not exist yet. Please provide handoff bundle.")

# %% Cell 5: Inspect Queries & Candidate Pool Statistics
import pandas as pd
from src.data.loader import load_records

if HANDOFF_DIR.exists() and (HANDOFF_DIR / "candidates.jsonl").exists():
    queries_data = read_json(HANDOFF_DIR / "queries.json")
    candidates_records = load_records(HANDOFF_DIR / "candidates.jsonl")
    cand_counts = [len(r.get("candidates", [])) for r in candidates_records]
    df_inspect = pd.DataFrame({
        "query_id": [r["id"] for r in candidates_records],
        "query_text": [q["text"] for q in queries_data[:len(candidates_records)]],
        "candidate_count": cand_counts,
    })
    logger.info(f"Loaded {len(candidates_records)} queries with total {sum(cand_counts)} candidates.")
    logger.info(f"Candidate count stats: Min={min(cand_counts)}, Max={max(cand_counts)}, Mean={sum(cand_counts)/len(cand_counts):.1f}")
    print(df_inspect.head(10).to_string(index=False))

# %% Cell 6: Load Reranker Configuration
from src.utils.config import load_config

CONFIG_PATH = Path(os.environ.get("P2_RERANKER_CONFIG", "configs/reranker/qwen_reranker.yaml"))
OUTPUT_DIR = Path(os.environ.get("P2_OUTPUT_DIR", "outputs/p2_qwen3_001"))

if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("../configs/reranker/qwen_reranker.yaml")

config = load_config(CONFIG_PATH)
reranker_cfg = config.get("reranker", config)

logger.info(f"Loaded configuration from: {CONFIG_PATH.resolve()}")
logger.info(f"Reranker Type: {reranker_cfg.get('type')}, Model: {reranker_cfg.get('model')}")
logger.info(f"Batch Size: {reranker_cfg.get('batch_size')}, Dtype: {reranker_cfg.get('dtype')}")
logger.info(f"Output Directory: {OUTPUT_DIR.resolve()}")

# %% Cell 7: Execute Reranking Benchmark Engine
from src.pipeline.reranker_benchmark import benchmark_reranking

if HANDOFF_DIR.exists():
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    logger.info(f"Running reranking benchmark from {HANDOFF_DIR} -> {OUTPUT_DIR}...")
    report = benchmark_reranking(
        HANDOFF_DIR,
        OUTPUT_DIR,
        reranker_config=config,
        ks=[1, 3, 5, 10, 20, 50, 100, 200],
    )
    logger.info(f"Benchmark finished! Evaluated {report['query_count']} queries, {report['candidate_count']} candidates.")
else:
    report = None
    logger.warning(f"Cannot run benchmark because {HANDOFF_DIR} was not found.")

# %% Cell 8: Macro Metrics Comparison (Before vs After Reranking)
if report is not None:
    ks = [1, 3, 5, 10, 20, 50, 100, 200]
    comparison_rows = []
    before_chunks = report["before"]["macro"]["chunks"]
    after_chunks = report["after"]["macro"]["chunks"]
    for metric_prefix in ["recall", "ndcg", "mrr", "precision"]:
        for k in ks:
            metric_key = f"{metric_prefix}@{k}"
            if metric_key in before_chunks and metric_key in after_chunks:
                b_val = before_chunks[metric_key]
                a_val = after_chunks[metric_key]
                delta = a_val - b_val
                comparison_rows.append({
                    "Metric": metric_key.upper(),
                    "Before (Retrieval)": f"{b_val:.4f}",
                    "After (Reranking)": f"{a_val:.4f}",
                    "Delta": f"{delta:+.4f}",
                    "Status": "Improved" if delta > 0 else ("Unchanged" if delta == 0 else "Degraded"),
                })
    df_metrics = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 70)
    print("      MACRO EVALUATION: RETRIEVAL BASELINE vs QWEN RERANKER")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("=" * 70)

# %% Cell 9: Latency, Throughput & Resource Profiling
if report is not None:
    perf = report["performance"]
    perf_rows = [
        {"Resource Metric": "End-to-End Elapsed Time", "Value": f"{perf.get('end_to_end_seconds', 0):.2f} s"},
        {"Resource Metric": "Model Load Time", "Value": f"{perf.get('model_load_seconds', 0):.2f} s"},
        {"Resource Metric": "Mean Query Latency", "Value": f"{perf.get('latency_seconds_mean', 0)*1000:.1f} ms"},
        {"Resource Metric": "Median Query Latency (P50)", "Value": f"{perf.get('latency_seconds_p50', 0)*1000:.1f} ms"},
        {"Resource Metric": "P95 Query Latency", "Value": f"{perf.get('latency_seconds_p95', 0)*1000:.1f} ms"},
        {"Resource Metric": "Throughput (Queries/s)", "Value": f"{perf.get('queries_per_second', 0):.2f} qps"},
        {"Resource Metric": "Candidate Scoring Throughput", "Value": f"{perf.get('candidates_per_second', 0):.1f} chunks/s"},
    ]
    if perf.get("peak_cuda_allocated_bytes"):
        vram_gb = perf["peak_cuda_allocated_bytes"] / (1024**3)
        perf_rows.append({"Resource Metric": "Peak CUDA VRAM", "Value": f"{vram_gb:.2f} GB"})
    df_perf = pd.DataFrame(perf_rows)
    print("\n" + "-" * 60)
    print("               RESOURCE & INFERENCE PROFILING")
    print("-" * 60)
    print(df_perf.to_string(index=False))
    print("-" * 60)

# %% Cell 10: Per-Query Impact Analysis (Top Gains & Losses)
if report is not None:
    scored_candidates = load_records(OUTPUT_DIR / "reranked.jsonl")
    labels = read_json(HANDOFF_DIR / "labels.json")
    label_map = {row["id"]: set(row.get("relevant_chunks", [])) for row in labels}
    query_gains = []
    for sc in scored_candidates:
        qid = sc["id"]
        rel_cids = label_map.get(qid, set())
        if not rel_cids:
            continue
        cands = sc.get("candidates", [])
        initial_ranks = [i + 1 for i, c in enumerate(cands) if c.get("chunk_id") in rel_cids]
        sorted_after = sorted(cands, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        after_ranks = [i + 1 for i, c in enumerate(sorted_after) if c.get("chunk_id") in rel_cids]
        best_init = min(initial_ranks) if initial_ranks else 999
        best_after = min(after_ranks) if after_ranks else 999
        rank_change = best_init - best_after
        query_gains.append({
            "Query ID": qid,
            "Initial Top Rank": best_init,
            "Reranked Top Rank": best_after,
            "Rank Delta": rank_change,
            "Outcome": "Improved" if rank_change > 0 else ("Degraded" if rank_change < 0 else "Neutral"),
        })
    df_queries = pd.DataFrame(query_gains)
    print("\nPer-Query Impact Summary:")
    print(df_queries.head(10).to_string(index=False))

# %% Cell 11: Visualizations (Recall@K Curves)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is not None and report is not None:
    ks = [1, 3, 5, 10, 20, 50, 100, 200]
    b_recalls = [report["before"]["macro"]["chunks"].get(f"recall@{k}", 0.0) for k in ks]
    a_recalls = [report["after"]["macro"]["chunks"].get(f"recall@{k}", 0.0) for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, b_recalls, marker="o", linestyle="--", color="#1f77b4", label="Before Reranking (RRF)")
    ax.plot(ks, a_recalls, marker="s", linestyle="-", color="#ff7f0e", label="After Reranking (Qwen3)")
    ax.set_title("Recall@K Curve: Before vs After Reranking", fontsize=13)
    ax.set_xlabel("Cutoff K", fontsize=11)
    ax.set_ylabel("Macro Recall", fontsize=11)
    ax.set_xticks(ks)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart_file = OUTPUT_DIR / "reranker_recall_curve.png"
    plt.savefig(chart_file, dpi=200)
    logger.info(f"Saved recall curve chart to {chart_file.resolve()}")
    plt.show()
else:
    logger.info("Matplotlib not available or report empty; skipping visualization.")

# %% Cell 12: Export Summary Report CSV & Artifacts
if report is not None:
    summary_path = OUTPUT_DIR / "reranking_benchmark_summary.csv"
    if "df_metrics" in locals():
        df_metrics.to_csv(summary_path, index=False)
        logger.info(f"Saved metrics summary CSV to: {summary_path.resolve()}")
    print("\nEvaluation successfully completed! All artifacts generated in:", OUTPUT_DIR.resolve())
