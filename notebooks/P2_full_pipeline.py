# %% [markdown]
# # P2 Full Pipeline: Reranking (BGE-M3), F2 Optimization & P3 Handoff Bundle
# ### Khung điều phối toàn bộ công việc của Thành viên 2 (P2: Mạc Duy) theo Checklist
#
# Notebook này đóng vai trò là **khung điều phối hoàn chỉnh** cho Thành viên 2:
# - **Stage 1**: Xác thực & nạp gói bàn giao từ Thành viên 1 (`candidates.jsonl`, `manifest.json`).
# - **Stage 2**: Thực thi Reranking Engine tập trung **BGE-Reranker-v2-m3** với mixed precision và batch inference (**P2-01, P2-11, P2-15**).
# - **Stage 3**: Đánh giá đa chiều Trước vs Sau Rerank (Recall@K, MRR@K, NDCG@K, throughput, latency, VRAM) (**P2-11**).
# - **Stage 4**: Quét tối ưu đồng thời Selector `(threshold, fallback, max)` trực tiếp theo Macro Chunk F2 và xác định vùng plateau (**P2-02, P2-03, P2-04, P2-05, P2-12**).
# - **Stage 5**: Đánh giá chi tiết Chunk-Level F2 per-query qua module P2-02 (`evaluate_f2.py`) và xuất báo cáo kiểm định (**P2-02**).
# - **Stage 6**: Trực quan hóa độ nhạy tham số F2 và biểu đồ plateau.
# - **Stage 7**: Áp dụng selector tối ưu, ánh xạ mã định danh sang official ID và kiểm thử hợp lệ qua `SubmissionValidator`.
# - **Stage 8**: Phân tích lỗi False Negative (tách lỗi P1 Retrieval Miss vs P2 Reranker/Threshold Miss) (**P2-14**).
# - **Stage 9**: Đóng gói toàn bộ tài nguyên đầu ra vào `p2_to_p3_handoff/` kèm `manifest.json` và `p2_f2_evaluation_report.csv` để **Thành viên 3 (P3) sử dụng ngay cho Doc Aggregation và hoàn thiện Submission** (**P3-02, P3-04, P3-05, P3-06, P3-15**).

# %% Cell 1: Environment & Workspace Setup
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p2_full_pipeline")

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
REPO_URL = "https://github.com/Djuybu/cocopila_2.0.git"
REPO_DIR = WORKING_DIR / "cocopila_2.0"

# Setup repository workspace on Kaggle or local
if Path("/kaggle").exists() or not (WORKING_DIR / "src").exists():
    if not REPO_DIR.exists():
        print(f"Cloning repository {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    else:
        print("Pulling latest updates...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=False)

    for item in ["src", "config", "configs"]:
        src_path = REPO_DIR / item
        dest_path = WORKING_DIR / item
        if src_path.exists() and not dest_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)

    subprocess.run(["pip", "install", "-q", "-e", str(REPO_DIR)], check=False)
    os.chdir(WORKING_DIR)

if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))

# Hardware check: GPU CUDA & VRAM
cuda_avail = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0

print("=" * 70)
print(f"  WORKSPACE READY: {os.getcwd()}")
print(f"  DEVICE: {device_name} (CUDA: {cuda_avail}, Devices: {num_gpus}, VRAM: {vram_gb:.2f} GB)")
print("=" * 70)

# %% Cell 2: Resolve & Validate P1 Handoff Input Bundle
import zipfile
from src.pipeline.handoff import validate_reranking_input
from src.utils.io import read_json

possible_handoff_locations = [
    Path(os.environ.get("P1_HANDOFF_DIR", "")),
    Path("/kaggle/input/datasets/duymcminh/r2ai-phase3-reranker-model-test-dataset"),
    Path("/kaggle/input/r2ai-phase3-reranker-model-test-dataset"),
    Path("data/p1_p2_handoff_qwen3"),
    Path("data/p1_p2_handoff"),
    Path("outputs/bge_reranked_run"),
    Path("outputs/p1_p2_handoff_qwen3"),
    Path("outputs/p1_p2_handoff"),
    Path("../data/p1_p2_handoff_qwen3"),
    Path("../outputs/p1_p2_handoff_qwen3"),
]

HANDOFF_DIR = None

# Search explicitly known paths
for loc in possible_handoff_locations:
    if loc and loc.exists():
        if loc.is_dir() and (loc / "manifest.json").exists():
            HANDOFF_DIR = loc
            break
        if loc.is_dir():
            manifests = list(loc.rglob("manifest.json"))
            if manifests:
                HANDOFF_DIR = manifests[0].parent
                break
        zips = list(loc.glob("*.zip")) if loc.is_dir() else ([loc] if loc.suffix == ".zip" else [])
        if zips:
            extract_target = WORKING_DIR / "p1_p2_handoff"
            extract_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(extract_target)
            manifests = list(extract_target.rglob("manifest.json"))
            if manifests:
                HANDOFF_DIR = manifests[0].parent
                break

# Dynamic search anywhere in /kaggle/input if needed
if HANDOFF_DIR is None and Path("/kaggle/input").exists():
    found_manifests = list(Path("/kaggle/input").rglob("manifest.json"))
    if found_manifests:
        HANDOFF_DIR = found_manifests[0].parent

if HANDOFF_DIR is None:
    HANDOFF_DIR = Path("outputs/bge_reranked_run")

print(f"Handoff Input Directory: {HANDOFF_DIR.resolve()}")
if HANDOFF_DIR.exists() and (HANDOFF_DIR / "manifest.json").exists():
    manifest = validate_reranking_input(HANDOFF_DIR)
    print("\n--- P1 Handoff Manifest Validated ---")
    print(f"  Schema Version  : {manifest.get('schema_version')}")
    print(f"  Source Run ID   : {manifest.get('source_run_id')}")
    print(f"  Query Count     : {manifest.get('query_count')}")
    print(f"  Candidate Count : {manifest.get('candidate_count')}")
    print(f"  Label Quality   : {manifest.get('label_quality')}")
    print(f"  Split           : {manifest.get('dataset_split')}")
else:
    manifest = None
    print(f"[WARNING] Handoff bundle not found at {HANDOFF_DIR.resolve()}. Please attach dataset.")

# %% Cell 3: Configuration (BGE Reranker & Output Paths)
from src.utils.config import load_config

P2_OUTPUT_DIR = Path(os.environ.get("P2_OUTPUT_DIR", "outputs/p2_bge_reranker_run"))
P3_HANDOFF_DIR = Path(os.environ.get("P3_HANDOFF_DIR", "outputs/p2_to_p3_handoff"))

# Select Reranker Configuration: Focus on BGE-Reranker-v2-m3
CONFIG_PATH = Path(os.environ.get("P2_RERANKER_CONFIG", "configs/reranker/bge_reranker.yaml"))
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("../configs/reranker/bge_reranker.yaml")

config = load_config(CONFIG_PATH) if CONFIG_PATH.exists() else {
    "reranker": {
        "enabled": True,
        "type": "bge",
        "model": "BAAI/bge-reranker-v2-m3",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 16 if torch.cuda.is_available() else 4,
        "max_length": 512,
    }
}
reranker_cfg = config.get("reranker", config)

# Auto-adjust device and batch size to match runtime environment
if not torch.cuda.is_available() and reranker_cfg.get("device") == "cuda":
    reranker_cfg["device"] = "cpu"
    reranker_cfg["batch_size"] = min(reranker_cfg.get("batch_size", 4), 4)

print("=" * 70)
print(f"  P2 RERANKER MODEL : {reranker_cfg.get('model')}")
print(f"  DEVICE / BATCH     : {reranker_cfg.get('device')} | Batch Size: {reranker_cfg.get('batch_size')}")
print(f"  P2 OUTPUT DIR      : {P2_OUTPUT_DIR.resolve()}")
print(f"  P3 HANDOFF DIR     : {P3_HANDOFF_DIR.resolve()}")
print("=" * 70)

# %% Cell 4: Execute Reranking Benchmark Engine (P2-01, P2-11, P2-15)
from src.pipeline.reranker_benchmark import benchmark_reranking

if HANDOFF_DIR.exists() and (HANDOFF_DIR / "candidates.jsonl").exists():
    if P2_OUTPUT_DIR.exists():
        raise FileExistsError(P2_OUTPUT_DIR)

    print(f"Scoring candidates from {HANDOFF_DIR} -> {P2_OUTPUT_DIR}...")
    report = benchmark_reranking(
        HANDOFF_DIR,
        P2_OUTPUT_DIR,
        reranker_config=config,
        ks=[1, 3, 5, 10, 20, 50, 100, 200],
    )
    print(f"\nReranking completed! Scored {report['query_count']} queries, {report['candidate_count']} candidates.")
else:
    report = None
    print("[WARNING] Skipping reranking execution: handoff bundle missing.")

# %% Cell 5: Ranking Benchmark & Latency Profiling (P2-11)
import pandas as pd

if report is not None:
    ks = [1, 3, 5, 10, 20, 50, 100, 200]
    comparison_rows = []
    before_chunks = report["before"]["macro"]["chunks"]
    after_chunks = report["after"]["macro"]["chunks"]

    for prefix in ["recall", "ndcg", "mrr", "precision"]:
        for k in ks:
            metric_key = f"{prefix}@{k}"
            if metric_key in before_chunks and metric_key in after_chunks:
                b_val = before_chunks[metric_key]
                a_val = after_chunks[metric_key]
                delta = a_val - b_val
                comparison_rows.append({
                    "Metric": metric_key.upper(),
                    "Before (RRF)": f"{b_val:.4f}",
                    "After (Reranker)": f"{a_val:.4f}",
                    "Delta": f"{delta:+.4f}",
                    "Status": "Improved" if delta > 0 else ("Unchanged" if delta == 0 else "Degraded"),
                })

    df_metrics = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 70)
    print("      MACRO EVALUATION: RETRIEVAL BASELINE vs RERANKER")
    print("=" * 70)
    print(df_metrics.to_string(index=False))

    perf = report["performance"]
    print("\n" + "-" * 60)
    print("               RESOURCE & LATENCY PROFILING")
    print("-" * 60)
    print(f"  Mean Latency / Query : {perf.get('latency_seconds_mean', 0)*1000:.1f} ms")
    print(f"  Median Latency (P50) : {perf.get('latency_seconds_p50', 0)*1000:.1f} ms")
    print(f"  P95 Latency          : {perf.get('latency_seconds_p95', 0)*1000:.1f} ms")
    print(f"  Throughput           : {perf.get('queries_per_second', 0):.2f} queries/s ({perf.get('candidates_per_second', 0):.1f} chunks/s)")
    if perf.get("peak_cuda_allocated_bytes"):
        print(f"  Peak CUDA VRAM       : {perf['peak_cuda_allocated_bytes'] / (1024**3):.2f} GB")
    print("-" * 60)

# %% Cell 6: Joint Chunk Selector Sweep & F2 Optimization (P2-02, P2-03, P2-04, P2-05, P2-12)
from src.data.loader import load_records
from src.scoring.sweep import sweep_chunk_selector
from src.scoring.cv_threshold import cross_validate_selector
from src.utils.io import write_json

if P2_OUTPUT_DIR.exists() and (P2_OUTPUT_DIR / "reranked.jsonl").exists():
    scored_records = load_records(P2_OUTPUT_DIR / "reranked.jsonl")
    labels = read_json(P2_OUTPUT_DIR / "labels.json")
    registry = read_json(P2_OUTPUT_DIR / "registry.json")
    mapping = registry.get("internal_to_official", {})
    chunk_to_doc = registry.get("chunk_to_doc")

    print(f"Running joint selector sweep for {len(scored_records)} queries...")
    best_config, sweep_df, plateau_info = sweep_chunk_selector(
        scored_records=scored_records,
        labels=labels,
        thresholds=None,  # Dynamic grid across score distribution
        fallbacks=[0, 1, 2, 3, 5],
        maximums=[3, 5, 10, 15, 20],
        internal_to_official=mapping,
        chunk_to_doc=chunk_to_doc,
    )
    best_config["evaluation_scope"] = "tuning_on_supplied_labels"
    selector_cv = cross_validate_selector(
        scored_records, labels, fallbacks=[0, 1, 2, 3, 5], maximums=[3, 5, 10, 15, 20],
        internal_to_official=mapping, chunk_to_doc=chunk_to_doc,
    )
    write_json(P2_OUTPUT_DIR / "selector_cv_report.json", selector_cv)
    if selector_cv["status"] == "complete":
        print(f"Out-of-fold Macro F2: {selector_cv['macro']['f2']:.4f}")

    print("\n" + "=" * 70)
    print("      OPTIMAL CHUNK SELECTOR (MAXIMIZING MACRO F2 CHUNK)")
    print("=" * 70)
    print(f"  Optimal Threshold     : {best_config['chunk_threshold']}")
    print(f"  Optimal Fallback      : {best_config['chunk_fallback']}")
    print(f"  Optimal Max Chunks    : {best_config['chunk_max']}")
    print(f"  Macro F2 Score        : {best_config['macro_f2']:.4f}")
    print(f"  Macro Recall          : {best_config['macro_recall']:.4f}")
    print(f"  Macro Precision       : {best_config['macro_precision']:.4f}")
    print(f"  Avg Chunks / Query    : {best_config['avg_chunks_per_query']:.1f}")
    print(f"  Plateau Region        : [{plateau_info['plateau_min_threshold']}, {plateau_info['plateau_max_threshold']}] ({plateau_info['plateau_count']} configs within 1% of best F2)")
    print("=" * 70)

    print("\nTop 5 Parameter Configurations:")
    print(sweep_df.head(5).to_string(index=False))

    # P2-02: Detailed per-query evaluation of the optimal selector
    from src.evaluation.evaluate_f2 import (
        evaluate_chunk_predictions,
        export_metrics_csv,
        export_metrics_json,
        format_metrics_table,
    )
    from src.scoring.sweep import generate_p2_chunk_predictions

    optimal_chunk_preds = generate_p2_chunk_predictions(
        scored_records,
        best_config,
        internal_to_official=mapping,
        chunk_to_doc=chunk_to_doc,
    )

    f2_report = evaluate_chunk_predictions(optimal_chunk_preds, labels)

    print("\n" + "=" * 70)
    print("      DETAILED PER-QUERY F2 CHUNK EVALUATION (P2-02)")
    print("=" * 70)
    print(format_metrics_table(f2_report))

    f2_csv_path = P2_OUTPUT_DIR / "p2_f2_evaluation_report.csv"
    f2_json_path = P2_OUTPUT_DIR / "p2_f2_evaluation_report.json"
    export_metrics_csv(f2_report, f2_csv_path)
    export_metrics_json(f2_report, f2_json_path)
    print(f"\n[P2-02] Saved detailed evaluation reports to:")
    print(f"  - CSV : {f2_csv_path.resolve()}")
    print(f"  - JSON: {f2_json_path.resolve()}")
else:
    best_config, sweep_df, plateau_info = None, None, None
    print("[WARNING] Cannot perform sweep: reranked.jsonl not found.")

# %% Cell 7: Visualization: F2 Sensitivity & Plateau Curves
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is not None and sweep_df is not None and not sweep_df.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Curve 1: F2 vs Threshold across Fallback values
    for fb in [0, 1, 2, 3]:
        sub = sweep_df[(sweep_df["chunk_fallback"] == fb) & (sweep_df["chunk_max"] == best_config["chunk_max"])]
        sub = sub.dropna(subset=["chunk_threshold"]).sort_values("chunk_threshold")
        if not sub.empty:
            ax1.plot(sub["chunk_threshold"], sub["macro_f2"], marker="o", label=f"Fallback={fb}")

    p_min = plateau_info.get("plateau_min_threshold") if plateau_info else None
    p_max = plateau_info.get("plateau_max_threshold") if plateau_info else None
    p_rec = plateau_info.get("recommended_threshold") if plateau_info else None
    if p_min is not None and p_max is not None:
        ax1.axvspan(p_min, p_max, color="yellow", alpha=0.25, label=f"Plateau [{p_min:.3f}, {p_max:.3f}]")
    if p_rec is not None:
        ax1.axvline(p_rec, color="blue", linestyle="-.", linewidth=2, label=f"Recommended ({p_rec:.3f})")

    ax1.set_title(f"Macro F2 vs Threshold (Max={best_config['chunk_max']})", fontsize=12)
    ax1.set_xlabel("Chunk Threshold", fontsize=10)
    ax1.set_ylabel("Macro F2", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Curve 2: Precision vs Recall Trade-off across configurations
    scatter = ax2.scatter(
        sweep_df["macro_recall"],
        sweep_df["macro_precision"],
        c=sweep_df["macro_f2"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Macro F2", fontsize=10)
    ax2.set_title("Precision vs Recall Distribution (Color = F2)", fontsize=12)
    ax2.set_xlabel("Macro Recall", fontsize=10)
    ax2.set_ylabel("Macro Precision", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_path = P2_OUTPUT_DIR / "p2_f2_optimization_curves.png"
    plt.savefig(chart_path, dpi=200)
    print(f"Saved optimization visualization to: {chart_path.resolve()}")
    plt.show()

# %% Cell 8: False Negative Error Analysis (P2-14)
from src.scoring.sweep import analyze_false_negatives

if scored_records and labels and best_config:
    fn_df, fn_summary = analyze_false_negatives(
        scored_records=scored_records,
        labels=labels,
        best_config=best_config,
        internal_to_official=mapping,
        chunk_to_doc=chunk_to_doc,
    )

    print("\n" + "=" * 70)
    print("          FALSE NEGATIVE (FN) ERROR ATTRIBUTION (P2-14)")
    print("=" * 70)
    print(f"  Total False Negatives     : {fn_summary['total_false_negatives']}")
    print(f"  P1 Retrieval Miss         : {fn_summary['p1_retrieval_miss']} ({fn_summary['p1_miss_pct']}%) -> Missing from P1 candidates")
    print(f"  P2 Threshold Pruned       : {fn_summary['p2_threshold_pruned']} -> Filtered out by chunk_threshold")
    print(f"  P2 Max Cap Pruned         : {fn_summary['p2_max_cap_pruned']} -> Filtered out by chunk_max")
    print(f"  Total P2 Pruned Misses    : {fn_summary['p2_miss_pct']}%")
    print("=" * 70)

    if not fn_df.empty:
        print("\nSample False Negative Entries:")
        print(fn_df.head(10)[["query_id", "chunk_id", "category", "rerank_score", "reason"]].to_string(index=False))
else:
    fn_df, fn_summary = None, None

# %% Cell 9: Package Final Deliverables for Member 3 (P3 Handoff)
from src.scoring.sweep import package_p2_to_p3_handoff

if P2_OUTPUT_DIR.exists() and best_config and sweep_df is not None and fn_df is not None:
    print(f"\nPackaging complete P2 deliverables for Member 3 into: {P3_HANDOFF_DIR.resolve()}...")
    package_p2_to_p3_handoff(
        output_dir=P3_HANDOFF_DIR,
        source_run_dir=P2_OUTPUT_DIR,
        best_config=best_config,
        sweep_df=sweep_df,
        fn_df=fn_df,
        benchmark_report=report,
        archive_zip=True,
    )

    p3_manifest = read_json(P3_HANDOFF_DIR / "manifest.json")
    print("\n" + "=" * 70)
    print("  [SUCCESS] P2 -> P3 HANDOFF PACKAGE GENERATED AND VERIFIED")
    print("=" * 70)
    print(f"  Handoff Directory  : {P3_HANDOFF_DIR.resolve()}")
    print(f"  Archive ZIP        : {P3_HANDOFF_DIR.parent / f'{P3_HANDOFF_DIR.name}.zip'}")
    print("\nDeliverables included for Member 3:")
    print("  [x] reranked.jsonl           -> Cho P3-02, P3-03 (Doc Aggregation: max-p, top-k mean)")
    print("  [x] best_chunk_selector.yaml -> Cho P3-04, P3-15 (Cấu hình chunk selector tối ưu)")
    print("  [x] p2_selected_chunks.json  -> Cho P3-05, P3-06 (Consistency check & merge vào Submission)")
    print("  [x] p2_f2_evaluation_report.csv -> Báo cáo đánh giá F2 per-query chi tiết (P2-02)")
    print("  [x] threshold_sweep.csv      -> Báo cáo sweep F2 chunk và plateau")
    print("  [x] fn_analysis.csv          -> Báo cáo phân tích FN phối hợp P1 & P2")
    print("  [x] manifest.json            -> Checksum toàn vẹn và metadata truy vết P1")
    print("=" * 70)
    print("\nMember 3 (P3) is now unblocked and ready to complete Doc Aggregation & Official Submission!")
else:
    print("[WARNING] Cannot package handoff: prerequisites not completed.")
