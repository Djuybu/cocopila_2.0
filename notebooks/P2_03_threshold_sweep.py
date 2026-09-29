# %% [markdown]
# # P2-03: Reranker Threshold Sweep & Plateau Analysis
# ### Quét Ngưỡng Điểm Reranker & Nhận Diện Vùng Bình Nguyên (Plateau) Theo Chuẩn Checklist P2-03
#
# Notebook này thực thi nhiệm vụ **P2-03** (Phụ trách: **Mạc Duy** - Ưu tiên: **P0**):
# 1. **Khởi tạo môi trường & Workspace**: Kiểm tra môi trường Kaggle / Local.
# 2. **Nạp dữ liệu Scored Candidates**: Nạp danh sách ứng viên đã được chấm điểm từ Reranker (`reranked.jsonl` hoặc `candidates.jsonl`) cùng `labels.json`.
# 3. **(Tùy chọn) Thực thi Reranking**: Chạy nhanh BGE-Reranker-v2-m3 nếu chưa có điểm số (P2-01).
# 4. **Quét dải ngưỡng điểm (Threshold Sweep)**: Đo đạc Macro Precision, Recall, F1, F2 và số chunk trung bình trên từng query (`avg_chunks_per_query`).
# 5. **Nhận diện Vùng Bình Nguyên (Threshold Plateau)**: Cô lập dải ngưỡng $[T_{min}, T_{max}]$ đạt đỉnh $F_2$ ổn định và chọn ngưỡng khuyến nghị (median) tránh overfit.
# 6. **Trực quan hóa đồ thị**: Vẽ đường cong $F_2$-Precision-Recall và phân tích xu hướng Zero-Chunk queries (tiền đề cho P2-04 Fallback).
# 7. **Xuất sản phẩm bàn giao (Deliverable)**: Tạo file `threshold_sweep.csv` nghiệm thu tiêu chí P2-03.

# %% Cell 1: Environment & Workspace Setup
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p2_03_threshold_sweep")

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
device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
print("=" * 70)
print(f"  WORKSPACE READY : {os.getcwd()}")
print(f"  COMPUTE DEVICE  : {device_name} (CUDA: {cuda_avail})")
print("=" * 70)

# %% Cell 2: Resolve Input Dataset & Scored Candidates
import zipfile
from src.utils.io import read_json
from src.data.loader import load_records

possible_locations = [
    Path(os.environ.get("P2_RUN_DIR", "")),
    Path(os.environ.get("P1_HANDOFF_DIR", "")),
    Path("/kaggle/input/datasets/duymcminh/r2ai-phase3-reranker-model-test-dataset"),
    Path("/kaggle/input/r2ai-phase3-reranker-model-test-dataset"),
    Path("outputs/bge_reranked_run"),
    Path("outputs/p1_p2_handoff_qwen3"),
    Path("data/p1_p2_handoff_qwen3"),
    Path("../outputs/bge_reranked_run"),
]

INPUT_DIR = None
for loc in possible_locations:
    if loc and loc.exists():
        if loc.is_dir() and ((loc / "reranked.jsonl").exists() or (loc / "candidates.jsonl").exists()):
            INPUT_DIR = loc
            break
        if loc.is_dir():
            matches = list(loc.rglob("candidates.jsonl")) + list(loc.rglob("reranked.jsonl"))
            if matches:
                INPUT_DIR = matches[0].parent
                break
        zips = list(loc.glob("*.zip")) if loc.is_dir() else ([loc] if loc.suffix == ".zip" else [])
        if zips:
            extract_target = WORKING_DIR / "handoff_data"
            extract_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(extract_target)
            matches = list(extract_target.rglob("candidates.jsonl")) + list(extract_target.rglob("reranked.jsonl"))
            if matches:
                INPUT_DIR = matches[0].parent
                break

if INPUT_DIR is None:
    INPUT_DIR = Path("outputs/bge_reranked_run")

print(f"Candidate / Scored Input Directory: {INPUT_DIR.resolve()}")
reranked_file = INPUT_DIR / "reranked.jsonl"
candidates_file = INPUT_DIR / "candidates.jsonl"
labels_file = INPUT_DIR / "labels.json"
registry_file = INPUT_DIR / "registry.json"

scored_file = reranked_file if (reranked_file.exists() and reranked_file.stat().st_size > 0) else candidates_file

if scored_file.exists() and labels_file.exists():
    scored_records = load_records(scored_file)
    labels = read_json(labels_file)
    registry = read_json(registry_file) if registry_file.exists() else {}
    print(f"Loaded {len(scored_records)} scored queries and {len(labels)} ground-truth labels from {scored_file.name}")
else:
    scored_records, labels, registry = None, None, None
    print(f"[WARNING] Input files missing in {INPUT_DIR.resolve()}. Please attach dataset.")

# %% Cell 3: (Optional) Run Reranker Scoring if Raw Candidates Provided (P2-01)
OUTPUT_DIR = Path(os.environ.get("P2_OUTPUT_DIR", "outputs/p2_threshold_sweep_run"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if scored_records and "rerank_score" not in (scored_records[0].get("candidates", [{}])[0]):
    print("Candidates have baseline retrieval scores. Checking if reranker inference is required...")
    # Candidates already have fusion/BM25/dense scores usable as score baseline.
    # To run BGE Reranker v2 M3 on GPU:
    # from src.pipeline.rerank import run_reranking
    # run_reranking(INPUT_DIR, OUTPUT_DIR)
    # scored_records = load_records(OUTPUT_DIR / "reranked.jsonl")

# %% Cell 4: Execute Threshold Sweep & Plateau Analysis (P2-03 Core)
from src.scoring.sweep import sweep_reranker_thresholds

if scored_records and labels:
    mapping = registry.get("internal_to_official", {})
    chunk_to_doc = registry.get("chunk_to_doc")

    print("Running P2-03 Threshold Sweep over validation split...")
    sweep_df, plateau_info, recommended_config = sweep_reranker_thresholds(
        scored_records=scored_records,
        labels=labels,
        thresholds=None,   # Auto-detects min/max score and generates 50 steps
        steps=50,
        fallback=0,        # Pure thresholding for P2-03
        max_chunks=None,   # Uncapped for P2-03
        internal_to_official=mapping,
        chunk_to_doc=chunk_to_doc,
        tolerance=0.01,    # 1% relative tolerance for plateau
    )

    # Export deliverable threshold_sweep.csv
    csv_path = OUTPUT_DIR / "threshold_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 76)
    print("      P2-03: THRESHOLD SWEEP & PLATEAU IDENTIFICATION REPORT")
    print("=" * 76)
    print(f"  Deliverable CSV File   : {csv_path.resolve()}")
    print(f"  Total Grid Steps       : {len(sweep_df)}")
    print(f"  Peak Validation F2     : {plateau_info['best_f2']:.4f}")
    print(f"  Plateau Region [1%]    : [{plateau_info['plateau_min_threshold']}, {plateau_info['plateau_max_threshold']}]")
    print(f"  Plateau Width          : {plateau_info['plateau_width']}")
    print(f"  Plateau Config Count   : {plateau_info['plateau_count']} threshold steps")
    print(f"  Plateau Mean F2        : {plateau_info['plateau_mean_f2']:.4f}")
    print(f"  Recommended Threshold  : {recommended_config['threshold']} (Median / Robust Center)")
    print(f"  Metrics at Rec. Cutoff : F2={recommended_config['macro_f2']:.4f} | R={recommended_config['macro_recall']:.4f} | P={recommended_config['macro_precision']:.4f}")
    print(f"  Avg Chunks per Query   : {recommended_config['avg_chunks_per_query']:.1f}")
    print(f"  Zero-Chunk Queries     : {recommended_config['zero_chunk_queries']} (Cơ sở trực tiếp cho P2-04 Fallback)")
    print("=" * 76)

    print("\nTop 5 Evaluated Thresholds (Ranked by Macro F2):")
    cols = ["threshold", "macro_f2", "macro_recall", "macro_precision", "avg_chunks_per_query", "zero_chunk_queries"]
    print(sweep_df.head(5)[cols].to_string(index=False))
else:
    sweep_df, plateau_info, recommended_config = None, None, None
    print("[WARNING] Cannot run sweep: data not loaded.")

# %% Cell 5: Interactive Visualizations (Plateau & Precision-Recall Curves)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is not None and sweep_df is not None:
    df_sorted = sweep_df.sort_values("threshold").dropna(subset=["threshold"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Curve 1: F2, Recall, Precision vs Threshold with Shaded Plateau
    ax1.plot(df_sorted["threshold"], df_sorted["macro_f2"], "b-", linewidth=2.5, label="Macro F2 (Primary)")
    ax1.plot(df_sorted["threshold"], df_sorted["macro_recall"], "g--", linewidth=1.8, label="Macro Recall")
    ax1.plot(df_sorted["threshold"], df_sorted["macro_precision"], "r:", linewidth=1.8, label="Macro Precision")

    p_min = plateau_info.get("plateau_min_threshold")
    p_max = plateau_info.get("plateau_max_threshold")
    p_rec = plateau_info.get("recommended_threshold")

    if p_min is not None and p_max is not None:
        ax1.axvspan(p_min, p_max, color="yellow", alpha=0.3, label=f"Plateau [{p_min:.3f}, {p_max:.3f}]")
    if p_rec is not None:
        ax1.axvline(p_rec, color="blue", linestyle="-.", linewidth=2, label=f"Recommended ({p_rec:.3f})")

    ax1.set_title("P2-03: Macro Metrics vs Cutoff Threshold", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cutoff Threshold", fontsize=10)
    ax1.set_ylabel("Metric Score", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best")

    # Curve 2: Average Chunks per Query & Zero-Chunk Query Count
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(df_sorted["threshold"], df_sorted["avg_chunks_per_query"], "m-", linewidth=2.2, label="Avg Chunks / Query")
    l2 = ax2_twin.plot(df_sorted["threshold"], df_sorted["zero_chunk_queries"], "k:", linewidth=2.0, label="Zero-Chunk Queries")

    if p_rec is not None:
        ax2.axvline(p_rec, color="blue", linestyle="-.", linewidth=1.5)

    ax2.set_title("Chunk Volume & Zero-Output Query Risk vs Threshold", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cutoff Threshold", fontsize=10)
    ax2.set_ylabel("Avg Chunks / Query", color="m", fontsize=10)
    ax2_twin.set_ylabel("Zero-Chunk Query Count", color="k", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Combine legends for twin axis
    lines = l1 + l2
    labels_leg = [l.get_label() for l in lines]
    ax2.legend(lines, labels_leg, loc="upper right")

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "p2_03_threshold_curves.png"
    plt.savefig(chart_path, dpi=200)
    print(f"Saved visualization plot to: {chart_path.resolve()}")
    plt.show()

# %% Cell 6: Definition of Done & Checklist Verification
if sweep_df is not None and (OUTPUT_DIR / "threshold_sweep.csv").exists():
    print("=" * 70)
    print("  CHECKLIST VERIFICATION: TASK P2-03 (SWEEP THRESHOLD RERANKER)")
    print("=" * 70)
    print("  [x] Quét dải ngưỡng điểm trên validation split: HOÀN THÀNH")
    print("  [x] Đo đạc Precision, Recall, F2 và số chunk/query: HOÀN THÀNH")
    print(f"  [x] File bàn giao Deliverable (threshold_sweep.csv): {OUTPUT_DIR / 'threshold_sweep.csv'}")
    print(f"  [x] Tiêu chí nghiệm thu: Đã xác định được vùng threshold plateau [{plateau_info['plateau_min_threshold']}, {plateau_info['plateau_max_threshold']}]")
    print(f"  [x] Ngưỡng khuyến nghị (Median): {recommended_config['threshold']}")
    print("=" * 70)
    print("\nNext Steps:")
    print("  -> P2-04: Thử Minimum Fallback Top-N để giải quyết các Zero-Chunk Queries.")
    print("  -> P2-05: Thử Maximum Chunk Output Cap để bảo toàn Precision.")
    print("  -> P2-12: Joint Sweep Selector (threshold, fallback, max) tối ưu cuối cùng.")
