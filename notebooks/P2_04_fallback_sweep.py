# %% [markdown]
# # P2-04: Minimum Fallback Top-N Sweep & Ablation Analysis
# ### Thử Nghiệm Cơ Chế Fallback Top-N Khi Dưới Ngưỡng Điểm Theo Chuẩn Checklist P2-04
#
# Notebook này thực thi nhiệm vụ **P2-04** (Phụ trách: **Mạc Duy** - Ưu tiên: **P0**):
# 1. **Khởi tạo môi trường & Workspace**: Kaggle / Local environment setup.
# 2. **Nạp dữ liệu & Ngưỡng điểm P2-03**: Đọc `reranked.jsonl` và ngưỡng khuyến nghị từ `plateau_summary.json` (hoặc `threshold_sweep.csv`).
# 3. **Quét dải Fallback Top-N ($N_{min}$)**: Thử nghiệm $N_{min} \in \{0, 1, 2, 3, 5, 10\}$.
# 4. **Phân tích Ablation**: So sánh đối chiếu trực tiếp giữa cấu hình *Threshold-Only* ($N_{min}=0$) và *Threshold + Fallback* ($N_{min} > 0$).
# 5. **Trực quan hóa**: Vẽ biểu đồ Macro F2, Recall, Precision và số lượng queries bị rỗng kết quả (zero-chunk queries).
# 6. **Xuất sản phẩm bàn giao (Deliverable)**: Tạo `fallback_sweep.csv` và `fallback_ablation.json` thỏa mãn tiêu chuẩn nghiệm thu P2-04.

# %% Cell 1: Environment & Workspace Setup
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p2_04_fallback_sweep")

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
REPO_URL = "https://github.com/Djuybu/cocopila_2.0.git"
REPO_DIR = WORKING_DIR / "cocopila_2.0"

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

cuda_avail = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
print("=" * 70)
print(f"  WORKSPACE READY : {os.getcwd()}")
print(f"  COMPUTE DEVICE  : {device_name} (CUDA: {cuda_avail})")
print("=" * 70)

# %% Cell 2: Load Scored Data & P2-03 Threshold
from src.data.loader import load_records
from src.scoring.ablation import ablation_threshold_vs_fallback, sweep_fallback
from src.utils.io import read_json, write_json

RUN_DIR = WORKING_DIR / "outputs" / "bge_reranked_run"
if not RUN_DIR.exists():
    # Fallback to local test output if run dir does not exist
    RUN_DIR = Path("outputs/bge_reranked_run").resolve()

print(f"Looking for data in: {RUN_DIR}")

# 1. Scored candidates
scored_file = None
for candidate_name in ["reranked.jsonl", "candidates.jsonl"]:
    p = RUN_DIR / candidate_name
    if p.exists() and p.stat().st_size > 0:
        scored_file = p
        break

if not scored_file:
    raise FileNotFoundError(f"No candidate file with content found in {RUN_DIR}")

scored_records = load_records(scored_file)
labels = read_json(RUN_DIR / "labels.json")

registry = read_json(RUN_DIR / "registry.json") if (RUN_DIR / "registry.json").exists() else {}
internal_to_official = registry.get("internal_to_official")
chunk_to_doc = registry.get("chunk_to_doc")

# 2. Extract recommended threshold from P2-03
plateau_file = RUN_DIR / "plateau_summary.json"
threshold = 0.5
if plateau_file.exists():
    p_data = read_json(plateau_file)
    threshold = p_data.get("recommended_config", {}).get("threshold") or p_data.get("plateau_info", {}).get("recommended_threshold", 0.5)
    print(f"Loaded recommended threshold from P2-03: {threshold}")
else:
    print(f"plateau_summary.json not found, using threshold: {threshold}")

print(f"Loaded {len(scored_records)} queries and {len(labels)} ground truth labels.")

# %% Cell 3: Sweep Minimum Fallback Top-N
FALLBACKS = [0, 1, 2, 3, 5, 8, 10]

print(f"\n--- Sweeping Fallbacks {FALLBACKS} at Threshold = {threshold} ---")
sweep_df = sweep_fallback(
    scored_records=scored_records,
    labels=labels,
    threshold=threshold,
    fallbacks=FALLBACKS,
    internal_to_official=internal_to_official,
    chunk_to_doc=chunk_to_doc,
)

print(sweep_df.to_string(index=False))

# %% Cell 4: Ablation Analysis (Threshold-Only vs Fallback)
ablation_report = ablation_threshold_vs_fallback(
    scored_records=scored_records,
    labels=labels,
    threshold=threshold,
    fallbacks=FALLBACKS,
    internal_to_official=internal_to_official,
    chunk_to_doc=chunk_to_doc,
)

baseline = ablation_report["baseline_threshold_only"]
best_fb = ablation_report["best_fallback_config"]
delta = ablation_report["delta"]

print("\n" + "=" * 75)
print("  P2-04 ABLATION REPORT: THRESHOLD-ONLY VS FALLBACK TOP-N")
print("=" * 75)
print(f"  • Baseline (fb=0) : Macro F2={baseline['macro_f2']:.4f} | R={baseline['macro_recall']:.4f} | P={baseline['macro_precision']:.4f} | Zero-Queries={baseline['zero_chunk_queries']}")
print(f"  • Best Fallback   : Top-{best_fb['fallback']} -> Macro F2={best_fb['macro_f2']:.4f} | R={best_fb['macro_recall']:.4f} | P={best_fb['macro_precision']:.4f} | Zero-Queries={best_fb['zero_chunk_queries']}")
print(f"  • Delta F2        : {delta['delta_f2']:+.4f}")
print(f"  • Delta Recall    : {delta['delta_recall']:+.4f}")
print(f"  • Zero-Q Reduced  : {delta['zero_queries_eliminated']}")
print(f"  • Nhận định       : {ablation_report['verdict']}")
print("=" * 75)

# %% Cell 5: Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Metrics vs Fallback
df_sorted = sweep_df.sort_values(by="fallback")
ax1.plot(df_sorted["fallback"], df_sorted["macro_f2"], marker="o", color="blue", linewidth=2, label="Macro F2")
ax1.plot(df_sorted["fallback"], df_sorted["macro_recall"], marker="s", color="green", linestyle="--", label="Macro Recall")
ax1.plot(df_sorted["fallback"], df_sorted["macro_precision"], marker="^", color="orange", linestyle=":", label="Macro Precision")
ax1.axvline(best_fb["fallback"], color="red", linestyle="-.", alpha=0.7, label=f"Best Fallback ({best_fb['fallback']})")
ax1.set_xlabel("Minimum Fallback ($N_{min}$)")
ax1.set_ylabel("Metric Score")
ax1.set_title(f"Ablation Metrics vs Fallback (Threshold = {threshold})")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Zero-Chunk Queries & Avg Chunks
ax2_twin = ax2.twinx()
bars = ax2.bar(df_sorted["fallback"] - 0.15, df_sorted["zero_chunk_queries"], width=0.3, color="crimson", alpha=0.7, label="Zero-Chunk Queries")
lines = ax2_twin.plot(df_sorted["fallback"], df_sorted["avg_chunks_per_query"], marker="D", color="purple", linewidth=2, label="Avg Chunks / Query")

ax2.set_xlabel("Minimum Fallback ($N_{min}$)")
ax2.set_ylabel("Count of Zero-Chunk Queries", color="crimson")
ax2_twin.set_ylabel("Avg Chunks per Query", color="purple")
ax2.set_title("Zero-Chunk Queries & Chunk Volume")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = RUN_DIR / "fallback_ablation_curves.png"
plt.savefig(viz_path, dpi=200)
plt.show()
print(f"Saved visualization to: {viz_path}")

# %% Cell 6: Export Deliverables
csv_path = RUN_DIR / "fallback_sweep.csv"
json_path = RUN_DIR / "fallback_ablation.json"

sweep_df.to_csv(csv_path, index=False)
write_json(json_path, ablation_report)

print(f"Deliverables exported successfully:")
print(f"  1. CSV:  {csv_path} ({len(sweep_df)} configurations)")
print(f"  2. JSON: {json_path}")

# %% Cell 7: Verify Checklist Criteria
print("\n" + "=" * 70)
print("  CHECKLIST VERIFICATION: P2-04 (Fallback Selector)")
print("=" * 70)
c1 = csv_path.exists() and csv_path.stat().st_size > 0
c2 = json_path.exists() and json_path.stat().st_size > 0
c3 = "baseline_threshold_only" in ablation_report and "best_fallback_config" in ablation_report
c4 = delta["delta_f2"] >= -0.01  # F2 không giảm đáng kể trên validation

print(f"  [✓] Có file fallback_sweep.csv: {c1}")
print(f"  [✓] Có file fallback_ablation.json: {c2}")
print(f"  [✓] Có báo cáo so sánh đối chiếu threshold-only vs fallback: {c3}")
print(f"  [✓] Tiêu chuẩn F2 không giảm (Delta F2 = {delta['delta_f2']:+.4f}): {c4}")

if c1 and c2 and c3 and c4:
    print("\n  >>> KẾT QUẢ: P2-04 ĐẠT 100% TIÊU CHUẨN CHECKLIST! SẴN SÀNG NGHIỆM THU. <<<")
else:
    print("\n  >>> CẢNH BÁO: Kiểm tra lại các điều kiện nghiệm thu. <<<")
print("=" * 70)
