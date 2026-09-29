# %% [markdown]
# # P2-05: Output Cap Max-N Sweep & Tradeoff Analysis
# ### Thử Nghiệm Giới Hạn Số Chunk Đầu Ra (Max-N) & Phân Tích Đánh Đổi Precision/Recall Theo Checklist P2-05
#
# Notebook này thực thi nhiệm vụ **P2-05** (Phụ trách: **Mạc Duy** - Ưu tiên: **P1**):
# 1. **Khởi tạo môi trường & Workspace**: Setup môi trường Kaggle / Local.
# 2. **Nạp dữ liệu & Cấu hình tối ưu từ P2-03, P2-04**: Đọc ngưỡng `threshold` và `fallback` tốt nhất.
# 3. **Quét dải Output Cap Max-N**: Thử nghiệm $M \in \{1, 2, 3, 5, 8, 10, 15, 20, 50, 100\}$.
# 4. **Phân tích đánh đổi (Tradeoff Analysis)**: Đánh giá tác động của việc chặn trên số chunk lên Precision, Recall và Macro F2.
# 5. **Trực quan hóa**: Biểu đồ kép Precision vs Recall và độ dốc F2 theo từng ngưỡng Max-N.
# 6. **Xuất sản phẩm bàn giao (Deliverable)**: Tạo `max_output_sweep.csv` và `max_output_tradeoff.json` nghiệm thu tiêu chuẩn P2-05.

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
logger = logging.getLogger("p2_05_max_output_sweep")

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

# %% Cell 2: Load Scored Data & Fixed Threshold / Fallback
from src.data.loader import load_records
from src.scoring.ablation import analyze_precision_recall_tradeoff, sweep_max_output
from src.utils.io import read_json, write_json

RUN_DIR = WORKING_DIR / "outputs" / "bge_reranked_run"
if not RUN_DIR.exists():
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

# 3. Extract recommended fallback from P2-04
fallback_file = RUN_DIR / "fallback_ablation.json"
fallback = 0
if fallback_file.exists():
    f_data = read_json(fallback_file)
    fallback = f_data.get("best_fallback_config", {}).get("fallback", 0)

print(f"Baseline Settings: Threshold={threshold}, Fallback={fallback}")
print(f"Loaded {len(scored_records)} queries and {len(labels)} ground truth labels.")

# %% Cell 3: Sweep Max Output Cap (Max-N)
MAX_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 50, 100]

print(f"\n--- Sweeping Max Output Cap across {MAX_VALUES} ---")
sweep_df = sweep_max_output(
    scored_records=scored_records,
    labels=labels,
    threshold=threshold,
    fallback=fallback,
    maximums=MAX_VALUES,
    internal_to_official=internal_to_official,
    chunk_to_doc=chunk_to_doc,
)

print(sweep_df.to_string(index=False))

# %% Cell 4: Tradeoff Analysis
tradeoff_report = analyze_precision_recall_tradeoff(sweep_df)
tradeoff_report["threshold"] = threshold
tradeoff_report["fallback"] = fallback

best_max = tradeoff_report["best_max_chunks"]
unlim = tradeoff_report["unlimited_baseline"]
trade = tradeoff_report["tradeoff"]

print("\n" + "=" * 75)
print("  P2-05 TRADEOFF REPORT: PRECISION / RECALL / F2 IMPACT")
print("=" * 75)
print(f"  • Unlimited Cap ({unlim['max_chunks']}) : Macro F2={unlim['macro_f2']:.4f} | P={unlim['macro_precision']:.4f} | R={unlim['macro_recall']:.4f}")
print(f"  • Best Cap ({best_max})       : Macro F2={tradeoff_report['best_macro_f2']:.4f} | P={tradeoff_report['best_macro_precision']:.4f} | R={tradeoff_report['best_macro_recall']:.4f}")
print(f"  • Precision Gain   : {trade['precision_gain']:+.4f}")
print(f"  • Recall Loss      : {trade['recall_loss']:.4f}")
print(f"  • F2 Delta         : {trade['f2_delta']:+.4f}")
print(f"  • Nhận định        : {tradeoff_report['analysis']}")
print("=" * 75)

# %% Cell 5: Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

df_sorted = sweep_df.sort_values(by="max_chunks")

# Subplot 1: Precision vs Recall Tradeoff Curves
ax1.plot(df_sorted["max_chunks"], df_sorted["macro_precision"], marker="^", color="orange", linewidth=2, label="Macro Precision")
ax1.plot(df_sorted["max_chunks"], df_sorted["macro_recall"], marker="s", color="green", linewidth=2, label="Macro Recall")
ax1.axvline(best_max, color="red", linestyle="--", alpha=0.7, label=f"Best Cap ({best_max})")
ax1.set_xlabel("Output Cap ($Max-N$)")
ax1.set_ylabel("Metric Score")
ax1.set_title("Precision vs Recall Tradeoff vs Max-N")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Macro F2 vs Avg Chunks per Query
ax2_twin = ax2.twinx()
p_f2 = ax2.plot(df_sorted["max_chunks"], df_sorted["macro_f2"], marker="o", color="blue", linewidth=2, label="Macro F2")
p_vol = ax2_twin.plot(df_sorted["max_chunks"], df_sorted["avg_chunks_per_query"], marker="D", color="purple", linestyle=":", label="Avg Chunks / Query")

ax2.axvline(best_max, color="red", linestyle="--", alpha=0.7, label=f"Best Cap ({best_max})")
ax2.set_xlabel("Output Cap ($Max-N$)")
ax2.set_ylabel("Macro F2", color="blue")
ax2_twin.set_ylabel("Avg Chunks per Query", color="purple")
ax2.set_title("Macro F2 & Chunk Output Volume")

# Combine legends
lines = p_f2 + p_vol + [plt.Line2D([0], [0], color="red", linestyle="--")]
labels_leg = ["Macro F2", "Avg Chunks", f"Best Cap ({best_max})"]
ax2.legend(lines, labels_leg, loc="lower right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = RUN_DIR / "max_output_tradeoff_curves.png"
plt.savefig(viz_path, dpi=200)
plt.show()
print(f"Saved visualization to: {viz_path}")

# %% Cell 6: Export Deliverables
csv_path = RUN_DIR / "max_output_sweep.csv"
json_path = RUN_DIR / "max_output_tradeoff.json"

sweep_df.to_csv(csv_path, index=False)
write_json(json_path, tradeoff_report)

print(f"Deliverables exported successfully:")
print(f"  1. CSV:  {csv_path} ({len(sweep_df)} configurations)")
print(f"  2. JSON: {json_path}")

# %% Cell 7: Verify Checklist Criteria
print("\n" + "=" * 70)
print("  CHECKLIST VERIFICATION: P2-05 (Max-Output Selector)")
print("=" * 70)
c1 = csv_path.exists() and csv_path.stat().st_size > 0
c2 = json_path.exists() and json_path.stat().st_size > 0
c3 = "unlimited_baseline" in tradeoff_report and "best_max_chunks" in tradeoff_report
c4 = "precision_gain" in trade and "recall_loss" in trade

print(f"  [✓] Có file max_output_sweep.csv: {c1}")
print(f"  [✓] Có file max_output_tradeoff.json: {c2}")
print(f"  [✓] Có phân tích tác động so với unlimited: {c3}")
print(f"  [✓] Phân tích định lượng Precision/Recall/F2 ({trade['precision_gain']:+.4f} P, {trade['recall_loss']:.4f} R): {c4}")

if c1 and c2 and c3 and c4:
    print("\n  >>> KẾT QUẢ: P2-05 ĐẠT 100% TIÊU CHUẨN CHECKLIST! SẴN SÀNG NGHIỆM THU. <<<")
else:
    print("\n  >>> CẢNH BÁO: Kiểm tra lại các điều kiện nghiệm thu. <<<")
print("=" * 70)
