# %% [markdown]
# # P2-09: Cross-Validation Threshold Tuning
# ### Điều Chỉnh Ngưỡng Điểm Bằng Cross-Validation & Bootstrap Theo Chuẩn Checklist P2-09
#
# Notebook này thực thi nhiệm vụ **P2-09** (Phụ trách: **Mạc Duy** - Ưu tiên: **P1**):
# 1. **Khởi tạo môi trường & Workspace**: Setup môi trường Kaggle / Local.
# 2. **Nạp dữ liệu Scored Candidates**: Đọc `reranked.jsonl` và nhãn `labels.json`.
# 3. **Thực thi K-Fold Cross-Validation**: Phân chia queries thành $K$ folds độc lập, đo đạc mean và std của Macro F2 trên từng ngưỡng điểm.
# 4. **Thực thi Bootstrap Resampling**: Resample nhiều vòng để kiểm định độ bền vững (robustness) của ngưỡng khuyến nghị.
# 5. **Lựa chọn Ngưỡng Ổn Định (Stable Threshold)**: Lựa chọn ngưỡng tối ưu hàm mục tiêu cân bằng giữa hiệu năng cao và phương sai thấp ($\text{mean\_f2} - 0.25 \times \text{std\_f2}$).
# 6. **Trực quan hóa Dải Sai Số (Error Bands)**: Vẽ đường cong Mean F2 cùng khoảng tin cậy $\pm 1 \text{Std}$ và $95\%$ Confidence Interval.
# 7. **Xuất sản phẩm bàn giao (Deliverable)**: Tạo `cv_threshold_sweep.csv` và `cv_threshold_report.json` nghiệm thu tiêu chuẩn P2-09.

# %% Cell 1: Environment & Workspace Setup
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p2_09_cv_threshold")

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

# %% Cell 2: Load Scored Data & Labels
from src.data.loader import load_records
from src.scoring.cv_threshold import cross_validate_threshold
from src.utils.io import read_json, write_json

RUN_DIR = WORKING_DIR / "outputs" / "bge_reranked_run"
if not RUN_DIR.exists():
    RUN_DIR = Path("outputs/bge_reranked_run").resolve()

print(f"Looking for data in: {RUN_DIR}")

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

# Optional best fallback from P2-04
fallback_file = RUN_DIR / "fallback_ablation.json"
fallback = 0
if fallback_file.exists():
    fb_data = read_json(fallback_file)
    fallback = fb_data.get("best_fallback_config", {}).get("fallback", 0)

print(f"Loaded {len(scored_records)} queries and {len(labels)} ground truth labels. Fallback={fallback}")

# %% Cell 3: Execute K-Fold Cross-Validation
N_FOLDS = min(5, len(labels))
print(f"\n--- Running {N_FOLDS}-Fold Cross-Validation Threshold Tuning ---")

kfold_df, kfold_report = cross_validate_threshold(
    scored_records=scored_records,
    labels=labels,
    steps=40,
    fallback=fallback,
    method="kfold",
    n_folds=N_FOLDS,
    seed=42,
    internal_to_official=internal_to_official,
    chunk_to_doc=chunk_to_doc,
)

print("\nTop 5 Most Stable Threshold Configurations (K-Fold):")
print(kfold_df.head(5)[["threshold", "mean_f2", "std_f2", "stability_score", "mean_precision", "mean_recall"]].to_string(index=False))

# %% Cell 4: Execute Bootstrap Resampling Validation
N_ROUNDS = 20
print(f"\n--- Running {N_ROUNDS}-Round Bootstrap Validation ---")

boot_df, boot_report = cross_validate_threshold(
    scored_records=scored_records,
    labels=labels,
    steps=40,
    fallback=fallback,
    method="bootstrap",
    n_rounds=N_ROUNDS,
    seed=100,
    internal_to_official=internal_to_official,
    chunk_to_doc=chunk_to_doc,
)

print("\nTop 5 Most Stable Threshold Configurations (Bootstrap):")
print(boot_df.head(5)[["threshold", "mean_f2", "std_f2", "stability_score", "mean_precision", "mean_recall"]].to_string(index=False))

# %% Cell 5: Compare Stability and Report Summary
k_stable = kfold_report["recommended_stable_threshold"]
k_pure = kfold_report["pure_peak_threshold"]
b_stable = boot_report["recommended_stable_threshold"]
oof = kfold_report["out_of_fold_generalization"]

print("\n" + "=" * 75)
print("  P2-09 CROSS-VALIDATION SUMMARY & STABILITY REPORT")
print("=" * 75)
print(f"  • K-Fold ({N_FOLDS} folds) Stable Th   : {k_stable['threshold']} -> Mean F2 = {k_stable['mean_f2']:.4f} ± {k_stable['std_f2']:.4f}")
print(f"  • K-Fold 95% Confidence Interval  : [{k_stable['confidence_interval_95'][0]:.4f}, {k_stable['confidence_interval_95'][1]:.4f}]")
print(f"  • K-Fold Peak (Non-Regularized)   : {k_pure['threshold']} -> Mean F2 = {k_pure['mean_f2']:.4f} ± {k_pure['std_f2']:.4f}")
print(f"  • Bootstrap ({N_ROUNDS} rounds) Stable Th : {b_stable['threshold']} -> Mean F2 = {b_stable['mean_f2']:.4f} ± {b_stable['std_f2']:.4f}")
print(f"  • Out-of-Fold Generalization F2   : {oof['mean_oof_f2']:.4f} ± {oof['std_oof_f2']:.4f}")
print("=" * 75)

# %% Cell 6: Visualization (Error Bands & Confidence Intervals)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: K-Fold Mean F2 with Std Band
df_k_sorted = kfold_df.sort_values(by="threshold")
ax1.plot(df_k_sorted["threshold"], df_k_sorted["mean_f2"], color="blue", linewidth=2, label="Mean F2")
ax1.fill_between(
    df_k_sorted["threshold"],
    df_k_sorted["mean_f2"] - df_k_sorted["std_f2"],
    df_k_sorted["mean_f2"] + df_k_sorted["std_f2"],
    color="blue",
    alpha=0.2,
    label="± 1 Std Band",
)
ax1.axvline(k_stable["threshold"], color="red", linestyle="--", linewidth=1.5, label=f"Stable Th ({k_stable['threshold']})")
ax1.set_xlabel("Reranker Threshold")
ax1.set_ylabel("Macro F2")
ax1.set_title(f"K-Fold Cross-Validation ({N_FOLDS} Folds)")
ax1.legend(loc="lower left")
ax1.grid(True, alpha=0.3)

# Plot 2: Bootstrap Mean F2 with Std Band
df_b_sorted = boot_df.sort_values(by="threshold")
ax2.plot(df_b_sorted["threshold"], df_b_sorted["mean_f2"], color="teal", linewidth=2, label="Bootstrap Mean F2")
ax2.fill_between(
    df_b_sorted["threshold"],
    df_b_sorted["mean_f2"] - df_b_sorted["std_f2"],
    df_b_sorted["mean_f2"] + df_b_sorted["std_f2"],
    color="teal",
    alpha=0.2,
    label="± 1 Std Band",
)
ax2.axvline(b_stable["threshold"], color="darkorange", linestyle="--", linewidth=1.5, label=f"Stable Th ({b_stable['threshold']})")
ax2.set_xlabel("Reranker Threshold")
ax2.set_ylabel("Macro F2")
ax2.set_title(f"Bootstrap Resampling ({N_ROUNDS} Rounds)")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = RUN_DIR / "cv_threshold_error_bands.png"
plt.savefig(viz_path, dpi=200)
plt.show()
print(f"Saved visualization to: {viz_path}")

# %% Cell 7: Export Deliverables
csv_path = RUN_DIR / "cv_threshold_sweep.csv"
json_path = RUN_DIR / "cv_threshold_report.json"

kfold_df.to_csv(csv_path, index=False)
write_json(json_path, kfold_report)

print(f"Deliverables exported successfully:")
print(f"  1. CSV:  {csv_path} ({len(kfold_df)} threshold steps)")
print(f"  2. JSON: {json_path}")

# %% Cell 8: Verify Checklist Criteria
print("\n" + "=" * 70)
print("  CHECKLIST VERIFICATION: P2-09 (CV Threshold Report)")
print("=" * 70)
c1 = csv_path.exists() and csv_path.stat().st_size > 0
c2 = json_path.exists() and json_path.stat().st_size > 0
c3 = "mean_f2" in k_stable and "std_f2" in k_stable
c4 = k_stable["std_f2"] >= 0.0 and len(k_stable["confidence_interval_95"]) == 2

print(f"  [✓] Có file cv_threshold_sweep.csv: {c1}")
print(f"  [✓] Có file cv_threshold_report.json: {c2}")
print(f"  [✓] Báo cáo mean/std F2 ({k_stable['mean_f2']:.4f} ± {k_stable['std_f2']:.4f}): {c3}")
print(f"  [✓] Xác định ngưỡng ổn định (threshold={k_stable['threshold']}): {c4}")

if c1 and c2 and c3 and c4:
    print("\n  >>> KẾT QUẢ: P2-09 ĐẠT 100% TIÊU CHUẨN CHECKLIST! SẴN SÀNG NGHIỆM THU. <<<")
else:
    print("\n  >>> CẢNH BÁO: Kiểm tra lại các điều kiện nghiệm thu. <<<")
print("=" * 70)
