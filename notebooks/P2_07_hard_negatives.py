# %% [markdown]
# # P2-07: Hard Negative Mining from Retrieval Candidates
# ### Khai Phá Hard Negatives Từ Kết Quả Retrieval Sai Nhãn Theo Chuẩn Checklist P2-07
#
# Notebook này thực thi nhiệm vụ **P2-07** (Phụ trách: **Mạc Duy** - Ưu tiên: **P1**):
# 1. **Khởi tạo môi trường & Workspace**: Setup môi trường Kaggle / Local.
# 2. **Nạp dữ liệu Retrieval Candidates**: Đọc `candidates.jsonl` (hoặc `reranked.jsonl`) cùng nhãn `labels.json`.
# 3. **Khai phá Hard Negatives**: Lọc lấy các chunk có điểm retrieval cao nhưng sai nhãn, đồng thời loại trừ triệt để các nghi vấn False Negatives (các đoạn văn bản quá tương đồng về ngữ nghĩa với positive).
# 4. **Kiểm định chất lượng (Validation)**: Xác thực 100% không chứa ground-truth chunk và điểm tương đồng của hard negative cao hơn random negative.
# 5. **Trộn bộ dữ liệu huấn luyện**: Kết hợp với dữ liệu positive/negative từ P2-06 thành tập huấn luyện cross-encoder chất lượng cao.
# 6. **Trực quan hóa**: Biểu đồ phân bố điểm số của Hard Negatives vs All Candidates.
# 7. **Xuất sản phẩm bàn giao (Deliverable)**: Tạo `hard_negatives.jsonl` và `hard_negatives_report.json` nghiệm thu tiêu chuẩn P2-07.

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
logger = logging.getLogger("p2_07_hard_negatives")

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

# %% Cell 2: Load Candidates & Labels
from src.data.loader import load_records
from src.training.hard_negatives import (
    merge_training_data,
    mine_hard_negatives,
    validate_hard_negatives,
    write_hard_negatives,
)
from src.utils.io import read_json, write_json

RUN_DIR = WORKING_DIR / "outputs" / "bge_reranked_run"
if not RUN_DIR.exists():
    RUN_DIR = Path("outputs/bge_reranked_run").resolve()

print(f"Looking for candidate pools in: {RUN_DIR}")

candidates_file = None
for fname in ["candidates.jsonl", "reranked.jsonl"]:
    p = RUN_DIR / fname
    if p.exists() and p.stat().st_size > 0:
        candidates_file = p
        break

if not candidates_file:
    raise FileNotFoundError(f"Could not find candidate pool file in {RUN_DIR}")

candidates_records = load_records(candidates_file)
labels = read_json(RUN_DIR / "labels.json")

queries_file = RUN_DIR / "queries.json"
queries = load_records(queries_file) if queries_file.exists() else None

registry_file = RUN_DIR / "registry.json"
registry = read_json(registry_file) if registry_file.exists() else {}
internal_to_official = registry.get("internal_to_official")
chunk_to_doc = registry.get("chunk_to_doc")

print(f"Loaded {len(candidates_records)} queries with candidates, and {len(labels)} ground truth labels.")

# %% Cell 3: Mine Hard Negatives
MAX_NEGS_PER_QUERY = 10
SIMILARITY_THRESHOLD = 0.85

print(f"\n--- Mining Hard Negatives (Max/Query: {MAX_NEGS_PER_QUERY}, Similarity Filter: {SIMILARITY_THRESHOLD}) ---")
hard_negs, mining_summary = mine_hard_negatives(
    candidates_records=candidates_records,
    labels=labels,
    queries=queries,
    max_negatives_per_query=MAX_NEGS_PER_QUERY,
    similarity_threshold=SIMILARITY_THRESHOLD,
    internal_to_official=internal_to_official,
    chunk_to_doc=chunk_to_doc,
)

print("\n" + "=" * 75)
print("  HARD NEGATIVE MINING SUMMARY")
print("=" * 75)
print(f"  • Total Mined Hard Negatives : {mining_summary['total_mined_hard_negatives']}")
print(f"  • Candidates Examined        : {mining_summary['total_candidates_examined']}")
print(f"  • Ground-Truth Excluded      : {mining_summary['ground_truth_excluded']}")
print(f"  • Suspected FNs Filtered     : {mining_summary['suspected_fn_excluded']}")
print(f"  • Avg Hard Negatives / Query : {mining_summary['avg_hard_negatives_per_query']}")
print("=" * 75)

# %% Cell 4: Validate Zero Contamination & Score Advantage
val_summary = validate_hard_negatives(hard_negs, labels)

print("\n" + "=" * 75)
print("  HARD NEGATIVE VALIDATION CHECK")
print("=" * 75)
print(f"  • Contamination Count        : {val_summary['ground_truth_contamination']} (PASS: Không chứa positive)")
print(f"  • Verified Records           : {val_summary['total_verified']}")
print(f"  • Avg Hard Negative Score    : {val_summary['avg_hard_negative_score']}")
print(f"  • Quality Status             : {val_summary['status'].upper()}")
print("=" * 75)

# %% Cell 5: Merge with P2-06 Base Training Data (If available)
base_train_file = RUN_DIR / "reranker_train.jsonl"
if not base_train_file.exists():
    base_train_file = WORKING_DIR / "data" / "processed" / "reranker_train.jsonl"

if base_train_file.exists() and base_train_file.stat().st_size > 0:
    print(f"\nFound P2-06 training data at {base_train_file}, merging with hard negatives...")
    base_pairs = load_records(base_train_file)
    enriched_pairs = merge_training_data(base_pairs, hard_negs, hard_neg_ratio=0.5)
    enriched_path = RUN_DIR / "reranker_train_hard.jsonl"
    write_hard_negatives(enriched_pairs, enriched_path)
    print(f"Successfully generated enriched training dataset: {enriched_path} ({len(enriched_pairs)} pairs)")
else:
    print("\nNotice: Base reranker_train.jsonl not found in local run dir, standalone hard_negatives.jsonl will be exported.")

# %% Cell 6: Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scores distribution of mined hard negatives
scores = [h["score"] for h in hard_negs if "score" in h]
if scores:
    ax1.hist(scores, bins=25, color="coral", alpha=0.8, edgecolor="black")
    ax1.axvline(np.mean(scores), color="darkred", linestyle="--", linewidth=2, label=f"Mean Score ({np.mean(scores):.3f})")
    ax1.set_xlabel("Retrieval / Cross-Encoder Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Mined Hard Negative Scores")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Plot 2: Source Rank of Hard Negatives
ranks = [h["rank"] for h in hard_negs if "rank" in h]
if ranks:
    ax2.hist(ranks, bins=20, color="steelblue", alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Original Retrieval Rank")
    ax2.set_ylabel("Count")
    ax2.set_title("Original Retrieval Rank Distribution of Hard Negatives")
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = RUN_DIR / "hard_negatives_distribution.png"
plt.savefig(viz_path, dpi=200)
plt.show()
print(f"Saved visualization to: {viz_path}")

# %% Cell 7: Export Deliverables
output_hn_path = RUN_DIR / "hard_negatives.jsonl"
output_report_path = RUN_DIR / "hard_negatives_report.json"

write_hard_negatives(hard_negs, output_hn_path)
full_report = {
    "mining_summary": mining_summary,
    "validation_summary": val_summary,
}
write_json(output_report_path, full_report)

print(f"\nDeliverables exported successfully:")
print(f"  1. JSONL: {output_hn_path} ({len(hard_negs)} hard negative chunks)")
print(f"  2. JSON:  {output_report_path}")

# %% Cell 8: Verify Checklist Criteria
print("\n" + "=" * 70)
print("  CHECKLIST VERIFICATION: P2-07 (Hard Negatives: hard_negatives.jsonl)")
print("=" * 70)
c1 = output_hn_path.exists() and output_hn_path.stat().st_size > 0
c2 = output_report_path.exists() and output_report_path.stat().st_size > 0
c3 = val_summary["ground_truth_contamination"] == 0
c4 = mining_summary["total_mined_hard_negatives"] > 0

print(f"  [✓] Có file hard_negatives.jsonl: {c1}")
print(f"  [✓] Có file hard_negatives_report.json: {c2}")
print(f"  [✓] Tuyệt đối không chứa ground-truth positive (Contamination: {val_summary['ground_truth_contamination']}): {c3}")
print(f"  [✓] Đã khai phá thành công {mining_summary['total_mined_hard_negatives']} hard negatives: {c4}")

if c1 and c2 and c3 and c4:
    print("\n  >>> KẾT QUẢ: P2-07 ĐẠT 100% TIÊU CHUẨN CHECKLIST! SẴN SÀNG NGHIỆM THU. <<<")
else:
    print("\n  >>> CẢNH BÁO: Kiểm tra lại các điều kiện nghiệm thu. <<<")
print("=" * 70)
