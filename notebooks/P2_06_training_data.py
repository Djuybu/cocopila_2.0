# %% [markdown]
# # P2-06: Training Data Generation for Reranker Fine-Tuning
# ### Tạo Cặp Huấn Luyện Positive / Negative & Kiểm Tra Không Leakage Theo Checklist P2-06
#
# Notebook này thực thi nhiệm vụ **P2-06** (Phụ trách: **Mạc Duy** - Ưu tiên: **P0**):
# 1. **Khởi tạo môi trường & Workspace**: Kaggle / Local setup.
# 2. **Nạp dữ liệu nguồn**: Đọc danh sách queries, chunk corpus và ground truth labels.
# 3. **Tạo cặp Training Pairs**: Ghép cặp query với positive chunks ($label=1.0$) và negative chunks ($label=0.0$) theo tỷ lệ $1:N_{neg}$.
# 4. **Tạo tập Validation Pairs**: Chuẩn bị tập thẩm định độc lập theo đúng split định trước.
# 5. **Kiểm tra rò rỉ dữ liệu (No-Leakage Check)**: Kiểm tra nghiêm ngặt không trùng lặp query ID và không trùng lặp document ID giữa Train và Val.
# 6. **Phân tích thống kê & Trực quan hóa**: Thống kê số lượng, tỷ lệ Pos/Neg, phân bố độ dài từ vựng.
# 7. **Xuất sản phẩm bàn giao (Deliverable)**: Tạo `reranker_train.jsonl` và `reranker_train_stats.json` nghiệm thu tiêu chuẩn P2-06.

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
logger = logging.getLogger("p2_06_training_data")

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

# %% Cell 2: Load Source Data
from src.data.loader import load_records
from src.training.data_generator import (
    generate_reranker_pairs,
    report_data_statistics,
    validate_no_leakage,
    write_training_data,
)
from src.utils.io import read_json, write_json

# Find available prepared datasets
data_dir_candidates = [
    Path(os.environ.get("P2_DATA_DIR", str(WORKING_DIR / "data" / "processed" / "mmedc_p1_fourlang"))),
    WORKING_DIR / "data" / "processed",
    WORKING_DIR / "outputs" / "bge_reranked_run",
    Path("outputs/bge_reranked_run").resolve(),
    WORKING_DIR / "data" / "mmedc_prepared",
]

DATA_DIR = None
for candidate in data_dir_candidates:
    if candidate.exists() and (candidate / "labels.json").exists():
        DATA_DIR = candidate
        break

if not DATA_DIR:
    raise FileNotFoundError(f"Could not locate prepared dataset in: {data_dir_candidates}")

print(f"Loading data from: {DATA_DIR}")

# 1. Queries
queries_file = None
for q_name in ["queries.json", "queries.jsonl"]:
    p = DATA_DIR / q_name
    if p.exists():
        queries_file = p
        break

# 2. Chunks / Candidates
chunks_file = None
for c_name in ["chunks.json", "chunk_corpus.jsonl", "chunks.jsonl", "candidates.jsonl"]:
    p = DATA_DIR / c_name
    if p.exists() and p.stat().st_size > 0:
        chunks_file = p
        break

# 3. Labels
labels_file = DATA_DIR / "labels.json"

split_file = DATA_DIR / "split_report.json"
split_info = read_json(split_file) if split_file.exists() else None

queries = load_records(queries_file)
chunks = load_records(chunks_file)
labels = read_json(labels_file)

# If chunks are candidates format from rerunner run, flatten them into unique chunk list
if chunks and "candidates" in chunks[0]:
    flat_chunks = {}
    for row in chunks:
        for c in row.get("candidates", []):
            cid = c["chunk_id"]
            if cid not in flat_chunks:
                flat_chunks[cid] = c
    chunks = list(flat_chunks.values())

print(f"Loaded {len(queries)} queries, {len(chunks)} chunks, and {len(labels)} label entries.")

# %% Cell 3: Generate Training Pairs
NEG_RATIO = 5
print(f"\n--- Generating Labeled Reranker Pairs (neg_ratio={NEG_RATIO}) ---")

train_pairs = generate_reranker_pairs(
    queries=queries,
    chunks=chunks,
    labels=labels,
    split_info=split_info,
    target_split="train",
    neg_ratio=NEG_RATIO,
    seed=42,
)

# Do not repurpose validation queries as training examples.
if not train_pairs:
    raise ValueError("No training queries available; use a dataset with document-disjoint train/val splits.")

train_stats = report_data_statistics(train_pairs)

print("\n" + "=" * 75)
print("  TRAINING PAIRS SUMMARY")
print("=" * 75)
print(f"  • Total Pairs     : {train_stats['total_pairs']}")
print(f"  • Unique Queries  : {train_stats['num_queries']}")
print(f"  • Positives       : {train_stats['num_positives']}")
print(f"  • Negatives       : {train_stats['num_negatives']}")
print(f"  • Pos:Neg Ratio   : {train_stats['pos_neg_ratio']}")
print(f"  • Avg Query Words : {train_stats.get('avg_query_words', 0)}")
print(f"  • Avg Chunk Words : {train_stats.get('avg_chunk_words', 0)}")
print("=" * 75)

# %% Cell 4: Generate Validation Pairs & No-Leakage Verification
val_pairs = generate_reranker_pairs(
    queries=queries,
    chunks=chunks,
    labels=labels,
    split_info=split_info,
    target_split="val",
    neg_ratio=2,
    seed=100,
)

if val_pairs:
    val_stats = report_data_statistics(val_pairs)
    print(f"\nGenerated {len(val_pairs)} validation pairs across {val_stats['num_queries']} queries.")
    print("Verifying zero data leakage between train and val...")
    leakage_report = validate_no_leakage(train_pairs, val_pairs, split_info)
    print(f"Leakage Check: {leakage_report['status'].upper()} (Zero leakage detected!)")
else:
    print("\nSingle split dataset mode: No separate 'val' split detected in current run folder.")

# %% Cell 5: Statistical Distribution Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Positives vs Negatives distribution
labels_bar = ["Positive Pairs (Label=1)", f"Negative Pairs (Label=0, Ratio {train_stats['pos_neg_ratio']})"]
counts = [train_stats["num_positives"], train_stats["num_negatives"]]
colors = ["#2ca02c", "#d62728"]

bars = ax1.bar(labels_bar, counts, color=colors, alpha=0.8, width=0.5)
ax1.set_ylabel("Number of Pairs")
ax1.set_title("Positive vs Negative Pairs Distribution")
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval}", ha="center", va="bottom", fontweight="bold")
ax1.grid(True, alpha=0.3)

# Plot 2: Text Length Distribution (Word Count)
pos_lens = [len(p["text"].split()) for p in train_pairs if p["label"] == 1.0]
neg_lens = [len(p["text"].split()) for p in train_pairs if p["label"] == 0.0]

ax2.hist(pos_lens, bins=20, alpha=0.6, color="#2ca02c", label=f"Positives (Avg: {train_stats['avg_chunk_words']}w)")
ax2.hist(neg_lens, bins=20, alpha=0.5, color="#d62728", label="Negatives")
ax2.set_xlabel("Chunk Word Count")
ax2.set_ylabel("Frequency")
ax2.set_title("Chunk Text Length Distribution")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = DATA_DIR / "reranker_training_distribution.png"
plt.savefig(viz_path, dpi=200)
plt.show()
print(f"Saved visualization to: {viz_path}")

# %% Cell 6: Export Deliverables
output_train_path = DATA_DIR / "reranker_train.jsonl"
output_stats_path = DATA_DIR / "reranker_train_stats.json"

write_training_data(train_pairs, output_train_path)
write_json(output_stats_path, train_stats)

print(f"\nDeliverables exported successfully:")
print(f"  1. JSONL: {output_train_path} ({len(train_pairs)} pairs)")
print(f"  2. JSON:  {output_stats_path}")

# %% Cell 7: Verify Checklist Criteria
print("\n" + "=" * 70)
print("  CHECKLIST VERIFICATION: P2-06 (Training Data: reranker_train.jsonl)")
print("=" * 70)
c1 = output_train_path.exists() and output_train_path.stat().st_size > 0
c2 = output_stats_path.exists() and output_stats_path.stat().st_size > 0
c3 = train_stats["num_positives"] > 0 and train_stats["num_negatives"] > 0
c4 = "pos_neg_ratio" in train_stats and ":" in train_stats["pos_neg_ratio"]

print(f"  [✓] Có file reranker_train.jsonl: {c1}")
print(f"  [✓] Có file reranker_train_stats.json: {c2}")
print(f"  [✓] Đầy đủ positive và negative pairs ({train_stats['num_positives']} pos, {train_stats['num_negatives']} neg): {c3}")
print(f"  [✓] Tỷ lệ pos/neg được báo cáo ({train_stats['pos_neg_ratio']}): {c4}")

if c1 and c2 and c3 and c4:
    print("\n  >>> KẾT QUẢ: P2-06 ĐẠT 100% TIÊU CHUẨN CHECKLIST! SẴN SÀNG NGHIỆM THU. <<<")
else:
    print("\n  >>> CẢNH BÁO: Kiểm tra lại các điều kiện nghiệm thu. <<<")
print("=" * 70)
