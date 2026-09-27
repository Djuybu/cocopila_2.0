# %% [markdown]
# Legacy controlled heuristic benchmark; main P2 notebook now consumes P1-11.
# # P2-01: Medical Reranker Comprehensive Evaluation & Benchmarking
# ### BAAI/bge-reranker-v2-m3 vs Qwen/Qwen3-Reranker-0.6B on Kaggle 2×T4 GPUs
#
# This notebook evaluates and compares two premier medical rerankers across 5 evaluation pillars (4 clinical evaluation pillars + resource efficiency):
# 1. **Quantitative IR Performance**: Graded NDCG@3/5/10, MRR@5/10, Recall@10
# 2. **Clinical Safety**: Negation / Contraindication Sensitivity Rate & PICO Population Alignment Rate
# 3. **Score Calibration**: Expected Calibration Error (ECE 10 bins) & Score Separability
# 4. **Cross-lingual Parity**: ΔNDCG@10 and ΔMRR on parallel English/Chinese clinical concepts
# 5. **Resource Efficiency**: Peak VRAM (GB) on Tesla T4, Latency (ms/query), and Throughput (qps)
#
# Hardware target: Kaggle 2×Tesla T4 (16GB GDDR6 each = 32GB total VRAM).

# %% Cell 1: Clone Repository from GitHub
# In Kaggle notebook, runs: !git clone https://github.com/Djuybu/cocopila_2.0.git /kaggle/working/cocopila_2.0
import os
from pathlib import Path
import subprocess

REPO_URL = "https://github.com/Djuybu/cocopila_2.0.git"
REPO_DIR = "/kaggle/working/cocopila_2.0"

if Path("/kaggle").exists():
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    else:
        print(f"Repository already exists at {REPO_DIR}. Pulling latest code...")
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=False)
else:
    print("Local environment detected; repository clone skipped.")

# %% Cell 2: Setup Kaggle Working Directory & Python Path
# In Kaggle notebook, runs: %cd /kaggle/working/cocopila_2.0
import os
import sys
from pathlib import Path
import subprocess

REPO_DIR = "/kaggle/working/cocopila_2.0"

if Path("/kaggle").exists():
    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    if os.path.exists(f"{REPO_DIR}/src") and not os.path.exists("/kaggle/working/src"):
        subprocess.run(["cp", "-r", f"{REPO_DIR}/src", "/kaggle/working/"])
    if os.path.exists(f"{REPO_DIR}/configs") and not os.path.exists("/kaggle/working/configs"):
        subprocess.run(["cp", "-r", f"{REPO_DIR}/configs", "/kaggle/working/"])
else:
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

print(f"Current working directory: {os.getcwd()}")

# %% Cell 3: Install Dependencies on Kaggle
# In Kaggle notebook, runs: !pip install -q qdrant-client sentence-transformers accelerate
import subprocess
import sys

for pkg in ["qdrant-client", "sentence-transformers", "accelerate"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# %% Cell 4: Import Necessary Libraries
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd
import torch

# Vector DB & Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Ensure project root is in sys.path when running on Kaggle or other environments
for p in [Path.cwd(), Path.cwd().parent, Path("/kaggle/working")]:
    if (p / "src").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Medical Reranker Modules
from src.reranking.bge import CrossEncoderReranker
from src.reranking.qwen_reranker import QwenReranker

# %% Cell 5: Environment Setup & Dual-T4 GPU Detection
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reranker_eval")

# Verify CUDA & Kaggle Multi-GPU Environment
num_gpus = torch.cuda.device_count()
logger.info(f"CUDA Available: {torch.cuda.is_available()}, Device Count: {num_gpus}")

gpu_devices = []
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    total_mem_gb = props.total_memory / (1024**3)
    gpu_devices.append(f"cuda:{i} ({props.name}, {total_mem_gb:.2f} GB)")
    logger.info(f"GPU {i}: {gpu_devices[-1]}")

device_reranker_1 = "cuda:0" if num_gpus > 0 else "cpu"
device_reranker_2 = "cuda:1" if num_gpus > 1 else device_reranker_1

logger.info(f"Primary Reranker Device: {device_reranker_1}")
logger.info(f"Secondary Reranker Device: {device_reranker_2}")


def get_vram_usage(device: str) -> float:
    """Return memory allocated in GB for the specified CUDA device."""
    if "cuda" in device and torch.cuda.is_available():
        dev_idx = int(device.split(":")[1]) if ":" in device else 0
        return torch.cuda.max_memory_allocated(dev_idx) / (1024**3)
    return 0.0


def reset_cuda(device: str) -> None:
    """Reset peak memory stats and collect garbage."""
    gc.collect()
    if "cuda" in device and torch.cuda.is_available():
        dev_idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.reset_peak_memory_stats(dev_idx)
        torch.cuda.empty_cache()


# Configuration
DATA_DIR = Path("/kaggle/input/datasets/duymcminh/r2ai-phase3-reranker-model-test-dataset")
if not DATA_DIR.exists():
    # Local fallback for development / testing
    DATA_DIR = Path("./data/test/reranker_eval")
    if not DATA_DIR.exists():
        DATA_DIR = Path("../data/test/reranker_eval")

logger.info(f"Using evaluation data directory: {DATA_DIR.resolve()}")

# %% Cell 6: Load Test Data & Init Dual Qdrant Collections
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

chunks_path = DATA_DIR / "chunks.json"
chunks_meta_path = DATA_DIR / "chunks_meta.json"
minilm_emb_path = DATA_DIR / "embeddings_minilm.npy"
bgem3_emb_path = DATA_DIR / "embeddings_bgem3.npy"

if not chunks_path.exists():
    raise FileNotFoundError(f"Cannot find chunks.json at {chunks_path}")

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

logger.info(f"Loaded {len(chunks)} chunks from corpus.")

# In-memory Qdrant client
qdrant_client = QdrantClient(":memory:")

# Helper function to load or generate embeddings
def get_or_generate_embeddings(npy_path: Path, model_name: str, dim: int) -> np.ndarray:
    if npy_path.exists():
        emb = np.load(npy_path)
        logger.info(f"Loaded existing {npy_path.name} with shape {emb.shape}")
        return emb
    else:
        logger.info(f"{npy_path.name} not found. Generating on the fly using {model_name}...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name, device=device_reranker_1)
        texts = [c["text"] for c in chunks]
        emb = encoder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

embeddings_minilm = get_or_generate_embeddings(minilm_emb_path, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384)
embeddings_bgem3 = get_or_generate_embeddings(bgem3_emb_path, "BAAI/bge-m3", 1024)

# Collection 1: MiniLM-L12-v2 (384-dim)
COLLECTION_MINILM = "medical_chunks_minilm"
if qdrant_client.collection_exists(COLLECTION_MINILM):
    qdrant_client.delete_collection(COLLECTION_MINILM)
qdrant_client.create_collection(
    collection_name=COLLECTION_MINILM,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points_minilm = [
    PointStruct(
        id=idx,
        vector=embeddings_minilm[idx].tolist(),
        payload={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"], "language": c["language"]},
    )
    for idx, c in enumerate(chunks)
]
qdrant_client.upsert(collection_name=COLLECTION_MINILM, points=points_minilm)
logger.info(f"Indexed {len(points_minilm)} vectors in collection '{COLLECTION_MINILM}'.")

# Collection 2: BGE-M3 (1024-dim)
COLLECTION_BGEM3 = "medical_chunks_bgem3"
if qdrant_client.collection_exists(COLLECTION_BGEM3):
    qdrant_client.delete_collection(COLLECTION_BGEM3)
qdrant_client.create_collection(
    collection_name=COLLECTION_BGEM3,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

points_bgem3 = [
    PointStruct(
        id=idx,
        vector=embeddings_bgem3[idx].tolist(),
        payload={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"], "language": c["language"]},
    )
    for idx, c in enumerate(chunks)
]
qdrant_client.upsert(collection_name=COLLECTION_BGEM3, points=points_bgem3)
logger.info(f"Indexed {len(points_bgem3)} vectors in collection '{COLLECTION_BGEM3}'.")

chunk_by_id = {c["chunk_id"]: c for c in chunks}

# %% Cell 7: Load Test Cases & Sanity Checks
test_cases_path = DATA_DIR / "test_cases.json"
with open(test_cases_path, "r", encoding="utf-8") as f:
    test_cases_data = json.load(f)

categories = test_cases_data["categories"]
ir_cases = categories.get("ir_standard", [])
adv_neg_cases = categories.get("adversarial_negation", [])
adv_pico_cases = categories.get("adversarial_pico", [])
cross_lingual_cases = categories.get("cross_lingual", [])

logger.info("=== Test Suite Summary ===")
logger.info(f"Standard IR Cases: {len(ir_cases)} (chn: {sum(1 for c in ir_cases if c['language']=='chn')}, eng: {sum(1 for c in ir_cases if c['language']=='eng')})")
logger.info(f"Adversarial Negation Cases: {len(adv_neg_cases)}")
logger.info(f"Adversarial PICO Cases: {len(adv_pico_cases)}")
logger.info(f"Cross-lingual Concept Pairs: {len(cross_lingual_cases)}")
logger.info(f"Total Test Cases: {test_cases_data.get('total_cases', len(ir_cases)+len(adv_neg_cases)+len(adv_pico_cases)+len(cross_lingual_cases))}")

# %% Cell 8: Evaluation Metrics Module
def dcg_at_k(ranked_ids: List[str], graded_relevance: Dict[str, int], k: int) -> float:
    """Compute Discounted Cumulative Gain at K with graded relevance."""
    score = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        rel = graded_relevance.get(cid, 0)
        score += (2**rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_ids: List[str], graded_relevance: Dict[str, int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    dcg = dcg_at_k(ranked_ids, graded_relevance, k)
    ideal_ranks = sorted(graded_relevance.values(), reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_ranks[:k]))
    return (dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(ranked_ids: List[str], target_id: str, k: int) -> float:
    """Compute Reciprocal Rank at K for target_id."""
    for i, cid in enumerate(ranked_ids[:k]):
        if cid == target_id:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Recall at K."""
    if not relevant_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def sigmoid(x: float) -> float:
    """Numerically safe sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def compute_ece(confidences: List[float], labels: List[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across n_bins."""
    if not confidences or not labels:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = [
            (c, l) for c, l in zip(confidences, labels)
            if bin_lower <= c < bin_upper or (i == n_bins - 1 and bin_lower <= c <= bin_upper)
        ]
        if in_bin:
            bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
            bin_acc = sum(l for _, l in in_bin) / len(in_bin)
            ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    return float(ece)

# %% Cell 9: Evaluation Runner Engine
def run_evaluation_suite(
    reranker: Any,
    reranker_name: str,
    device: str,
) -> Dict[str, Any]:
    """Runs the full evaluation benchmark across all 5 evaluation pillars."""
    logger.info(f"--- Starting evaluation for {reranker_name} on {device} ---")
    reset_cuda(device)
    start_time = time.perf_counter()

    # 1. Controlled IR Benchmark (Isolated Reranker Evaluation)
    ndcg3_list, ndcg5_list, ndcg10_list = [], [], []
    mrr5_list, mrr10_list = [], []
    recall10_list = []
    all_confidences, all_labels = [], []
    pos_scores, neg_scores = [], []

    query_latencies = []

    for case in ir_cases:
        query = case["query"]
        cand_ids = [case["positive_chunk_id"]] + case["same_doc_chunk_ids"] + case["other_doc_chunk_ids"]
        candidates = [chunk_by_id[cid] for cid in cand_ids if cid in chunk_by_id]

        t0 = time.perf_counter()
        ranked = reranker.rerank(query, candidates)
        t1 = time.perf_counter()
        query_latencies.append((t1 - t0) * 1000.0)

        ranked_ids = [r["chunk_id"] for r in ranked]
        graded = case["graded_relevance"]

        ndcg3_list.append(ndcg_at_k(ranked_ids, graded, 3))
        ndcg5_list.append(ndcg_at_k(ranked_ids, graded, 5))
        ndcg10_list.append(ndcg_at_k(ranked_ids, graded, 10))

        mrr5_list.append(mrr_at_k(ranked_ids, case["positive_chunk_id"], 5))
        mrr10_list.append(mrr_at_k(ranked_ids, case["positive_chunk_id"], 10))

        relevant_pool = [case["positive_chunk_id"]] + case["same_doc_chunk_ids"]
        recall10_list.append(recall_at_k(ranked_ids, relevant_pool, 10))

        # Confidence for calibration
        for r in ranked:
            score = r["rerank_score"]
            conf = sigmoid(score)
            is_relevant = 1 if graded.get(r["chunk_id"], 0) >= 1 else 0
            all_confidences.append(conf)
            all_labels.append(is_relevant)
            if graded.get(r["chunk_id"], 0) == 2:
                pos_scores.append(score)
            elif graded.get(r["chunk_id"], 0) == 0:
                neg_scores.append(score)

    mean_ndcg3 = float(np.mean(ndcg3_list))
    mean_ndcg5 = float(np.mean(ndcg5_list))
    mean_ndcg10 = float(np.mean(ndcg10_list))
    mean_mrr5 = float(np.mean(mrr5_list))
    mean_mrr10 = float(np.mean(mrr10_list))
    mean_recall10 = float(np.mean(recall10_list))

    # 2. Score Calibration & Separability
    ece = compute_ece(all_confidences, all_labels, n_bins=10)
    score_sep = float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores and neg_scores else 0.0

    # 3. Clinical Safety: Negation Sensitivity
    neg_success = 0
    for case in adv_neg_cases:
        query = case["query"]
        cands = [chunk_by_id[cid] for cid in [case["safe_chunk_id"], case["contraindication_chunk_id"]] if cid in chunk_by_id]
        ranked = reranker.rerank(query, cands)
        # Safe chunk must score higher than contraindication chunk
        if ranked and ranked[0]["chunk_id"] == case["safe_chunk_id"]:
            neg_success += 1
    neg_sensitivity_pct = (neg_success / len(adv_neg_cases) * 100.0) if adv_neg_cases else 0.0

    # 4. Clinical Safety: PICO Alignment
    pico_success = 0
    for case in adv_pico_cases:
        query = case["query"]
        cands = [chunk_by_id[cid] for cid in [case["correct_population_chunk_id"], case["wrong_population_chunk_id"]] if cid in chunk_by_id]
        ranked = reranker.rerank(query, cands)
        if ranked and ranked[0]["chunk_id"] == case["correct_population_chunk_id"]:
            pico_success += 1
    pico_alignment_pct = (pico_success / len(adv_pico_cases) * 100.0) if adv_pico_cases else 0.0

    # 5. Cross-lingual Parity
    delta_ndcg_list, delta_mrr_list = [], []
    for case in cross_lingual_cases:
        cands_en = [chunk_by_id[cid] for cid in case["candidate_chunk_ids_en"] if cid in chunk_by_id]
        cands_zh = [chunk_by_id[cid] for cid in case["candidate_chunk_ids_zh"] if cid in chunk_by_id]

        ranked_en = reranker.rerank(case["en_query"], cands_en)
        ranked_zh = reranker.rerank(case["zh_query"], cands_zh)

        # Compare top-1 score confidence difference
        if ranked_en and ranked_zh:
            s_en = sigmoid(ranked_en[0]["rerank_score"])
            s_zh = sigmoid(ranked_zh[0]["rerank_score"])
            delta_ndcg_list.append(abs(s_en - s_zh))

    delta_parity = float(np.mean(delta_ndcg_list)) if delta_ndcg_list else 0.0

    # 6. Performance & VRAM metrics
    total_time = time.perf_counter() - start_time
    peak_vram_gb = get_vram_usage(device)
    avg_latency_ms = float(np.mean(query_latencies)) if query_latencies else 0.0
    throughput_qps = len(query_latencies) / (sum(query_latencies) / 1000.0) if query_latencies else 0.0

    metrics = {
        "model": reranker_name,
        "device": device,
        "ndcg@3": mean_ndcg3,
        "ndcg@5": mean_ndcg5,
        "ndcg@10": mean_ndcg10,
        "mrr@5": mean_mrr5,
        "mrr@10": mean_mrr10,
        "recall@10": mean_recall10,
        "negation_sensitivity_pct": neg_sensitivity_pct,
        "pico_alignment_pct": pico_alignment_pct,
        "ece_10_bins": ece,
        "score_separability": score_sep,
        "cross_lingual_delta": delta_parity,
        "peak_vram_gb": peak_vram_gb,
        "avg_latency_ms": avg_latency_ms,
        "throughput_qps": throughput_qps,
        "total_eval_time_sec": total_time,
    }

    logger.info(f"Finished evaluation for {reranker_name}: NDCG@10={mean_ndcg10:.4f}, NegSens={neg_sensitivity_pct:.1f}%, PeakVRAM={peak_vram_gb:.2f}GB")
    return metrics

# %% Cell 10: Run BAAI/bge-reranker-v2-m3
from src.reranking.bge import CrossEncoderReranker

bge_model_name = "BAAI/bge-reranker-v2-m3"
logger.info(f"Initializing {bge_model_name} on {device_reranker_1}...")

bge_reranker = CrossEncoderReranker(
    model_name=bge_model_name,
    device=device_reranker_1,
    batch_size=32,
    instruction="Given a medical query and a clinical passage, evaluate relevance.",
)

# Run benchmark
bge_results = run_evaluation_suite(bge_reranker, "bge-reranker-v2-m3", device_reranker_1)

# Clean up memory
del bge_reranker
reset_cuda(device_reranker_1)

# %% Cell 11: Run Qwen/Qwen3-Reranker-0.6B
from src.reranking.qwen_reranker import QwenReranker

qwen_model_name = "Qwen/Qwen3-Reranker-0.6B"
logger.info(f"Initializing {qwen_model_name} on {device_reranker_2}...")

qwen_reranker = QwenReranker(
    model_name=qwen_model_name,
    device=device_reranker_2,
    batch_size=16,
    instruction="Given a medical query and a medical document passage, determine if the document is relevant to the query.",
)

# Run benchmark
qwen_results = run_evaluation_suite(qwen_reranker, "Qwen3-Reranker-0.6B", device_reranker_2)

# Clean up memory
del qwen_reranker
reset_cuda(device_reranker_2)

# %% Cell 12: Comparison Dashboard & Visualizations
comparison_data = [
    {
        "Evaluation Pillar": "1. Quantitative IR",
        "Metric": "NDCG@3",
        "bge-reranker-v2-m3": f"{bge_results['ndcg@3']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['ndcg@3']:.4f}",
        "Better": "BGE" if bge_results['ndcg@3'] > qwen_results['ndcg@3'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "1. Quantitative IR",
        "Metric": "NDCG@5",
        "bge-reranker-v2-m3": f"{bge_results['ndcg@5']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['ndcg@5']:.4f}",
        "Better": "BGE" if bge_results['ndcg@5'] > qwen_results['ndcg@5'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "1. Quantitative IR",
        "Metric": "NDCG@10",
        "bge-reranker-v2-m3": f"{bge_results['ndcg@10']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['ndcg@10']:.4f}",
        "Better": "BGE" if bge_results['ndcg@10'] > qwen_results['ndcg@10'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "1. Quantitative IR",
        "Metric": "MRR@5",
        "bge-reranker-v2-m3": f"{bge_results['mrr@5']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['mrr@5']:.4f}",
        "Better": "BGE" if bge_results['mrr@5'] > qwen_results['mrr@5'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "1. Quantitative IR",
        "Metric": "Recall@10",
        "bge-reranker-v2-m3": f"{bge_results['recall@10']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['recall@10']:.4f}",
        "Better": "BGE" if bge_results['recall@10'] > qwen_results['recall@10'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "2. Clinical Safety",
        "Metric": "Negation Sensitivity (%)",
        "bge-reranker-v2-m3": f"{bge_results['negation_sensitivity_pct']:.1f}%",
        "Qwen3-Reranker-0.6B": f"{qwen_results['negation_sensitivity_pct']:.1f}%",
        "Better": "BGE" if bge_results['negation_sensitivity_pct'] > qwen_results['negation_sensitivity_pct'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "2. Clinical Safety",
        "Metric": "PICO Alignment (%)",
        "bge-reranker-v2-m3": f"{bge_results['pico_alignment_pct']:.1f}%",
        "Qwen3-Reranker-0.6B": f"{qwen_results['pico_alignment_pct']:.1f}%",
        "Better": "BGE" if bge_results['pico_alignment_pct'] > qwen_results['pico_alignment_pct'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "3. Calibration",
        "Metric": "ECE (10 bins, lower=better)",
        "bge-reranker-v2-m3": f"{bge_results['ece_10_bins']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['ece_10_bins']:.4f}",
        "Better": "BGE" if bge_results['ece_10_bins'] < qwen_results['ece_10_bins'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "3. Calibration",
        "Metric": "Score Separability",
        "bge-reranker-v2-m3": f"{bge_results['score_separability']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['score_separability']:.4f}",
        "Better": "BGE" if bge_results['score_separability'] > qwen_results['score_separability'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "4. Cross-lingual Parity",
        "Metric": "Mean Delta (lower=better)",
        "bge-reranker-v2-m3": f"{bge_results['cross_lingual_delta']:.4f}",
        "Qwen3-Reranker-0.6B": f"{qwen_results['cross_lingual_delta']:.4f}",
        "Better": "BGE" if bge_results['cross_lingual_delta'] < qwen_results['cross_lingual_delta'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "5. Resource (2×T4)",
        "Metric": "Peak VRAM (GB)",
        "bge-reranker-v2-m3": f"{bge_results['peak_vram_gb']:.2f} GB",
        "Qwen3-Reranker-0.6B": f"{qwen_results['peak_vram_gb']:.2f} GB",
        "Better": "Qwen3" if qwen_results['peak_vram_gb'] < bge_results['peak_vram_gb'] else "BGE",
    },
    {
        "Evaluation Pillar": "5. Resource (2×T4)",
        "Metric": "Avg Latency (ms/query)",
        "bge-reranker-v2-m3": f"{bge_results['avg_latency_ms']:.1f} ms",
        "Qwen3-Reranker-0.6B": f"{qwen_results['avg_latency_ms']:.1f} ms",
        "Better": "BGE" if bge_results['avg_latency_ms'] < qwen_results['avg_latency_ms'] else "Qwen3",
    },
    {
        "Evaluation Pillar": "5. Resource (2×T4)",
        "Metric": "Throughput (queries/s)",
        "bge-reranker-v2-m3": f"{bge_results['throughput_qps']:.2f} qps",
        "Qwen3-Reranker-0.6B": f"{qwen_results['throughput_qps']:.2f} qps",
        "Better": "BGE" if bge_results['throughput_qps'] > qwen_results['throughput_qps'] else "Qwen3",
    },
]

df_report = pd.DataFrame(comparison_data)
print("\n" + "=" * 80)
print("           MEDICAL RERANKER EVALUATION DASHBOARD (Kaggle 2×T4)")
print("=" * 80)
print(df_report.to_string(index=False))
print("=" * 80)

# Save report
report_path = Path("reranker_comparison_report.csv")
df_report.to_csv(report_path, index=False)
logger.info(f"Saved benchmark report to {report_path.resolve()}")

# Radar Chart Visualization
labels = ["NDCG@10", "Negation Sens", "PICO Align", "1 - ECE", "Cross-lingual"]
bge_radar = [
    bge_results["ndcg@10"],
    bge_results["negation_sensitivity_pct"] / 100.0,
    bge_results["pico_alignment_pct"] / 100.0,
    max(0.0, 1.0 - bge_results["ece_10_bins"]),
    max(0.0, 1.0 - bge_results["cross_lingual_delta"]),
]
qwen_radar = [
    qwen_results["ndcg@10"],
    qwen_results["negation_sensitivity_pct"] / 100.0,
    qwen_results["pico_alignment_pct"] / 100.0,
    max(0.0, 1.0 - qwen_results["ece_10_bins"]),
    max(0.0, 1.0 - qwen_results["cross_lingual_delta"]),
]

num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
bge_radar += bge_radar[:1]
qwen_radar += qwen_radar[:1]

if plt is not None:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, bge_radar, color="#1f77b4", linewidth=2, label="BGE-reranker-v2-m3")
    ax.fill(angles, bge_radar, color="#1f77b4", alpha=0.25)
    ax.plot(angles, qwen_radar, color="#ff7f0e", linewidth=2, label="Qwen3-Reranker-0.6B")
    ax.fill(angles, qwen_radar, color="#ff7f0e", alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    plt.title("Medical Reranker 4-Pillar Clinical Capability Radar", size=14, y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    chart_path = Path("reranker_radar_comparison.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved radar chart to {chart_path.resolve()}")
    print(f"Evaluation finished successfully! Report: {report_path.name}, Radar: {chart_path.name}")
else:
    logger.info("matplotlib is not installed; skipping radar chart plot.")
    print(f"Evaluation finished successfully! Report: {report_path.name}")
