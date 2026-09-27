"""Build test cases and embeddings for Medical Reranker evaluation.

Reads documents strictly from data/test/{chn,eng}, chunks them, generates
dual embeddings (MiniLM-L12-v2 and BGE-M3), and constructs structured test cases:
1. Standard IR test cases (50 cases: 25 Chinese, 25 English)
2. Adversarial Negation / Contraindication cases (10 cases)
3. Adversarial PICO Population mismatch cases (10 cases)
4. Cross-lingual parallel query pairs (10 cases)
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Predefined cross-lingual medical concept pairs
CROSS_LINGUAL_PAIRS: List[Tuple[str, str, str]] = [
    ("cl_001", "diabetes treatment options", "糖尿病治疗方案"),
    ("cl_002", "hypertension medication side effects", "高血压药物副作用"),
    ("cl_003", "surgical wound care", "手术伤口护理"),
    ("cl_004", "antibiotic resistance", "抗生素耐药性"),
    ("cl_005", "cancer screening guidelines", "癌症筛查指南"),
    ("cl_006", "pain management strategies", "疼痛管理策略"),
    ("cl_007", "heart disease risk factors", "心脏病风险因素"),
    ("cl_008", "bone fracture treatment", "骨折治疗方法"),
    ("cl_009", "eye disease diagnosis", "眼科疾病诊断"),
    ("cl_010", "skin disease symptoms", "皮肤病症状表现"),
]

# Clinical keywords for adversarial mining
NEGATION_PATTERNS = {
    "chn": [r"禁忌", r"禁用", r"不宜", r"不能", r"不可", r"慎用"],
    "eng": [r"\bcontraindicated\b", r"\bshould not\b", r"\bmust not\b", r"\bdo not use\b", r"\bavoid\b", r"\bprohibited\b"],
}

POPULATION_PATTERNS = {
    "chn": {
        "pediatric": [r"儿童", r"小儿", r"婴幼儿", r"幼儿"],
        "adult": [r"成人", r"成年人"],
        "elderly": [r"老年", r"高龄"],
        "pregnancy": [r"孕妇", r"妊娠期", r"哺乳期"],
    },
    "eng": {
        "pediatric": [r"\bpediatric\b", r"\bchildren\b", r"\binfant\b", r"\bneonatal\b"],
        "adult": [r"\badult\b", r"\badults\b"],
        "elderly": [r"\belderly\b", r"\bgeriatric\b", r"\bsenior\b"],
        "pregnancy": [r"\bpregnant\b", r"\bpregnancy\b", r"\blactating\b"],
    },
}


def load_documents_strictly_from_test_dir(data_dir: Path) -> List[Dict[str, Any]]:
    """Strictly loads text documents only from data_dir/chn and data_dir/eng.

    Does not touch any other directories under data/.
    """
    documents = []
    for lang in ["chn", "eng"]:
        lang_dir = data_dir / lang
        if not lang_dir.is_dir():
            logger.warning(f"Directory {lang_dir} does not exist, skipping.")
            continue

        for file_path in sorted(lang_dir.glob("*.txt")):
            doc_id = file_path.stem
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    documents.append({
                        "doc_id": doc_id,
                        "text": text,
                        "language": lang,
                        "file_path": str(file_path),
                    })
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Loaded {len(documents)} documents (chn + eng).")
    return documents


def chunk_document(
    doc: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[Dict[str, Any]]:
    """Chunks a document using sliding window of chunk_size characters with overlap."""
    text = doc["text"]
    doc_id = doc["doc_id"]
    lang = doc["language"]
    n_chars = len(text)
    chunks = []
    start = 0
    chunk_idx = 0
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size // 2

    while start < n_chars:
        end = min(start + chunk_size, n_chars)
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= 40:  # Skip tiny fragments
            chunk_id = f"{doc_id}_chunk_{chunk_idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text,
                "language": lang,
                "char_start": start,
                "char_end": end,
            })
            chunk_idx += 1
        if end >= n_chars:
            break
        start += step

    return chunks


def build_chunks_corpus(
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[Dict[str, Any]]:
    """Chunks all documents into a single chunk collection."""
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
    logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents.")
    return all_chunks


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Generates dense embeddings for all chunk texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    texts = [c["text"] for c in chunks]
    logger.info(f"Encoding {len(texts)} chunks with {model_name} (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def extract_query_from_chunk(text: str, language: str, rng: random.Random) -> str:
    """Extracts a concise, realistic query (10-15 words or chars) from text."""
    if language == "eng":
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 6]
        if sentences:
            sentence = rng.choice(sentences)
            words = sentence.split()
            if len(words) > 15:
                start_w = rng.randint(0, max(0, len(words) - 14))
                query = " ".join(words[start_w : start_w + rng.randint(8, 14)])
            else:
                query = " ".join(words)
            # Remove trailing punctuation
            return re.sub(r"[^\w\s]+$", "", query).strip()
        else:
            words = text.split()
            return " ".join(words[: min(12, len(words))])
    else:
        # Chinese: split by Chinese punctuation marks
        sentences = [s.strip() for s in re.split(r"[。！？；\n]+", text) if len(s.strip()) >= 8]
        if sentences:
            sentence = rng.choice(sentences)
            if len(sentence) > 18:
                start_c = rng.randint(0, len(sentence) - 15)
                query = sentence[start_c : start_c + rng.randint(10, 16)]
            else:
                query = sentence
            return query.strip()
        else:
            return text[: min(15, len(text))]


def build_ir_standard_test_cases(
    chunks: List[Dict[str, Any]],
    n_cases: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Builds standard IR test cases (25 Chinese, 25 English).

    Each case:
    - 1 positive chunk (source chunk)
    - 3 same-doc chunks (partially relevant / context)
    - 3 other-doc chunks (hard negatives from different docs in same language)
    """
    rng = random.Random(seed)
    chunks_by_lang: Dict[str, List[Dict[str, Any]]] = {"chn": [], "eng": []}
    doc_to_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for c in chunks:
        chunks_by_lang[c["language"]].append(c)
        doc_to_chunks.setdefault(c["doc_id"], []).append(c)

    test_cases = []
    cases_per_lang = n_cases // 2

    for lang in ["chn", "eng"]:
        lang_chunks = chunks_by_lang[lang]
        # Only use documents that have at least 4 chunks so we can pick 1 pos + 3 same-doc
        eligible_docs = [
            doc_id for doc_id, c_list in doc_to_chunks.items()
            if len(c_list) >= 4 and c_list[0]["language"] == lang
        ]

        if not eligible_docs:
            logger.warning(f"Not enough eligible documents for language {lang}.")
            continue

        selected_cases = 0
        shuffled_docs = eligible_docs.copy()
        rng.shuffle(shuffled_docs)
        doc_pointer = 0

        while selected_cases < cases_per_lang:
            doc_id = shuffled_docs[doc_pointer % len(shuffled_docs)]
            doc_pointer += 1
            available_chunks = doc_to_chunks[doc_id]

            pos_chunk = rng.choice(available_chunks)
            query = extract_query_from_chunk(pos_chunk["text"], lang, rng)
            if not query or len(query) < 5:
                continue

            same_doc_pool = [c for c in available_chunks if c["chunk_id"] != pos_chunk["chunk_id"]]
            same_doc_sample = rng.sample(same_doc_pool, min(3, len(same_doc_pool)))

            # Pick 3 chunks from 3 different documents
            other_docs = [d for d in eligible_docs if d != doc_id]
            if len(other_docs) >= 3:
                chosen_other_docs = rng.sample(other_docs, 3)
                other_doc_sample = [rng.choice(doc_to_chunks[od]) for od in chosen_other_docs]
            else:
                other_pool = [c for c in lang_chunks if c["doc_id"] != doc_id]
                other_doc_sample = rng.sample(other_pool, min(3, len(other_pool)))

            case_id = f"ir_{lang}_{selected_cases + 1:03d}"
            test_cases.append({
                "id": case_id,
                "category": "ir_standard",
                "language": lang,
                "query": query,
                "positive_chunk_id": pos_chunk["chunk_id"],
                "same_doc_chunk_ids": [c["chunk_id"] for c in same_doc_sample],
                "other_doc_chunk_ids": [c["chunk_id"] for c in other_doc_sample],
                "graded_relevance": {
                    pos_chunk["chunk_id"]: 2,
                    **{c["chunk_id"]: 1 for c in same_doc_sample},
                    **{c["chunk_id"]: 0 for c in other_doc_sample},
                },
            })
            selected_cases += 1

    logger.info(f"Built {len(test_cases)} IR standard test cases.")
    return test_cases


def build_adversarial_negation_test_cases(
    chunks: List[Dict[str, Any]],
    n_cases: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Builds clinical safety adversarial cases involving contraindications / negations.

    Finds chunks containing contraindication keywords and pairs them with safe usage chunks.
    """
    rng = random.Random(seed)
    contra_chunks: List[Dict[str, Any]] = []
    safe_chunks: List[Dict[str, Any]] = []

    for c in chunks:
        lang = c["language"]
        patterns = NEGATION_PATTERNS.get(lang, [])
        has_contra = any(re.search(pat, c["text"], re.IGNORECASE) for pat in patterns)
        if has_contra:
            contra_chunks.append(c)
        else:
            safe_chunks.append(c)

    test_cases = []
    rng.shuffle(contra_chunks)

    for i, c_contra in enumerate(contra_chunks):
        if len(test_cases) >= n_cases:
            break
        lang = c_contra["language"]
        matching_safe = [c for c in safe_chunks if c["language"] == lang and c["doc_id"] == c_contra["doc_id"]]
        if not matching_safe:
            matching_safe = [c for c in safe_chunks if c["language"] == lang]
        if not matching_safe:
            continue

        c_safe = rng.choice(matching_safe)
        query = extract_query_from_chunk(c_contra["text"], lang, rng)

        test_cases.append({
            "id": f"adv_neg_{len(test_cases) + 1:03d}",
            "category": "adversarial_negation",
            "language": lang,
            "query": query,
            "contraindication_chunk_id": c_contra["chunk_id"],
            "safe_chunk_id": c_safe["chunk_id"],
            # Ground truth: Safe treatment chunk should rank higher than contraindicated chunk for treatment query
            "expected_top_chunk_id": c_safe["chunk_id"],
            "notes": "Contraindication chunk should be demoted or scored lower to prevent clinical harm.",
        })

    logger.info(f"Built {len(test_cases)} adversarial negation test cases.")
    return test_cases


def build_adversarial_pico_test_cases(
    chunks: List[Dict[str, Any]],
    n_cases: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Builds adversarial PICO test cases targeting population mismatch.

    Pairs queries specifically targeting a clinical population (e.g. pediatric) with
    chunks describing other populations (e.g. adult/elderly).
    """
    rng = random.Random(seed)
    test_cases = []

    for lang in ["chn", "eng"]:
        patterns_by_pop = POPULATION_PATTERNS[lang]
        pop_chunks: Dict[str, List[Dict[str, Any]]] = {pop: [] for pop in patterns_by_pop}

        for c in chunks:
            if c["language"] != lang:
                continue
            for pop, patterns in patterns_by_pop.items():
                if any(re.search(pat, c["text"], re.IGNORECASE) for pat in patterns):
                    pop_chunks[pop].append(c)

        # Pair pediatric vs adult/elderly
        ped_list = pop_chunks.get("pediatric", [])
        adult_list = pop_chunks.get("adult", []) + pop_chunks.get("elderly", [])

        if ped_list and adult_list:
            n_target = n_cases // 2
            for _ in range(n_target):
                if len(test_cases) >= n_cases:
                    break
                ped_chunk = rng.choice(ped_list)
                adult_chunk = rng.choice(adult_list)

                if lang == "chn":
                    query = "儿童 " + extract_query_from_chunk(ped_chunk["text"], "chn", rng)
                else:
                    query = "pediatric " + extract_query_from_chunk(ped_chunk["text"], "eng", rng)

                test_cases.append({
                    "id": f"adv_pico_{len(test_cases) + 1:03d}",
                    "category": "adversarial_pico",
                    "language": lang,
                    "target_population": "pediatric",
                    "query": query,
                    "correct_population_chunk_id": ped_chunk["chunk_id"],
                    "wrong_population_chunk_id": adult_chunk["chunk_id"],
                    "expected_top_chunk_id": ped_chunk["chunk_id"],
                    "notes": "Query explicitly requests pediatric indications; adult chunk must rank lower.",
                })

    logger.info(f"Built {len(test_cases)} adversarial PICO test cases.")
    return test_cases


# Topic keywords for cross-lingual candidate retrieval
TOPIC_KEYWORDS: Dict[str, Tuple[List[str], List[str]]] = {
    "cl_001": (["diabet", "insulin", "glucose", "glycemic"], ["糖尿病", "血糖", "胰岛素"]),
    "cl_002": (["hypertens", "blood pressure", "antihypertens"], ["高血压", "血压", "降压"]),
    "cl_003": (["wound", "surgical", "incision", "suture", "dress"], ["伤口", "手术", "缝合", "敷料"]),
    "cl_004": (["antibiotic", "antimicrobial", "resist", "bacteri"], ["抗生素", "耐药", "细菌"]),
    "cl_005": (["cancer", "tumor", "oncolog", "screen", "malignan"], ["癌症", "肿瘤", "筛查", "恶性"]),
    "cl_006": (["pain", "analges", "opioid", "relie"], ["疼痛", "镇痛", "止痛"]),
    "cl_007": (["heart", "cardiac", "cardiovascular", "coronary"], ["心脏", "心血管", "冠心"]),
    "cl_008": (["fracture", "bone", "orthopedic", "trauma"], ["骨折", "骨骼", "创伤", "骨科"]),
    "cl_009": (["eye", "ocular", "ophthalm", "retin", "vision"], ["眼科", "眼睛", "视网膜", "视力"]),
    "cl_010": (["skin", "dermatol", "lesion", "rash", "prurit"], ["皮肤", "皮疹", "皮炎", "皮损"]),
}


def build_cross_lingual_test_cases(
    chunks: List[Dict[str, Any]],
    embeddings: Optional[np.ndarray] = None,
    embedding_model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Builds cross-lingual test cases from predefined medical concept pairs."""
    test_cases = []

    chn_chunks = [c for c in chunks if c["language"] == "chn"]
    eng_chunks = [c for c in chunks if c["language"] == "eng"]

    for case_id, en_query, zh_query in CROSS_LINGUAL_PAIRS:
        en_kw, zh_kw = TOPIC_KEYWORDS.get(case_id, ([], []))

        # Rank chunks by keyword occurrences
        def score_en(c):
            t = c["text"].lower()
            return sum(t.count(k) for k in en_kw)

        def score_zh(c):
            t = c["text"]
            return sum(t.count(k) for k in zh_kw)

        matched_en = sorted(eng_chunks, key=score_en, reverse=True)
        matched_zh = sorted(chn_chunks, key=score_zh, reverse=True)

        en_cand = [c["chunk_id"] for c in matched_en[:10]]
        zh_cand = [c["chunk_id"] for c in matched_zh[:10]]

        test_cases.append({
            "id": case_id,
            "category": "cross_lingual",
            "en_query": en_query,
            "zh_query": zh_query,
            "candidate_chunk_ids_en": en_cand,
            "candidate_chunk_ids_zh": zh_cand,
            "notes": "Evaluate score consistency and rank parity between English and Chinese queries.",
        })

    logger.info(f"Built {len(test_cases)} cross-lingual test cases.")
    return test_cases


def main():
    parser = argparse.ArgumentParser(description="Build test cases and embeddings for Medical Reranker evaluation")
    parser.add_argument("--data-dir", type=str, default="data/test", help="Path to data/test containing chn and eng")
    parser.add_argument("--output-dir", type=str, default="data/test/reranker_eval", help="Output directory")
    parser.add_argument("--embedding-models", nargs="+", default=["minilm", "bgem3"],
                        help="Embedding models to generate: minilm, bgem3, or both")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk window size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=128, help="Chunk overlap in characters")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--device", type=str, default=None, help="Device for embedding (e.g. cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation for fast test")
    parser.add_argument("--dummy-embeddings", action="store_true", help="Generate unit-normalized random embeddings for local testing")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step 1: Reading documents strictly from {data_dir}...")
    documents = load_documents_strictly_from_test_dir(data_dir)
    if not documents:
        raise ValueError(f"No documents found in {data_dir}/chn or {data_dir}/eng.")

    logger.info(f"Step 2: Chunking documents (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = build_chunks_corpus(documents, args.chunk_size, args.chunk_overlap)

    # Save chunks.json
    chunks_path = output_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")

    # Save chunks_meta.json
    chunks_meta = {
        c["chunk_id"]: {
            "doc_id": c["doc_id"],
            "language": c["language"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
        }
        for c in chunks
    }
    chunks_meta_path = output_dir / "chunks_meta.json"
    with open(chunks_meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved chunks metadata mapping to {chunks_meta_path}")

    # Step 3: Embeddings
    model_map = {
        "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "embeddings_minilm.npy", 384),
        "bgem3": ("BAAI/bge-m3", "embeddings_bgem3.npy", 1024),
    }

    if args.dummy_embeddings:
        logger.info("Step 3: Generating dummy unit-normalized embeddings for fast local verification...")
        rng = np.random.default_rng(args.seed)
        for model_key, (_, npy_filename, dim) in model_map.items():
            raw = rng.standard_normal((len(chunks), dim), dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            dummy_emb = raw / np.maximum(norms, 1e-9)
            emb_path = output_dir / npy_filename
            np.save(emb_path, dummy_emb)
            logger.info(f"Saved dummy {model_key} embeddings (shape: {dummy_emb.shape}) to {emb_path}")
    elif not args.skip_embeddings:
        logger.info("Step 3: Generating dense chunk embeddings...")
        models_to_run = args.embedding_models
        if "both" in models_to_run:
            models_to_run = ["minilm", "bgem3"]

        for model_key in models_to_run:
            if model_key not in model_map:
                logger.warning(f"Unknown embedding model key: {model_key}, skipping.")
                continue
            hf_model_name, npy_filename, _ = model_map[model_key]
            embeddings = generate_embeddings(
                chunks,
                hf_model_name,
                batch_size=args.batch_size,
                device=args.device,
            )
            emb_path = output_dir / npy_filename
            np.save(emb_path, embeddings)
            logger.info(f"Saved {model_key} embeddings (shape: {embeddings.shape}) to {emb_path}")
    else:
        logger.info("Skipping embedding generation (--skip-embeddings set).")

    # Step 4: Test Cases
    logger.info("Step 4: Building test cases...")
    ir_cases = build_ir_standard_test_cases(chunks, n_cases=50, seed=args.seed)
    adv_neg_cases = build_adversarial_negation_test_cases(chunks, n_cases=10, seed=args.seed)
    adv_pico_cases = build_adversarial_pico_test_cases(chunks, n_cases=10, seed=args.seed)
    cross_lingual_cases = build_cross_lingual_test_cases(chunks)

    test_cases_data = {
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "total_cases": len(ir_cases) + len(adv_neg_cases) + len(adv_pico_cases) + len(cross_lingual_cases),
        "corpus_stats": {
            "total_chunks": len(chunks),
            "chn_chunks": sum(1 for c in chunks if c["language"] == "chn"),
            "eng_chunks": sum(1 for c in chunks if c["language"] == "eng"),
            "total_docs": len(documents),
        },
        "categories": {
            "ir_standard": ir_cases,
            "adversarial_negation": adv_neg_cases,
            "adversarial_pico": adv_pico_cases,
            "cross_lingual": cross_lingual_cases,
        },
    }

    test_cases_path = output_dir / "test_cases.json"
    with open(test_cases_path, "w", encoding="utf-8") as f:
        json.dump(test_cases_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {test_cases_data['total_cases']} test cases to {test_cases_path}")

    logger.info("Test case generation completed successfully!")


if __name__ == "__main__":
    main()
