"""CLI entrypoint for P2-08: Cross-encoder reranker fine-tuning.

Fine-tunes cross-encoder (e.g. BAAI/bge-reranker-v2-m3) on positive + hard negatives,
records config, seed, and metrics, and produces Reranker Checkpoint v1.
"""
import argparse
import logging
from pathlib import Path
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_records
from src.training.finetune import (
    CrossEncoderTrainer,
    create_training_config,
    _validation_queries,
)
from src.utils.io import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_reranker")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P2-08: Fine-tune Cross-Encoder reranker on positive and hard negative pairs"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Pretrained model identifier or path",
    )
    parser.add_argument(
        "--val-queries", type=Path, help="Validation queries.json (defaults to the candidate directory)",
    )
    parser.add_argument(
        "--val-registry", type=Path, help="Validation registry.json for chunk ID mapping",
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=None,
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Path to labeled training pairs JSONL (positive + hard negatives)",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Path to validation candidates JSONL for evaluation",
    )
    parser.add_argument(
        "--val-labels",
        type=Path,
        default=None,
        help="Path to validation labels.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output checkpoint directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and data loading without running PyTorch training loop",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration from YAML if supplied
    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("training", {})

    model_name = args.model_name or cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
    output_dir = Path(args.output_dir or cfg.get("output_dir", "checkpoints/reranker_v1"))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 3))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 16))
    lr = float(args.lr if args.lr is not None else cfg.get("learning_rate", 2e-5))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else float(cfg.get("warmup_ratio", 0.1))

    train_data_path = args.train_data or (Path(cfg["train_data"]) if "train_data" in cfg else None)
    val_data_path = args.val_data or (Path(cfg["val_data"]) if cfg.get("val_data") else None)
    val_labels_path = args.val_labels or (Path(cfg["val_labels"]) if cfg.get("val_labels") else None)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    if not train_data_path or not train_data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {train_data_path}")

    logger.info("Loading training pairs from %s...", train_data_path)
    train_pairs = load_records(train_data_path)
    if not train_pairs:
        raise ValueError("Cannot train on empty pairs")
    if epochs < 1 or batch_size < 1 or lr <= 0 or not 0 <= warmup_ratio <= 1:
        raise ValueError("Invalid training hyperparameters")
    logger.info("Loaded %d training pairs.", len(train_pairs))

    val_records = None
    val_labels = None
    val_queries = None
    registry = {}
    if val_labels_path and not val_data_path:
        raise ValueError("--val-labels requires --val-data")
    if val_data_path:
        val_labels_path = val_labels_path or val_data_path.parent / "labels.json"
        logger.info("Loading validation records from %s...", val_data_path)
        val_records = load_records(val_data_path)
        val_labels = read_json(val_labels_path)
        queries_path = args.val_queries or (Path(cfg["val_queries"]) if cfg.get("val_queries") else val_data_path.parent / "queries.json")
        if queries_path.exists() or args.val_queries or cfg.get("val_queries"):
            val_queries = read_json(queries_path)
        registry_path = args.val_registry or (Path(cfg["val_registry"]) if cfg.get("val_registry") else val_data_path.parent / "registry.json")
        if registry_path.exists() or args.val_registry or cfg.get("val_registry"):
            registry = read_json(registry_path)
        _validation_queries(val_records, val_labels, val_queries)
        from src.training.data_generator import validate_no_leakage
        validation_pairs = [{"query_id": r["id"], "doc_id": c.get("doc_id")} for r in val_records for c in r.get("candidates", [])]
        validation_pairs.extend({"query_id": r["id"]} for r in val_records)
        validate_no_leakage(train_pairs, validation_pairs)

    trainer = CrossEncoderTrainer(
        model_name=model_name,
        output_dir=output_dir,
        seed=seed,
    )

    if args.dry_run:
        logger.info("Dry-run requested: skipping model.fit(). Verifying checkpoint serialization...")
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "model_name": model_name,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_training_pairs": len(train_pairs),
            "output_dir": str(output_dir),
            "dry_run": True,
            "validation_metrics": None,
        }
        write_json(output_dir / "training_manifest.json", manifest)
    else:
        logger.info("Starting CrossEncoder fine-tuning...")
        manifest = trainer.train(
            train_pairs=train_pairs,
            val_records=val_records,
            val_labels=val_labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            warmup_ratio=warmup_ratio,
            val_queries=val_queries,
            internal_to_official=registry.get("internal_to_official"),
            chunk_to_doc=registry.get("chunk_to_doc"),
        )
    training_config = create_training_config(
        model_name, output_dir, epochs, batch_size, lr, warmup_ratio=warmup_ratio,
        seed=seed, train_data=train_data_path, val_data=val_data_path,
    )
    training_config["training"].update(val_labels=str(val_labels_path) if val_labels_path else None)
    with (output_dir / "training_config.yaml").open("x", encoding="utf-8") as f:
        yaml.safe_dump(training_config, f)

    print("\n" + "=" * 80)
    print("P2-08: RERANKER FINE-TUNING EXECUTION REPORT")
    print("=" * 80)
    print(f"Base Model        : {model_name}")
    print(f"Training Samples  : {len(train_pairs)} pairs")
    print(f"Hyperparameters   : Epochs={epochs}, BatchSize={batch_size}, LR={lr}, Seed={seed}")
    print(f"Checkpoint Target : {output_dir}")
    if manifest.get("validation_metrics"):
        vm = manifest["validation_metrics"]
        print(f"Validation F2     : {vm.get('macro_f2')}")
        print(f"Validation Recall : {vm.get('macro_recall')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
