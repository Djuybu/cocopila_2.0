"""Per-run logging without configuring logging during import."""
import json
import logging


def run_logger(run_dir, config):
    logger = logging.getLogger(f"medical_rag.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.info("run_id=%s config=%s", run_dir.name, json.dumps(config, ensure_ascii=False))
    return logger
