"""Factory for initializing medical reranker models from configuration."""
from typing import Any, Dict
from src.reranking.base import BaseReranker
from src.reranking.bge import CrossEncoderReranker
from src.reranking.qwen_reranker import QwenReranker


def create_reranker(config: Dict[str, Any]) -> BaseReranker:
    """Create a reranker instance from configuration dict.

    Supported types:
    - 'bge' or 'cross_encoder': BAAI/bge-reranker-v2-m3 and compatible cross-encoders
    - 'qwen': Qwen/Qwen3-Reranker-0.6B with medical instruction support

    Args:
        config: Dictionary containing reranker configurations.
            Required/optional keys:
            - 'type': str ('bge', 'qwen', 'cross_encoder')
            - 'model' or 'model_name': str
            - 'device': str ('cuda', 'cuda:0', 'cuda:1', 'cpu')
            - 'batch_size': int
            - 'instruction': Optional[str]

    Returns:
        BaseReranker: Concrete reranker instance.

    Raises:
        ValueError: If reranker type is unsupported.
    """
    cfg = config.get("reranker", config)
    reranker_type = cfg.get("type", "bge").lower()
    model_name = cfg.get("model") or cfg.get("model_name")
    device = cfg.get("device")
    batch_size = cfg.get("batch_size")
    instruction = cfg.get("instruction")

    kwargs: Dict[str, Any] = {}
    if model_name is not None:
        kwargs["model_name"] = model_name
    if device is not None:
        kwargs["device"] = device
    if batch_size is not None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        kwargs["batch_size"] = batch_size
    if instruction is not None:
        kwargs["instruction"] = instruction

    if reranker_type in ("bge", "cross_encoder"):
        return CrossEncoderReranker(**kwargs)
    elif reranker_type == "qwen":
        for key in ("max_length", "dtype", "revision"):
            if cfg.get(key) is not None:
                kwargs[key] = cfg[key]
        return QwenReranker(**kwargs)
    else:
        raise ValueError(
            f"Unsupported reranker type: '{reranker_type}'. Supported types: 'bge', 'qwen', 'cross_encoder'."
        )
