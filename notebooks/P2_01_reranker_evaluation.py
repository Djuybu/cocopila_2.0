# %% [markdown]
# # P2-01: Qwen3 reranking on the received P1-11 candidate bundle
# No corpus/index regeneration, new labels, or ID remapping. Existing controlled
# BGE/Qwen experiments are preserved in notebooks/legacy/.
# Install the package: python -m pip install -e ".[qwen,test]"
# Run from any cwd after editable installation; pass absolute paths if needed:
# python notebooks/P2_01_reranker_evaluation.py --run-dir outputs/p1_p2_handoff \
#   --config configs/reranker/qwen_reranker.yaml --output-dir outputs/p2_qwen3_001

# %%
from src.pipeline.cli import main

if __name__ == "__main__":
    main("benchmark_reranking")
