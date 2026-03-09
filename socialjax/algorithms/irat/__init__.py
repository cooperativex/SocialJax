"""IRAT: Individual Reward Assisted Multi-Agent Reinforcement Learning.

Paper: "Individual Reward Assisted Multi-Agent Reinforcement Learning"
       (ICML 2022) https://arxiv.org/abs/2202.03612

Usage:
    # Register the algorithm (auto-imported by train.py)
    import socialjax.algorithms.irat

    # Run full 1B-step training via JAX-scan optimized script:
    # CUDA_VISIBLE_DEVICES=1 python scripts/train_irat.py --env harvest_common_open
"""

from socialjax.algorithms.irat.algorithm import IRATAlgorithm, IRATAlgorithmState
from socialjax.algorithms.irat.network import IRATActorCNN, IRATCriticCNN
from socialjax.algorithms.irat.config import IRAT_DEFAULT_CONFIG, get_irat_config

__all__ = [
    "IRATAlgorithm",
    "IRATAlgorithmState",
    "IRATActorCNN",
    "IRATCriticCNN",
    "IRAT_DEFAULT_CONFIG",
    "get_irat_config",
]
