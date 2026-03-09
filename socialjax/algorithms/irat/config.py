"""Configuration for IRAT algorithm.

IRAT: Individual Reward Assisted Multi-Agent Reinforcement Learning
Paper: https://arxiv.org/abs/2202.03612 (ICML 2022)
"""

from typing import Dict, Any

IRAT_DEFAULT_CONFIG: Dict[str, Any] = {
    "LR": 5e-4,
    "ANNEAL_LR": True,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "SCALE_CLIP_EPS": True,       # divide clip_eps by num_agents
    "VF_COEF": 0.5,
    "ENT_COEF": 0.01,
    "MAX_GRAD_NORM": 0.5,
    "UPDATE_EPOCHS": 2,
    "NUM_MINIBATCHES": 4,
    "NUM_STEPS": 1000,
    "NUM_ENVS": 256,
    "ACTIVATION": "relu",
}


def get_irat_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    config = IRAT_DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
