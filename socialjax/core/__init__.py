"""SocialJax core module.

This module provides the base classes and utilities for the SocialJax
multi-agent reinforcement learning framework.
"""

from socialjax.core.base_algorithm import (
    AlgorithmState,
    BaseAlgorithm,
    jit_method,
)

from socialjax.core.base_trainer import (
    Callback,
    TrainerState,
    TrainingMetrics,
    BaseTrainer,
)

__all__ = [
    # Algorithm
    "AlgorithmState",
    "BaseAlgorithm",
    "jit_method",
    # Trainer
    "Callback",
    "TrainerState",
    "TrainingMetrics",
    "BaseTrainer",
]
