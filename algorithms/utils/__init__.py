"""Shared utilities for all algorithms."""

from algorithms.utils.networks import (
    CNN,
    ActorCritic,
    Actor,
    Critic,
    SmallCNN,
    SmallActor,
    SmallCritic
)

from algorithms.utils.data_utils import (
    batchify,
    batchify_dict,
    batchify_numpy,
    unbatchify
)

from algorithms.utils.vdn_networks import (
    QNetwork
)

from algorithms.utils.io_utils import (
    save_params,
    load_params
)

from algorithms.utils.eval_utils import (
    evaluate_ippo,
    evaluate_mappo_style
)

from algorithms.utils.types import (
    Transition,
    MAPPOTransition,
    IRATTransition,
)

__all__ = [
    # Network architectures
    "CNN",
    "ActorCritic",
    "Actor",
    "Critic",
    "SmallCNN",
    "SmallActor",
    "SmallCritic",
    # VDN-specific networks
    "QNetwork",
    # Data manipulation utilities
    "batchify",
    "batchify_dict",
    "batchify_numpy",
    "unbatchify",
    # IO utilities
    "save_params",
    "load_params",
    # Evaluation utilities
    "evaluate_ippo",
    "evaluate_mappo_style",
    # Shared types
    "Transition",
    "MAPPOTransition",
    "IRATTransition",
]
