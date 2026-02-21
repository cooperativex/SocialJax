"""
Counterfactual Regret Algorithm for Multi-Agent Reinforcement Learning

This module implements the Counterfactual Regret algorithm from the paper
"Resolving Complex Social Dilemmas by Aligning Preferences with Counterfactual Regret"
(ICML 2026).

Key Components:
- Generative Model (Φ_m): Predicts rewards from joint observations and actions
- Counterfactual Reward: Rewards for alternative actions
- Counterfactual Regret: Measures suboptimality of current action
- Intrinsic Reward: Negative regret to encourage prosocial behavior
- Shaped Reward: Extrinsic + α * Intrinsic

Modules:
    generative_model: RewardModel implementation (M1)
    counterfactual: Counterfactual reward generation (M2, M3) - TODO
    regret: Counterfactual regret calculation (M4) - TODO
    intrinsic_reward: Intrinsic reward construction (M5) - TODO
    reward_shaping: Shaped reward combination (M6) - TODO
    policy: ActorCritic and PPO (M7) - TODO
    causal_attention: Optional CausalRewardModel (M8) - TODO
    cf_trainer: Training loop (M9) - TODO
    env_adapters: Environment adapters (M10) - TODO

Reference:
    Counterfactual/cf_method
"""

# M1: Generative Model (implemented)
from socialjax.algorithms.cf.generative_model import (
    RewardModel,
    CNNFeatureExtractor,
    generative_model_loss,
    compute_generative_model_loss,
    create_reward_model_train_state,
)

# M2, M3: Counterfactual (implemented)
from socialjax.algorithms.cf.counterfactual import (
    enumerate_counterfactual_actions,
    enumerate_all_agents_counterfactual_actions,
    generate_counterfactual_rewards,
    generate_counterfactual_rewards_vmap,
    generate_counterfactual_rewards_single_agent,
    compute_collective_cf_reward,
    compute_actual_collective_reward,
    get_counterfactual_analysis,
)

# M4: Regret (implemented)
from socialjax.algorithms.cf.regret import (
    compute_counterfactual_regret,
    compute_regret_with_best_action,
    compute_normalized_regret,
    get_regret_statistics,
)

# M5-M10: Other modules (to be implemented)
# from socialjax.algorithms.cf.intrinsic_reward import (
#     compute_intrinsic_reward,
# )
# from socialjax.algorithms.cf.reward_shaping import (
#     compute_shaped_reward,
# )

__all__ = [
    # M1: Generative Model
    "RewardModel",
    "CNNFeatureExtractor",
    "generative_model_loss",
    "compute_generative_model_loss",
    "create_reward_model_train_state",
    # M2, M3: Counterfactual
    "enumerate_counterfactual_actions",
    "enumerate_all_agents_counterfactual_actions",
    "generate_counterfactual_rewards",
    "generate_counterfactual_rewards_vmap",
    "generate_counterfactual_rewards_single_agent",
    "compute_collective_cf_reward",
    "compute_actual_collective_reward",
    "get_counterfactual_analysis",
    # M4: Regret
    "compute_counterfactual_regret",
    "compute_regret_with_best_action",
    "compute_normalized_regret",
    "get_regret_statistics",
    # M5: Intrinsic Reward (TODO)
    # "compute_intrinsic_reward",
    # M6: Reward Shaping (TODO)
    # "compute_shaped_reward",
]
