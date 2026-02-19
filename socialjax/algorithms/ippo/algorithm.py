"""IPPO (Independent Proximal Policy Optimization) Algorithm Implementation.

This module implements the IPPO algorithm for multi-agent reinforcement learning.
IPPO trains each agent independently using PPO, without explicit coordination
during training.
"""

from typing import Any, Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct

from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
from socialjax.algorithms.registry import register_algorithm
from socialjax.algorithms.ippo.config import IPPO_DEFAULT_CONFIG, get_ippo_config
from socialjax.algorithms.ippo.network import IPPOActorCritic


class Transition(NamedTuple):
    """Container for a single environment transition."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict[str, Any]


class IPPOAlgorithmState(struct.PyTreeNode):
    """State container for IPPO algorithm.

    Attributes:
        params: Network parameters
        optimizer_state: Optimizer state
        rng: Random key
        timestep: Current training timestep
        update_step: Number of parameter updates performed
    """
    params: Dict[str, Any]
    optimizer_state: Any
    rng: jax.random.PRNGKey
    timestep: int = 0
    update_step: int = 0


@register_algorithm("ippo")
class IPPOAlgorithm(BaseAlgorithm):
    """Independent Proximal Policy Optimization (IPPO) for multi-agent RL.

    IPPO trains each agent independently using PPO. All agents share the same
    network parameters (parameter sharing), which enables efficient learning
    and generalization across agents.

    This implementation follows the standard PPO algorithm with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus

    Attributes:
        observation_space: Environment observation space
        action_space: Environment action space
        config: Algorithm configuration dictionary
        network: IPPOActorCritic network
        optimizer: Optax optimizer
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the IPPO algorithm.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            config: Algorithm configuration (uses defaults if None)
        """
        merged_config = get_ippo_config(config)
        super().__init__(observation_space, action_space, merged_config)

    def _build_network(self) -> Any:
        """Build and return the Actor-Critic network.

        Returns:
            IPPOActorCritic network instance
        """
        action_dim = self.action_space.n if hasattr(self.action_space, 'n') else self.action_space
        return IPPOActorCritic(
            action_dim=action_dim,
            activation=self.config.get("ACTIVATION", "relu"),
            hidden_size=self.config.get("HIDDEN_SIZE", 64),
        )

    def _build_optimizer(self) -> Any:
        """Build and return the optimizer.

        Returns:
            Optax optimizer chain with gradient clipping and Adam
        """
        if self.config.get("ANNEAL_LR", True):
            # Learning rate schedule will be handled externally in training loop
            lr = self.config["LR"]
        else:
            lr = self.config["LR"]

        return optax.chain(
            optax.clip_by_global_norm(self.config.get("MAX_GRAD_NORM", 0.5)),
            optax.adam(learning_rate=lr, eps=1e-5),
        )

    def init_state(self, rng: jax.random.PRNGKey) -> IPPOAlgorithmState:
        """Initialize the algorithm state.

        Args:
            rng: JAX random key for initialization

        Returns:
            IPPOAlgorithmState with initialized parameters
        """
        # Get observation shape
        if hasattr(self.observation_space, 'shape'):
            obs_shape = self.observation_space.shape
        else:
            obs_shape = self.observation_space

        # Create dummy observation for initialization
        init_x = jnp.zeros((1, *obs_shape))

        # Initialize network parameters
        rng, net_rng = jax.random.split(rng)
        params = self.network.init(net_rng, init_x)

        # Initialize optimizer state
        optimizer_state = self.optimizer.init(params)

        return IPPOAlgorithmState(
            params=params,
            optimizer_state=optimizer_state,
            rng=rng,
            timestep=0,
            update_step=0,
        )

    def compute_action(
        self,
        state: AlgorithmState,
        observation: jnp.ndarray,
        rng: jax.random.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute action(s) given observation(s).

        Args:
            state: Current algorithm state
            observation: Observation array from environment
            rng: Random key for stochastic sampling
            deterministic: If True, return greedy action

        Returns:
            Tuple of (action, info_dict) where info contains log_prob and value
        """
        # Add batch dimension if needed (network expects batched input)
        single_obs = observation.ndim == len(self.observation_space.shape)
        if single_obs:
            observation = observation[jnp.newaxis, ...]

        pi, value = self.network.apply(state.params, observation)

        if deterministic:
            action = jnp.argmax(pi.logits, axis=-1)
            log_prob = pi.log_prob(action)
        else:
            action = pi.sample(seed=rng)
            log_prob = pi.log_prob(action)

        # Remove batch dimension from outputs only if we added it
        if single_obs:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        info = {
            "log_prob": log_prob,
            "value": value,
        }

        return action, info

    def compute_value(
        self,
        state: AlgorithmState,
        observation: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute value estimate for observation.

        Args:
            state: Current algorithm state
            observation: Observation array

        Returns:
            Value estimate
        """
        # Add batch dimension if needed (network expects batched input)
        single_obs = observation.ndim == len(self.observation_space.shape)
        if single_obs:
            observation = observation[jnp.newaxis, ...]

        _, value = self.network.apply(state.params, observation)

        # Remove batch dimension only if we added it
        if single_obs:
            value = value.squeeze(0)

        return value

    def compute_gae(
        self,
        traj_batch: Transition,
        last_value: jnp.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            traj_batch: Batch of transitions
            last_value: Value estimate for final observation
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (advantages, value_targets)
        """
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_value), last_value),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def update(
        self,
        state: AlgorithmState,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[AlgorithmState, Dict[str, float]]:
        """Update algorithm parameters using a batch of experience.

        Args:
            state: Current algorithm state
            batch: Dictionary containing:
                - obs: Observations
                - actions: Actions taken
                - advantages: GAE advantages
                - targets: Value targets
                - old_log_probs: Log probabilities from data collection

        Returns:
            Tuple of (new_state, metrics_dict)
        """
        obs = batch["obs"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        targets = batch["targets"]
        old_log_probs = batch["old_log_probs"]

        clip_eps = self.config.get("CLIP_EPS", 0.2)
        vf_coef = self.config.get("VF_COEF", 0.5)
        ent_coef = self.config.get("ENT_COEF", 0.01)

        def _loss_fn(params):
            # Forward pass
            pi, value = self.network.apply(params, obs)
            log_prob = pi.log_prob(actions)
            entropy = pi.entropy().mean()

            # Value loss with clipping
            old_value = batch.get("values", value)
            value_pred_clipped = old_value + (value - old_value).clip(-clip_eps, clip_eps)
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # Policy loss (PPO clipped surrogate objective)
            ratio = jnp.exp(log_prob - old_log_probs)
            advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss_actor1 = ratio * advantages_norm
            loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_norm
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

            # Total loss
            total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy

            return total_loss, (value_loss, loss_actor, entropy)

        (total_loss, (value_loss, actor_loss, entropy)), grads = jax.value_and_grad(
            _loss_fn, has_aux=True
        )(state.params)

        # Apply gradients
        updates, new_optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)

        new_state = IPPOAlgorithmState(
            params=new_params,
            optimizer_state=new_optimizer_state,
            rng=state.rng,
            timestep=state.timestep,
            update_step=getattr(state, 'update_step', 0) + 1,
        )

        metrics = {
            "total_loss": float(total_loss),
            "value_loss": float(value_loss),
            "actor_loss": float(actor_loss),
            "entropy": float(entropy),
        }

        return new_state, metrics
