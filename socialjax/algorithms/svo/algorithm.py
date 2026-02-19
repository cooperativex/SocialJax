"""SVO (Social Value Orientation) Algorithm Implementation.

This module implements the SVO algorithm for multi-agent reinforcement learning.
SVO incorporates social preferences into the learning process by transforming
rewards based on an SVO angle that balances self-interest and collective welfare.
"""

from typing import Any, Dict, Optional, Tuple, NamedTuple
import math

import jax
import jax.numpy as jnp
import optax
from flax import struct

from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
from socialjax.algorithms.registry import register_algorithm
from socialjax.algorithms.svo.config import SVO_DEFAULT_CONFIG, get_svo_config
from socialjax.algorithms.svo.network import SVOActorCritic


class Transition(NamedTuple):
    """Container for a single environment transition."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    original_reward: jnp.ndarray  # Original rewards before SVO transformation
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict[str, Any]


class SVOAlgorithmState(struct.PyTreeNode):
    """State container for SVO algorithm.

    Attributes:
        params: Network parameters
        optimizer_state: Optimizer state
        rng: Random key
        timestep: Current training timestep
        update_step: Number of parameter updates performed
        svo_angle: Current SVO angle (can be annealed)
    """
    params: Dict[str, Any]
    optimizer_state: Any
    rng: jax.random.PRNGKey
    timestep: int = 0
    update_step: int = 0
    svo_angle: float = 45.0  # Default cooperative angle


def compute_svo_reward(
    rewards: jnp.ndarray,
    svo_angle: float,
    use_fairness: bool = True,
    fairness_weight: float = 0.1,
) -> jnp.ndarray:
    """Compute SVO-transformed rewards.

    The SVO reward transformation combines self-interest with other-regarding
    preferences based on an angle parameter:
        r_svo = w_self * r_self + w_other * r_other

    Where:
        w_self = cos(angle)
        w_other = sin(angle)

    Additionally, a fairness penalty can be added to encourage equitable outcomes.

    Args:
        rewards: Array of shape (num_agents,) containing individual rewards
        svo_angle: SVO angle in degrees (0=selfish, 45=cooperative, 90=altruistic)
        use_fairness: Whether to add fairness penalty
        fairness_weight: Weight for fairness component

    Returns:
        Transformed rewards array of shape (num_agents,)
    """
    # Convert angle to radians
    angle_rad = math.radians(svo_angle)
    w_self = math.cos(angle_rad)
    w_other = math.sin(angle_rad)

    # Compute average of others' rewards for each agent
    num_agents = rewards.shape[-1]
    # For each agent, r_other = mean of all other agents' rewards
    r_sum = jnp.sum(rewards)
    r_other = (r_sum - rewards) / (num_agents - 1)

    # SVO transformation
    svo_rewards = w_self * rewards + w_other * r_other

    # Optional fairness penalty
    if use_fairness:
        # Fairness penalty: negative variance of rewards
        reward_mean = jnp.mean(rewards)
        reward_variance = jnp.mean((rewards - reward_mean) ** 2)
        fairness_penalty = -fairness_weight * reward_variance
        svo_rewards = svo_rewards + fairness_penalty

    return svo_rewards


def compute_batch_svo_reward(
    batch_rewards: jnp.ndarray,
    svo_angle: float,
    use_fairness: bool = True,
    fairness_weight: float = 0.1,
) -> jnp.ndarray:
    """Compute SVO-transformed rewards for a batch of transitions.

    Args:
        batch_rewards: Array of shape (num_steps, num_agents) or (num_steps, num_envs, num_agents)
        svo_angle: SVO angle in degrees
        use_fairness: Whether to add fairness penalty
        fairness_weight: Weight for fairness component

    Returns:
        Transformed rewards array with same shape as input
    """
    # Apply SVO transformation
    angle_rad = math.radians(svo_angle)
    w_self = math.cos(angle_rad)
    w_other = math.sin(angle_rad)

    # Compute SVO transformation (same for all shapes)
    num_agents = batch_rewards.shape[-1]
    r_sum = jnp.sum(batch_rewards, axis=-1, keepdims=True)
    r_other = (r_sum - batch_rewards) / (num_agents - 1)
    svo_rewards = w_self * batch_rewards + w_other * r_other

    # Optional fairness penalty (broadcast across agents dimension)
    if use_fairness:
        reward_mean = jnp.mean(batch_rewards, axis=-1, keepdims=True)
        reward_variance = jnp.mean((batch_rewards - reward_mean) ** 2, axis=-1, keepdims=True)
        fairness_penalty = -fairness_weight * reward_variance
        svo_rewards = svo_rewards + fairness_penalty

    return svo_rewards


@register_algorithm("svo")
class SVOAlgorithm(BaseAlgorithm):
    """Social Value Orientation (SVO) algorithm for multi-agent RL.

    SVO incorporates social preferences into learning by transforming rewards
    based on an SVO angle. This allows agents to balance self-interest with
    collective welfare.

    Key features:
    - SVO reward transformation based on angle parameter
    - Optional fairness-aware reward shaping
    - Based on PPO with GAE for policy optimization

    Attributes:
        observation_space: Environment observation space
        action_space: Environment action space
        config: Algorithm configuration dictionary
        network: SVOActorCritic network
        optimizer: Optax optimizer
        svo_angle: Current SVO angle in degrees
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SVO algorithm.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            config: Algorithm configuration (uses defaults if None)
        """
        merged_config = get_svo_config(config)
        super().__init__(observation_space, action_space, merged_config)

    def _build_network(self) -> Any:
        """Build and return the Actor-Critic network.

        Returns:
            SVOActorCritic network instance
        """
        action_dim = self.action_space.n if hasattr(self.action_space, 'n') else self.action_space
        return SVOActorCritic(
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
            lr = self.config["LR"]
        else:
            lr = self.config["LR"]

        return optax.chain(
            optax.clip_by_global_norm(self.config.get("MAX_GRAD_NORM", 0.5)),
            optax.adam(learning_rate=lr, eps=1e-5),
        )

    def init_state(self, rng: jax.random.PRNGKey) -> SVOAlgorithmState:
        """Initialize the algorithm state.

        Args:
            rng: JAX random key for initialization

        Returns:
            SVOAlgorithmState with initialized parameters
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

        return SVOAlgorithmState(
            params=params,
            optimizer_state=optimizer_state,
            rng=rng,
            timestep=0,
            update_step=0,
            svo_angle=self.config.get("SVO_ANGLE", 45.0),
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
                transition.reward,  # This is SVO-transformed reward
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

    def transform_rewards(
        self,
        rewards: jnp.ndarray,
        state: SVOAlgorithmState,
    ) -> jnp.ndarray:
        """Transform rewards using SVO angle.

        Args:
            rewards: Original rewards from environment
            state: Current algorithm state (contains SVO angle)

        Returns:
            SVO-transformed rewards
        """
        use_fairness = self.config.get("USE_FAIRNESS_REWARD", True)
        fairness_weight = self.config.get("FAIRNESS_WEIGHT", 0.1)

        return compute_batch_svo_reward(
            rewards,
            state.svo_angle,
            use_fairness=use_fairness,
            fairness_weight=fairness_weight,
        )

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

        new_state = SVOAlgorithmState(
            params=new_params,
            optimizer_state=new_optimizer_state,
            rng=state.rng,
            timestep=state.timestep,
            update_step=getattr(state, 'update_step', 0) + 1,
            svo_angle=getattr(state, 'svo_angle', 45.0),
        )

        metrics = {
            "total_loss": float(total_loss),
            "value_loss": float(value_loss),
            "actor_loss": float(actor_loss),
            "entropy": float(entropy),
            "svo_angle": float(state.svo_angle),
        }

        return new_state, metrics
