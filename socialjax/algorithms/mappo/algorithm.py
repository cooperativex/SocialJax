"""MAPPO (Multi-Agent Proximal Policy Optimization) Algorithm Implementation.

This module implements the MAPPO algorithm for multi-agent reinforcement learning.
MAPPO uses centralized training with decentralized execution (CTDE):
- Centralized critic has access to global state (all agent observations)
- Decentralized actor only has access to local observations

Key features:
- Separate actor and critic networks with independent optimizers
- Centralized value function estimation
- Parameter sharing across agents (all agents use same networks)
- PPO clipped surrogate objective
- Generalized Advantage Estimation (GAE)
"""

from typing import Any, Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct

from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
from socialjax.algorithms.registry import register_algorithm
from socialjax.algorithms.mappo.config import MAPPO_DEFAULT_CONFIG, get_mappo_config
from socialjax.algorithms.mappo.network import MAPPOActor, MAPPOCritic


class Transition(NamedTuple):
    """Container for a single environment transition in MAPPO.

    Attributes:
        global_done: Whether the entire episode is done
        done: Per-agent done flags
        action: Actions taken
        value: Value estimates
        reward: Rewards received
        log_prob: Log probabilities of actions
        obs: Local observations (for actor)
        world_state: Global state (for critic)
        info: Additional information
    """
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: Dict[str, Any]


class MAPPOAlgorithmState(struct.PyTreeNode):
    """State container for MAPPO algorithm.

    Attributes:
        actor_params: Actor network parameters
        critic_params: Critic network parameters
        actor_optimizer_state: Actor optimizer state
        critic_optimizer_state: Critic optimizer state
        rng: Random key
        timestep: Current training timestep
        update_step: Number of parameter updates performed
    """
    actor_params: Dict[str, Any]
    critic_params: Dict[str, Any]
    actor_optimizer_state: Any
    critic_optimizer_state: Any
    rng: jax.random.PRNGKey
    timestep: int = 0
    update_step: int = 0


@register_algorithm("mappo")
class MAPPOAlgorithm(BaseAlgorithm):
    """Multi-Agent Proximal Policy Optimization (MAPPO) for multi-agent RL.

    MAPPO uses centralized training with decentralized execution (CTDE):
    - The critic has access to global information (all agent observations)
    - The actor only has access to local observations (for decentralized execution)

    This implementation follows the standard MAPPO algorithm with:
    - Separate actor and critic networks with independent optimizers
    - Clipped surrogate objective for the actor
    - Value function clipping for the critic
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration

    Attributes:
        observation_space: Environment observation space
        action_space: Environment action space
        config: Algorithm configuration dictionary
        num_agents: Number of agents in the environment
        actor_network: MAPPOActor network
        critic_network: MAPPOCritic network
        actor_optimizer: Optax optimizer for actor
        critic_optimizer: Optax optimizer for critic
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        config: Optional[Dict[str, Any]] = None,
        num_agents: int = 1,
    ):
        """Initialize the MAPPO algorithm.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            config: Algorithm configuration (uses defaults if None)
            num_agents: Number of agents in the environment
        """
        merged_config = get_mappo_config(config)
        self.num_agents = num_agents
        super().__init__(observation_space, action_space, merged_config)

    def _build_network(self) -> Tuple[Any, Any]:
        """Build and return the Actor and Critic networks.

        Returns:
            Tuple of (actor_network, critic_network)
        """
        action_dim = self.action_space.n if hasattr(self.action_space, 'n') else self.action_space
        activation = self.config.get("ACTIVATION", "relu")
        hidden_size = self.config.get("HIDDEN_SIZE", 64)

        actor_network = MAPPOActor(
            action_dim=action_dim,
            activation=activation,
            hidden_size=hidden_size,
        )

        critic_network = MAPPOCritic(
            activation=activation,
            hidden_size=hidden_size,
        )

        return actor_network, critic_network

    def _build_optimizer(self) -> Tuple[Any, Any]:
        """Build and return the optimizers for actor and critic.

        Returns:
            Tuple of (actor_optimizer, critic_optimizer)
        """
        max_grad_norm = self.config.get("MAX_GRAD_NORM", 0.5)
        lr_actor = self.config.get("LR_ACTOR", self.config["LR"])
        lr_critic = self.config.get("LR_CRITIC", self.config["LR"])

        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=lr_actor, eps=1e-5),
        )

        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=lr_critic, eps=1e-5),
        )

        return actor_optimizer, critic_optimizer

    def init_state(self, rng: jax.random.PRNGKey) -> MAPPOAlgorithmState:
        """Initialize the algorithm state.

        Args:
            rng: JAX random key for initialization

        Returns:
            MAPPOAlgorithmState with initialized parameters
        """
        # Get observation shape
        if hasattr(self.observation_space, 'shape'):
            obs_shape = self.observation_space.shape
        else:
            obs_shape = self.observation_space

        # Create dummy observation for actor initialization
        actor_init_x = jnp.zeros((1, *obs_shape))

        # Create dummy world state for critic initialization
        # World state has shape (1, *obs_shape[:-1], obs_shape[-1] * num_agents)
        world_state_shape = (*obs_shape[:-1], obs_shape[-1] * self.num_agents)
        critic_init_x = jnp.zeros((1, *world_state_shape))

        # Initialize networks
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        actor_params = self.network[0].init(actor_rng, actor_init_x)
        critic_params = self.network[1].init(critic_rng, critic_init_x)

        # Initialize optimizers
        actor_optimizer_state = self.optimizer[0].init(actor_params)
        critic_optimizer_state = self.optimizer[1].init(critic_params)

        return MAPPOAlgorithmState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
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
            observation: Observation array from environment (local observation)
            rng: Random key for stochastic sampling
            deterministic: If True, return greedy action

        Returns:
            Tuple of (action, info_dict) where info contains log_prob
        """
        actor_network = self.network[0]
        pi = actor_network.apply(state.actor_params, observation)

        if deterministic:
            action = jnp.argmax(pi.logits, axis=-1)
            log_prob = pi.log_prob(action)
        else:
            action = pi.sample(seed=rng)
            log_prob = pi.log_prob(action)

        info = {
            "log_prob": log_prob,
        }

        return action, info

    def compute_value(
        self,
        state: AlgorithmState,
        world_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute value estimate for world state.

        Args:
            state: Current algorithm state
            world_state: Global state array (all agent observations)

        Returns:
            Value estimate
        """
        critic_network = self.network[1]
        return critic_network.apply(state.critic_params, world_state)

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
                - obs: Local observations
                - world_state: Global state (all agent observations)
                - actions: Actions taken
                - advantages: GAE advantages
                - targets: Value targets
                - old_log_probs: Log probabilities from data collection
                - old_values: Value estimates from data collection

        Returns:
            Tuple of (new_state, metrics_dict)
        """
        obs = batch["obs"]
        world_state = batch["world_state"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        targets = batch["targets"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch.get("values", batch.get("old_values"))

        clip_eps = self.config.get("CLIP_EPS", 0.2)
        vf_coef = self.config.get("VF_COEF", 0.5)
        ent_coef = self.config.get("ENT_COEF", 0.01)

        actor_network = self.network[0]
        critic_network = self.network[1]

        # Actor loss function
        def _actor_loss_fn(actor_params):
            pi = actor_network.apply(actor_params, obs)
            log_prob = pi.log_prob(actions)
            entropy = pi.entropy().mean()

            # Policy loss (PPO clipped surrogate objective)
            ratio = jnp.exp(log_prob - old_log_probs)
            advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss_actor1 = ratio * advantages_norm
            loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_norm
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

            # Total actor loss (with entropy bonus)
            actor_loss = loss_actor - ent_coef * entropy

            return actor_loss, (loss_actor, entropy, ratio)

        # Critic loss function
        def _critic_loss_fn(critic_params):
            value = critic_network.apply(critic_params, world_state)

            # Value loss with clipping
            value_pred_clipped = old_values + (value - old_values).clip(-clip_eps, clip_eps)
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            return vf_coef * value_loss, value_loss

        # Compute gradients for actor
        (actor_loss, (actor_loss_val, entropy, ratio)), actor_grads = jax.value_and_grad(
            _actor_loss_fn, has_aux=True
        )(state.actor_params)

        # Compute gradients for critic
        (critic_loss, value_loss), critic_grads = jax.value_and_grad(
            _critic_loss_fn, has_aux=True
        )(state.critic_params)

        # Apply gradients to actor
        actor_updates, new_actor_optimizer_state = self.optimizer[0].update(
            actor_grads, state.actor_optimizer_state, state.actor_params
        )
        new_actor_params = optax.apply_updates(state.actor_params, actor_updates)

        # Apply gradients to critic
        critic_updates, new_critic_optimizer_state = self.optimizer[1].update(
            critic_grads, state.critic_optimizer_state, state.critic_params
        )
        new_critic_params = optax.apply_updates(state.critic_params, critic_updates)

        new_state = MAPPOAlgorithmState(
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            actor_optimizer_state=new_actor_optimizer_state,
            critic_optimizer_state=new_critic_optimizer_state,
            rng=state.rng,
            timestep=state.timestep,
            update_step=getattr(state, 'update_step', 0) + 1,
        )

        total_loss = float(actor_loss) + float(critic_loss)

        metrics = {
            "total_loss": total_loss,
            "actor_loss": float(actor_loss_val),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
        }

        return new_state, metrics

    def save(self, state: AlgorithmState, path: str) -> None:
        """Save algorithm state to file.

        Args:
            state: Algorithm state to save
            path: File path to save to
        """
        import pickle
        save_dict = {
            "actor_params": state.actor_params,
            "critic_params": state.critic_params,
            "actor_optimizer_state": state.actor_optimizer_state,
            "critic_optimizer_state": state.critic_optimizer_state,
            "timestep": state.timestep,
            "update_step": getattr(state, 'update_step', 0),
            "config": self.config,
            "num_agents": self.num_agents,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, path: str) -> MAPPOAlgorithmState:
        """Load algorithm state from file.

        Args:
            path: File path to load from

        Returns:
            Loaded algorithm state
        """
        import pickle
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        rng = jax.random.PRNGKey(0)  # RNG will need to be reset
        return MAPPOAlgorithmState(
            actor_params=save_dict["actor_params"],
            critic_params=save_dict["critic_params"],
            actor_optimizer_state=save_dict["actor_optimizer_state"],
            critic_optimizer_state=save_dict["critic_optimizer_state"],
            rng=rng,
            timestep=save_dict.get("timestep", 0),
            update_step=save_dict.get("update_step", 0),
        )
