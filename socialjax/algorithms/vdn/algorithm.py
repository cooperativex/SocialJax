"""VDN (Value Decomposition Networks) Algorithm Implementation.

This module implements the VDN algorithm for cooperative multi-agent
reinforcement learning. VDN decomposes the team Q-value into individual
agent Q-values, enabling decentralized execution with coordinated learning.

Key features:
- Value decomposition: Q_tot = sum_i Q_i(s_i, a_i)
- Target networks for stable training
- Experience replay (off-policy learning)
- Epsilon-greedy exploration with decay
- Soft or hard target network updates

Reference:
    "Value-Decomposition Networks for Cooperative Multi-Agent Learning"
    (Sunehag et al., 2018)
"""

from typing import Any, Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct

from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
from socialjax.algorithms.registry import register_algorithm
from socialjax.algorithms.vdn.config import VDN_DEFAULT_CONFIG, get_vdn_config
from socialjax.algorithms.vdn.network import VDNQNetwork, compute_vdn_target


class VDNTransition(NamedTuple):
    """Container for a single environment transition in VDN.

    Attributes:
        obs: Observations
        actions: Actions taken
        rewards: Team rewards received
        dones: Done flags
        next_obs: Next observations
    """

    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    next_obs: jnp.ndarray


class VDNAlgorithmState(struct.PyTreeNode):
    """State container for VDN algorithm.

    Attributes:
        params: Main Q-network parameters
        target_params: Target Q-network parameters
        optimizer_state: Optimizer state
        rng: Random key
        timestep: Current training timestep
        update_step: Number of parameter updates performed
        epsilon: Current exploration epsilon
    """

    params: Dict[str, Any]
    target_params: Dict[str, Any]
    optimizer_state: Any
    rng: jax.random.PRNGKey
    timestep: int = 0
    update_step: int = 0
    epsilon: float = 1.0


@register_algorithm("vdn")
class VDNAlgorithm(BaseAlgorithm):
    """Value Decomposition Networks (VDN) for cooperative multi-agent RL.

    VDN decomposes the global Q-value into individual agent Q-values:
        Q_tot(s, a) = sum_i Q_i(s_i, a_i)

    This enables centralized training with decentralized execution (CTDE):
    - Training uses the decomposed Q-values for coordinated learning
    - Execution can be done independently by each agent using local Q-values

    This implementation includes:
    - Target networks for stable TD learning
    - Experience replay for off-policy learning
    - Epsilon-greedy exploration with decay schedule
    - Soft or hard target network updates

    Attributes:
        observation_space: Environment observation space
        action_space: Environment action space
        config: Algorithm configuration dictionary
        num_agents: Number of agents in the environment
        network: VDNQNetwork for Q-value estimation
        optimizer: Optax optimizer
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        config: Optional[Dict[str, Any]] = None,
        num_agents: int = 1,
    ):
        """Initialize the VDN algorithm.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            config: Algorithm configuration (uses defaults if None)
            num_agents: Number of agents in the environment
        """
        merged_config = get_vdn_config(config)
        self.num_agents = num_agents
        super().__init__(observation_space, action_space, merged_config)

    def _build_network(self) -> VDNQNetwork:
        """Build and return the Q-network.

        Returns:
            VDNQNetwork instance
        """
        action_dim = (
            self.action_space.n
            if hasattr(self.action_space, "n")
            else self.action_space
        )
        return VDNQNetwork(
            action_dim=action_dim,
            activation=self.config.get("ACTIVATION", "relu"),
            hidden_size=self.config.get("HIDDEN_SIZE", 64),
        )

    def _build_optimizer(self) -> Any:
        """Build and return the optimizer.

        Returns:
            Optax optimizer chain with gradient clipping and Adam/RAdam
        """
        return optax.chain(
            optax.clip_by_global_norm(self.config.get("MAX_GRAD_NORM", 10.0)),
            optax.adam(learning_rate=self.config["LR"], eps=1e-5),
        )

    def init_state(self, rng: jax.random.PRNGKey) -> VDNAlgorithmState:
        """Initialize the algorithm state.

        Args:
            rng: JAX random key for initialization

        Returns:
            VDNAlgorithmState with initialized parameters
        """
        # Get observation shape
        if hasattr(self.observation_space, "shape"):
            obs_shape = self.observation_space.shape
        else:
            obs_shape = self.observation_space

        # Create dummy observation for initialization
        init_x = jnp.zeros((1, *obs_shape))

        # Initialize network parameters
        rng, net_rng = jax.random.split(rng)
        params = self.network.init(net_rng, init_x)

        # Initialize target network with same parameters
        target_params = params

        # Initialize optimizer state
        optimizer_state = self.optimizer.init(params)

        return VDNAlgorithmState(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            rng=rng,
            timestep=0,
            update_step=0,
            epsilon=self.config.get("EPS_START", 1.0),
        )

    def get_epsilon(self, update_step: int) -> float:
        """Get current exploration epsilon based on decay schedule.

        Args:
            update_step: Current update step

        Returns:
            Current epsilon value
        """
        eps_start = self.config.get("EPS_START", 1.0)
        eps_finish = self.config.get("EPS_FINISH", 0.05)
        eps_decay = self.config.get("EPS_DECAY", 0.5)

        # Linear decay over eps_decay fraction of training
        # This is a simplified version - in practice, you'd use a schedule
        decay_steps = int(eps_decay * 10000)  # Assume 10K total updates
        epsilon = max(
            eps_finish,
            eps_start - (eps_start - eps_finish) * min(update_step / decay_steps, 1.0),
        )
        return epsilon

    def compute_action(
        self,
        state: AlgorithmState,
        observation: jnp.ndarray,
        rng: jax.random.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute action(s) given observation(s) using epsilon-greedy.

        Args:
            state: Current algorithm state
            observation: Observation array from environment
            rng: Random key for stochastic sampling
            deterministic: If True, always use greedy action (epsilon=0)

        Returns:
            Tuple of (action, info_dict) where info contains q_values
        """
        # Get Q-values for all agents
        # observation shape: (num_agents, batch, *obs_shape) or (batch, *obs_shape)
        if observation.ndim > len(self.observation_space.shape) + 1:
            # Multiple agents: vmap over agent dimension
            q_values = jax.vmap(self.network.apply, in_axes=(None, 0))(
                state.params, observation
            )
        else:
            # Single agent or single batch
            q_values = self.network.apply(state.params, observation)

        if deterministic:
            # Greedy action selection
            actions = jnp.argmax(q_values, axis=-1)
        else:
            # Epsilon-greedy action selection
            epsilon = state.epsilon

            # Greedy actions
            greedy_actions = jnp.argmax(q_values, axis=-1)

            # Random actions
            action_dim = q_values.shape[-1]
            random_actions = jax.random.randint(rng, greedy_actions.shape, 0, action_dim)

            # Epsilon-greedy selection
            explore = jax.random.uniform(rng, greedy_actions.shape) < epsilon
            actions = jnp.where(explore, random_actions, greedy_actions)

        info = {
            "q_values": q_values,
            "epsilon": state.epsilon,
        }

        return actions, info

    def compute_q_values(
        self,
        state: AlgorithmState,
        observation: jnp.ndarray,
        use_target: bool = False,
    ) -> jnp.ndarray:
        """Compute Q-values for given observations.

        Args:
            state: Current algorithm state
            observation: Observation array
            use_target: If True, use target network parameters

        Returns:
            Q-values array
        """
        params = state.target_params if use_target else state.params

        if observation.ndim > len(self.observation_space.shape) + 1:
            # Multiple agents
            q_values = jax.vmap(self.network.apply, in_axes=(None, 0))(
                params, observation
            )
        else:
            q_values = self.network.apply(params, observation)

        return q_values

    def compute_q_tot(
        self,
        state: AlgorithmState,
        observation: jnp.ndarray,
        actions: jnp.ndarray,
        use_target: bool = False,
    ) -> jnp.ndarray:
        """Compute total team Q-value for given state-action pair.

        VDN decomposition: Q_tot = sum_i Q_i(s_i, a_i)

        Args:
            state: Current algorithm state
            observation: Observation array (num_agents, batch, *obs_shape)
            actions: Actions taken (num_agents, batch) or (batch,)
            use_target: If True, use target network parameters

        Returns:
            Total Q-value (batch_size,)
        """
        params = state.target_params if use_target else state.params

        # Get Q-values for all agents
        q_values = jax.vmap(self.network.apply, in_axes=(None, 0))(
            params, observation
        )  # (num_agents, batch, action_dim)

        # Get Q-values for chosen actions
        if actions.ndim == 1:
            # Single agent or shared actions
            actions = actions[jnp.newaxis, :]  # Add agent dimension

        chosen_q_values = jnp.take_along_axis(
            q_values, actions[..., jnp.newaxis], axis=-1
        ).squeeze(-1)  # (num_agents, batch)

        # Sum across agents for Q_tot
        q_tot = jnp.sum(chosen_q_values, axis=0)

        return q_tot

    def update(
        self,
        state: AlgorithmState,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[AlgorithmState, Dict[str, float]]:
        """Update algorithm parameters using a batch of experience.

        This implements the VDN TD loss:
            L = E[(Q_tot(s, a) - y)^2]
            where y = r + gamma * sum_i max_a' Q_i^target(s', a')

        Args:
            state: Current algorithm state
            batch: Dictionary containing:
                - obs: Observations (num_agents, batch, *obs_shape)
                - actions: Actions taken (num_agents, batch)
                - rewards: Team rewards (batch,)
                - dones: Done flags (batch,)
                - next_obs: Next observations (num_agents, batch, *obs_shape)

        Returns:
            Tuple of (new_state, metrics_dict)
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        next_obs = batch["next_obs"]

        gamma = self.config.get("GAMMA", 0.99)
        tau = self.config.get("TAU", 1.0)

        def _loss_fn(params):
            # Compute current Q-values for all agents
            q_values = jax.vmap(self.network.apply, in_axes=(None, 0))(
                params, obs
            )  # (num_agents, batch, action_dim)

            # Get Q-values for chosen actions
            chosen_q_values = jnp.take_along_axis(
                q_values, actions[..., jnp.newaxis], axis=-1
            ).squeeze(-1)  # (num_agents, batch)

            # Sum across agents for Q_tot
            q_tot = jnp.sum(chosen_q_values, axis=0)  # (batch,)

            # Compute target Q-values using target network
            q_next_target = jax.vmap(self.network.apply, in_axes=(None, 0))(
                state.target_params, next_obs
            )  # (num_agents, batch, action_dim)

            # Get max Q-values for each agent and sum for Q_tot_target
            q_next_max = jnp.max(q_next_target, axis=-1)  # (num_agents, batch)
            q_tot_next = jnp.sum(q_next_max, axis=0)  # (batch,)

            # Compute TD target
            target = rewards + (1 - dones) * gamma * q_tot_next

            # MSE loss
            loss = jnp.mean((q_tot - target) ** 2)

            return loss, (q_tot.mean(), target.mean())

        (loss, (q_tot_mean, target_mean)), grads = jax.value_and_grad(
            _loss_fn, has_aux=True
        )(state.params)

        # Apply gradients
        updates, new_optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)

        # Update target network (soft or hard update)
        new_update_step = state.update_step + 1
        target_update_interval = self.config.get("TARGET_UPDATE_INTERVAL", 200)

        # Determine if we should update target network
        should_update = (new_update_step % target_update_interval) == 0

        # Soft update: target = tau * params + (1 - tau) * target
        # Hard update (tau=1): target = params
        new_target_params = jax.lax.cond(
            should_update,
            lambda _: optax.incremental_update(new_params, state.target_params, tau),
            lambda _: state.target_params,
            operand=None,
        )

        # Update epsilon
        new_epsilon = self.get_epsilon(new_update_step)

        new_state = VDNAlgorithmState(
            params=new_params,
            target_params=new_target_params,
            optimizer_state=new_optimizer_state,
            rng=state.rng,
            timestep=state.timestep,
            update_step=new_update_step,
            epsilon=new_epsilon,
        )

        metrics = {
            "loss": float(loss),
            "q_tot_mean": float(q_tot_mean),
            "target_mean": float(target_mean),
            "epsilon": float(new_epsilon),
        }

        return new_state, metrics

    def update_target_network(self, state: VDNAlgorithmState) -> VDNAlgorithmState:
        """Force update of target network parameters.

        Args:
            state: Current algorithm state

        Returns:
            State with updated target parameters
        """
        tau = self.config.get("TAU", 1.0)
        new_target_params = optax.incremental_update(
            state.params, state.target_params, tau
        )

        return state.replace(target_params=new_target_params)

    def save(self, state: AlgorithmState, path: str) -> None:
        """Save algorithm state to file.

        Args:
            state: Algorithm state to save
            path: File path to save to
        """
        import pickle

        save_dict = {
            "params": state.params,
            "target_params": state.target_params,
            "optimizer_state": state.optimizer_state,
            "timestep": state.timestep,
            "update_step": state.update_step,
            "epsilon": state.epsilon,
            "config": self.config,
            "num_agents": self.num_agents,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, path: str) -> VDNAlgorithmState:
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
        return VDNAlgorithmState(
            params=save_dict["params"],
            target_params=save_dict["target_params"],
            optimizer_state=save_dict["optimizer_state"],
            rng=rng,
            timestep=save_dict.get("timestep", 0),
            update_step=save_dict.get("update_step", 0),
            epsilon=save_dict.get("epsilon", 1.0),
        )
