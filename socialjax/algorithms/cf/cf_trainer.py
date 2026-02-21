"""
Module: CF Training Loop
Equation: Algorithm 1 from paper

Implements the complete Counterfactual Regret training loop that:
1. Collects experience from environment
2. Computes counterfactual regret using generative model
3. Computes shaped rewards (extrinsic + alpha * intrinsic)
4. Updates policy using PPO with shaped rewards
5. Updates generative model to improve reward prediction

Reference: Counterfactual/cf_method
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import Dict, Tuple, Optional, Any, NamedTuple, Sequence
from functools import partial
import optax
import numpy as np
import os
import pickle
from dataclasses import dataclass, field

# Import CF modules
from socialjax.algorithms.cf.generative_model import (
    RewardModel,
    generative_model_loss,
    compute_generative_model_loss,
    create_reward_model_train_state,
)
from socialjax.algorithms.cf.counterfactual import (
    generate_counterfactual_rewards_vmap,
    compute_collective_cf_reward,
    compute_actual_collective_reward,
    get_counterfactual_analysis,
)
from socialjax.algorithms.cf.regret import (
    compute_counterfactual_regret,
    get_regret_statistics,
)
from socialjax.algorithms.cf.intrinsic_reward import (
    compute_intrinsic_reward,
    get_intrinsic_reward_statistics,
)
from socialjax.algorithms.cf.reward_shaping import (
    compute_shaped_reward,
    compute_alpha_n_minus_1,
    get_shaped_reward_statistics,
)
from socialjax.algorithms.cf.policy import (
    ActorCritic,
    Transition,
    compute_gae,
    compute_ppo_loss,
    create_actor_critic_train_state,
    ppo_update_epoch,
    get_action,
    get_value,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CFConfig:
    """Configuration for CF training."""
    # Environment
    env_name: str = "coin_game"
    num_agents: int = 3
    num_envs: int = 8

    # Training
    total_timesteps: int = 1_000_000
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4

    # Learning rates
    policy_lr: float = 0.0003
    reward_lr: float = 0.001
    anneal_lr: bool = True

    # PPO parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # CF parameters
    alpha: float = 2.0  # Use N-1 for suggested value
    use_auto_alpha: bool = True

    # Reward model
    reward_model_warmup: int = 100  # Steps before using CF rewards
    reward_update_freq: int = 1

    # Network architecture
    cnn_features: Tuple[int, ...] = (32, 32, 32)
    cnn_kernels: Tuple[Tuple[int, int], ...] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    activation: str = "relu"

    # Checkpointing
    save_dir: str = "checkpoints/cf"
    save_freq: int = 10000

    # Logging
    log_freq: int = 100
    use_wandb: bool = False
    wandb_project: str = "socialjax"
    wandb_entity: str = None

    def __post_init__(self):
        """Compute derived values."""
        self.num_actors = self.num_agents * self.num_envs
        self.num_updates = self.total_timesteps // self.num_steps // self.num_envs
        self.minibatch_size = self.num_actors * self.num_steps // self.num_minibatches
        if self.use_auto_alpha:
            self.alpha = compute_alpha_n_minus_1(self.num_agents)


# ============================================================================
# Training State
# ============================================================================

class CFRunnerState(NamedTuple):
    """State maintained during training."""
    policy_state: TrainState
    reward_state: TrainState
    env_state: Any
    last_obs: jnp.ndarray
    global_step: int
    rng: jax.random.PRNGKey


class CFUpdateState(NamedTuple):
    """State for a single update."""
    runner_state: CFRunnerState
    traj_batch: Transition
    metrics: Dict[str, Any]


# ============================================================================
# Buffer for storing transitions
# ============================================================================

class TransitionBuffer:
    """Buffer for storing environment transitions."""

    def __init__(self, num_steps: int, num_envs: int, num_agents: int, obs_shape: Tuple):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_shape = obs_shape

        # Storage
        self.obs = np.zeros((num_steps, num_envs, num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, num_agents), dtype=np.int32)
        self.rewards = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, actions, rewards, dones, log_probs, values):
        """Add a transition to the buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values

        self.ptr = (self.ptr + 1) % self.num_steps
        if self.ptr == 0:
            self.full = True

    def get(self) -> Dict[str, np.ndarray]:
        """Get all transitions."""
        return {
            'obs': self.obs,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'log_probs': self.log_probs,
            'values': self.values,
        }

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.full = False


# ============================================================================
# CF Trainer
# ============================================================================

class CFTrainer:
    """
    Complete Counterfactual Regret training implementation.

    Integrates all modules (M1-M7) into a training loop following Algorithm 1:
    1. Collect experience using current policy
    2. Train generative model on collected (obs, action, reward) tuples
    3. Compute counterfactual regret using generative model
    4. Compute shaped rewards (extrinsic + alpha * intrinsic)
    5. Update policy using PPO with shaped rewards

    Attributes:
        config: CFConfig with training parameters
        env: JAX environment
        policy_network: ActorCritic network for policy
        reward_model: RewardModel for counterfactual reasoning
    """

    def __init__(
        self,
        config: CFConfig,
        env,
        policy_network: Optional[ActorCritic] = None,
        reward_model: Optional[RewardModel] = None,
    ):
        """
        Initialize the CF trainer.

        Args:
            config: Training configuration
            env: JAX environment (must have num_agents, action_space, observation_space)
            policy_network: Optional pre-defined policy network
            reward_model: Optional pre-defined reward model
        """
        self.config = config
        self.env = env

        # Get environment specs
        self.num_agents = env.num_agents
        self.action_dim = env.action_space().n
        self.obs_shape = env.observation_space()[0].shape

        # Create networks if not provided
        if policy_network is None:
            self.policy_network = ActorCritic(
                action_dim=self.action_dim,
                cnn_features=config.cnn_features,
                cnn_kernels=config.cnn_kernels,
                hidden_dim=config.hidden_dim,
                activation=config.activation,
            )
        else:
            self.policy_network = policy_network

        if reward_model is None:
            self.reward_model = RewardModel(
                num_agents=self.num_agents,
                action_dim=self.action_dim,
                cnn_features=config.cnn_features,
                cnn_kernels=config.cnn_kernels,
                hidden_dim=config.hidden_dim,
                activation=config.activation,
            )
        else:
            self.reward_model = reward_model

        # JIT compile training functions
        self._compile_functions()

    def _compile_functions(self):
        """JIT compile training functions for performance."""

        @jax.jit
        def _compute_shaped_rewards_batch(
            reward_params,
            obs,  # [num_steps, num_envs, num_agents, H, W, C]
            actions,  # [num_steps, num_envs, num_agents]
            rewards,  # [num_steps, num_envs, num_agents]
        ):
            """Compute shaped rewards for a batch of transitions."""
            num_steps = obs.shape[0]
            num_envs = obs.shape[1]
            num_agents = obs.shape[2]

            def _compute_for_step(step_data):
                obs_s, actions_s, rewards_s = step_data

                # Reshape for counterfactual: [batch, num_agents, H, W, C]
                obs_batch = obs_s.reshape(-1, num_agents, *self.obs_shape)
                actions_batch = actions_s.reshape(-1, num_agents)
                rewards_batch = rewards_s.reshape(-1, num_agents)

                # Generate counterfactual rewards (Eq.7)
                cf_rewards = generate_counterfactual_rewards_vmap(
                    self.reward_model.apply,
                    reward_params,
                    self.action_dim,
                    obs_batch,
                    actions_batch,
                )

                # Compute collective counterfactual rewards (Eq.8)
                collective_cf = compute_collective_cf_reward(cf_rewards, exclude_self=True)

                # Compute actual collective rewards
                actual_collective = compute_actual_collective_reward(rewards_batch)

                # Compute counterfactual regret (Eq.9)
                regret = compute_counterfactual_regret(collective_cf, actual_collective)

                # Compute intrinsic reward (Eq.10)
                intrinsic = compute_intrinsic_reward(regret)

                # Compute shaped reward (Eq.11)
                shaped = compute_shaped_reward(
                    rewards_batch, intrinsic, self.config.alpha
                )

                return shaped.reshape(num_envs, num_agents)

            # Compute for all steps
            shaped_rewards = jax.vmap(_compute_for_step)((obs, actions, rewards))
            return shaped_rewards

        self._compute_shaped_rewards_batch = _compute_shaped_rewards_batch

        @jax.jit
        def _update_reward_model(reward_state, obs, actions, rewards):
            """Update reward model with new data."""

            def loss_fn(params):
                pred = self.reward_model.apply(params, obs, actions)
                loss = jnp.mean((pred - rewards) ** 2)
                return loss, pred

            (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(reward_state.params)
            new_state = reward_state.apply_gradients(grads=grads)
            return new_state, loss, pred

        self._update_reward_model_jit = _update_reward_model

    def initialize(
        self,
        rng: jax.random.PRNGKey,
    ) -> CFRunnerState:
        """
        Initialize training state.

        Args:
            rng: JAX random key

        Returns:
            Initial CFRunnerState
        """
        config = self.config

        # Split random keys
        rng, policy_rng, reward_rng, env_rng = jax.random.split(rng, 4)

        # Initialize policy network
        sample_obs = jnp.zeros((config.num_actors, *self.obs_shape))
        policy_state, policy_rng = create_actor_critic_train_state(
            self.policy_network,
            policy_rng,
            sample_obs,
            learning_rate=config.policy_lr,
            max_grad_norm=config.max_grad_norm,
            anneal_lr=config.anneal_lr,
            num_updates=config.num_updates,
        )

        # Initialize reward model
        sample_obs_reward = jnp.zeros((1, self.num_agents, *self.obs_shape))
        sample_actions = jnp.zeros((1, self.num_agents), dtype=jnp.int32)
        reward_state, reward_rng = create_reward_model_train_state(
            self.reward_model,
            reward_rng,
            sample_obs_reward,
            sample_actions,
            learning_rate=config.reward_lr,
        )

        # Initialize environment
        env_rng = jax.random.split(env_rng, config.num_envs)
        obs, env_state = jax.vmap(self.env.reset)(env_rng)

        return CFRunnerState(
            policy_state=policy_state,
            reward_state=reward_state,
            env_state=env_state,
            last_obs=obs,
            global_step=0,
            rng=rng,
        )

    def _env_step(
        self,
        runner_state: CFRunnerState,
        unused,
    ) -> Tuple[CFRunnerState, Transition]:
        """
        Execute one environment step.

        Args:
            runner_state: Current training state
            unused: Unused (for jax.lax.scan)

        Returns:
            (new_runner_state, transition)
        """
        config = self.config
        policy_state, reward_state, env_state, last_obs, global_step, rng = runner_state

        # Prepare observations for policy [num_envs * num_agents, H, W, C]
        obs_batch = last_obs.reshape(-1, *self.obs_shape)

        # Sample actions from policy
        rng, action_rng = jax.random.split(rng)
        action_rngs = jax.random.split(action_rng, config.num_envs * self.num_agents)

        def get_action_single(rng_i, obs_i):
            pi, value = self.policy_network.apply(policy_state.params, obs_i[jnp.newaxis])
            action = pi.sample(seed=rng_i)
            log_prob = pi.log_prob(action)
            return action[0], log_prob[0], value[0]

        actions, log_probs, values = jax.vmap(get_action_single)(action_rngs, obs_batch)

        # Reshape back to [num_envs, num_agents]
        actions = actions.reshape(config.num_envs, self.num_agents)
        log_probs = log_probs.reshape(config.num_envs, self.num_agents)
        values = values.reshape(config.num_envs, self.num_agents)

        # Step environment
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, config.num_envs)

        # Convert actions to env format
        env_actions = [actions[:, i] for i in range(self.num_agents)]

        obsv, env_state_next, rewards, dones, infos = jax.vmap(
            self.env.step, in_axes=(0, 0, 0)
        )(step_rngs, env_state, env_actions)

        # Create transition
        transition = Transition(
            obs=last_obs,  # [num_envs, num_agents, H, W, C]
            action=actions,
            reward=rewards,  # Will be replaced with shaped reward
            done=dones["__all__"] if isinstance(dones, dict) else dones,
            log_prob=log_probs,
            value=values,
        )

        new_runner_state = CFRunnerState(
            policy_state=policy_state,
            reward_state=reward_state,
            env_state=env_state_next,
            last_obs=obsv,
            global_step=global_step + config.num_envs,
            rng=rng,
        )

        return new_runner_state, transition

    def _collect_trajectory(
        self,
        runner_state: CFRunnerState,
    ) -> Tuple[CFRunnerState, Transition]:
        """
        Collect a trajectory of transitions.

        Args:
            runner_state: Current training state

        Returns:
            (new_runner_state, trajectory_batch)
        """
        runner_state, traj_batch = jax.lax.scan(
            self._env_step,
            runner_state,
            None,
            length=self.config.num_steps,
        )
        return runner_state, traj_batch

    def _compute_cf_rewards(
        self,
        reward_state: TrainState,
        traj_batch: Transition,
    ) -> jnp.ndarray:
        """
        Compute counterfactual shaped rewards for trajectory.

        Implements M2 -> M3 -> M4 -> M5 -> M6 pipeline.

        Args:
            reward_state: Reward model training state
            traj_batch: Batch of transitions

        Returns:
            shaped_rewards: Shaped rewards [num_steps, num_envs, num_agents]
        """
        return self._compute_shaped_rewards_batch(
            reward_state.params,
            traj_batch.obs,
            traj_batch.action,
            traj_batch.reward,
        )

    def _update_step(
        self,
        runner_state: CFRunnerState,
        unused,
    ) -> Tuple[CFRunnerState, Dict[str, Any]]:
        """
        Perform one update step.

        This implements the main training loop:
        1. Collect trajectory
        2. Update reward model
        3. Compute CF shaped rewards
        4. Update policy with PPO

        Args:
            runner_state: Current training state
            unused: Unused (for jax.lax.scan)

        Returns:
            (new_runner_state, metrics)
        """
        config = self.config
        policy_state, reward_state, env_state, last_obs, global_step, rng = runner_state

        # 1. Collect trajectory
        runner_state, traj_batch = self._collect_trajectory(runner_state)
        policy_state, reward_state, env_state, last_obs, global_step, rng = runner_state

        # 2. Update reward model
        # Reshape for reward model: [num_steps * num_envs, num_agents, ...]
        obs_rm = traj_batch.obs.reshape(-1, self.num_agents, *self.obs_shape)
        actions_rm = traj_batch.action.reshape(-1, self.num_agents)
        rewards_rm = traj_batch.reward.reshape(-1, self.num_agents)

        # Update reward model multiple times
        def update_reward_loop(rs, _):
            new_rs, loss, pred = self._update_reward_model_jit(
                rs, obs_rm, actions_rm, rewards_rm
            )
            return new_rs, loss

        reward_state, rm_losses = jax.lax.scan(
            update_reward_loop,
            reward_state,
            None,
            length=config.update_epochs,
        )
        rm_loss = rm_losses.mean()

        # 3. Compute shaped rewards using CF
        shaped_rewards = self._compute_cf_rewards(reward_state, traj_batch)

        # 4. Compute GAE with shaped rewards
        # Get last value for GAE
        last_obs_batch = last_obs.reshape(-1, *self.obs_shape)
        _, last_value = jax.vmap(
            lambda o: self.policy_network.apply(policy_state.params, o[jnp.newaxis])
        )(last_obs_batch)
        last_value = last_value.reshape(config.num_envs, self.num_agents)

        # Create shaped transition for GAE
        shaped_traj = Transition(
            obs=traj_batch.obs,
            action=traj_batch.action,
            reward=shaped_rewards,
            done=traj_batch.done,
            log_prob=traj_batch.log_prob,
            value=traj_batch.value,
        )

        # Compute GAE for each agent
        def compute_gae_for_agent(agent_id):
            agent_traj = Transition(
                obs=shaped_traj.obs[:, :, agent_id],
                action=shaped_traj.action[:, :, agent_id],
                reward=shaped_traj.reward[:, :, agent_id],
                done=shaped_traj.done,
                log_prob=shaped_traj.log_prob[:, :, agent_id],
                value=shaped_traj.value[:, :, agent_id],
            )
            advantages, targets = compute_gae(
                agent_traj,
                last_value[:, agent_id],
                config.gamma,
                config.gae_lambda,
            )
            return advantages, targets

        # Compute for all agents
        advantages_all, targets_all = jax.vmap(compute_gae_for_agent)(
            jnp.arange(self.num_agents)
        )
        # Reshape: [num_agents, num_steps, num_envs] -> [num_steps, num_envs, num_agents]
        advantages_all = advantages_all.transpose(1, 2, 0)
        targets_all = targets_all.transpose(1, 2, 0)

        # 5. Update policy with PPO
        rng, update_rng = jax.random.split(rng)

        def _ppo_epoch_step(carry, _):
            """Single PPO epoch step for scan."""
            ps, rng = carry
            ps, rng, epoch_metrics = ppo_update_epoch(
                ps,
                shaped_traj,
                advantages_all,
                targets_all,
                rng,
                num_minibatches=config.num_minibatches,
                update_epochs=1,  # Single epoch in scan
                clip_eps=config.clip_eps,
                vf_coef=config.vf_coef,
                ent_coef=config.ent_coef,
            )
            return (ps, rng), epoch_metrics

        (policy_state, rng), ppo_metrics = jax.lax.scan(
            _ppo_epoch_step,
            (policy_state, update_rng),
            None,
            length=config.update_epochs,
        )

        # Aggregate metrics
        metrics = {
            'reward_model_loss': rm_loss,
            'policy_loss': ppo_metrics['total_loss'].mean(),
            'value_loss': ppo_metrics['value_loss'].mean(),
            'actor_loss': ppo_metrics['actor_loss'].mean(),
            'entropy': ppo_metrics['entropy'].mean(),
            'mean_reward': traj_batch.reward.mean(),
            'mean_shaped_reward': shaped_rewards.mean(),
        }

        new_runner_state = CFRunnerState(
            policy_state=policy_state,
            reward_state=reward_state,
            env_state=env_state,
            last_obs=last_obs,
            global_step=global_step,
            rng=rng,
        )

        return new_runner_state, metrics

    def train(
        self,
        num_updates: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> Tuple[CFRunnerState, Dict[str, Any]]:
        """
        Run the training loop.

        Args:
            num_updates: Number of updates to run (default: config.num_updates)
            callback: Optional callback function called each update with (step, metrics)

        Returns:
            (final_runner_state, training_metrics)
        """
        if num_updates is None:
            num_updates = self.config.num_updates

        # Initialize
        rng = jax.random.PRNGKey(0)
        runner_state = self.initialize(rng)

        # Training loop
        all_metrics = []

        for update_idx in range(num_updates):
            runner_state, metrics = self._update_step(runner_state, None)
            all_metrics.append(metrics)

            # Log
            if update_idx % self.config.log_freq == 0:
                print(f"Update {update_idx}/{num_updates}")
                print(f"  Reward Model Loss: {metrics['reward_model_loss']:.4f}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
                print(f"  Mean Shaped Reward: {metrics['mean_shaped_reward']:.4f}")

            # Callback
            if callback is not None:
                callback(update_idx, metrics)

            # Save checkpoint
            if self.config.save_freq > 0 and update_idx % self.config.save_freq == 0:
                self.save(runner_state, update_idx)

        # Aggregate final metrics
        final_metrics = {
            k: np.mean([m[k] for m in all_metrics[-100:]])
            for k in all_metrics[0].keys()
        }

        return runner_state, final_metrics

    def save(
        self,
        runner_state: CFRunnerState,
        step: int,
    ) -> str:
        """
        Save training checkpoint.

        Args:
            runner_state: Current training state
            step: Training step number

        Returns:
            Path to saved checkpoint
        """
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"checkpoint_{step}.pkl")

        checkpoint = {
            'step': step,
            'policy_params': runner_state.policy_state.params,
            'reward_params': runner_state.reward_state.params,
            'config': self.config.__dict__,
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Saved checkpoint to {path}")
        return path

    def load(
        self,
        path: str,
        rng: jax.random.PRNGKey,
    ) -> CFRunnerState:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
            rng: JAX random key

        Returns:
            Restored CFRunnerState
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Initialize state
        runner_state = self.initialize(rng)

        # Restore parameters by creating new TrainStates
        # (TrainState is a frozen dataclass, so we need to replace the whole object)
        policy_state = TrainState.create(
            apply_fn=runner_state.policy_state.apply_fn,
            params=checkpoint['policy_params'],
            tx=runner_state.policy_state.tx,
        )

        reward_state = TrainState.create(
            apply_fn=runner_state.reward_state.apply_fn,
            params=checkpoint['reward_params'],
            tx=runner_state.reward_state.tx,
        )

        # Create new runner state with restored params
        runner_state = CFRunnerState(
            policy_state=policy_state,
            reward_state=reward_state,
            env_state=runner_state.env_state,
            last_obs=runner_state.last_obs,
            global_step=checkpoint.get('step', 0),
            rng=runner_state.rng,
        )

        print(f"Loaded checkpoint from {path} (step {checkpoint['step']})")
        return runner_state


# ============================================================================
# Convenience Functions
# ============================================================================

def create_cf_trainer(
    env_name: str = "coin_game",
    num_agents: int = 3,
    num_envs: int = 8,
    total_timesteps: int = 1_000_000,
    **kwargs,
) -> Tuple[CFTrainer, Any]:
    """
    Create a CF trainer with default configuration.

    Args:
        env_name: Name of environment
        num_agents: Number of agents
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        **kwargs: Additional config overrides

    Returns:
        (trainer, env) tuple
    """
    import socialjax

    # Create environment
    env = socialjax.make(env_name, num_agents=num_agents)

    # Create config
    config = CFConfig(
        env_name=env_name,
        num_agents=num_agents,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        **kwargs,
    )

    # Create trainer
    trainer = CFTrainer(config, env)

    return trainer, env


def train_cf(
    env_name: str = "coin_game",
    num_agents: int = 3,
    num_envs: int = 8,
    total_timesteps: int = 1_000_000,
    num_updates: Optional[int] = None,
    **kwargs,
) -> Tuple[CFRunnerState, Dict[str, Any]]:
    """
    Train CF algorithm with default configuration.

    Args:
        env_name: Name of environment
        num_agents: Number of agents
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        num_updates: Number of updates (optional override)
        **kwargs: Additional config overrides

    Returns:
        (final_state, metrics) tuple
    """
    trainer, env = create_cf_trainer(
        env_name=env_name,
        num_agents=num_agents,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        **kwargs,
    )

    return trainer.train(num_updates=num_updates)


# ============================================================================
# JIT-compiled training step for maximum performance
# ============================================================================

def make_jitted_update_step(trainer: CFTrainer):
    """
    Create a JIT-compiled update step function.

    This provides maximum performance by JIT-compiling the entire update.

    Args:
        trainer: CFTrainer instance

    Returns:
        JIT-compiled update function
    """
    @jax.jit
    def jitted_update(runner_state):
        return trainer._update_step(runner_state, None)

    return jitted_update
