"""Trainer for SocialJax using vmap parallel environments and jax.lax.scan.

Example:
    >>> from socialjax.training import Trainer, create_trainer
    >>> trainer = create_trainer("ippo", "coin_game")
    >>> state, metrics = trainer.train(total_timesteps=1_000_000_000)
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import time
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import socialjax
from socialjax.wrappers.baselines import LogWrapper
from socialjax.algorithms.registry import get_algorithm
from socialjax.config.manager import ConfigManager, SocialJaxConfig, create_default_config
from socialjax.training.callbacks.base_callback import BaseCallback, CallbackList


# ---------------------------------------------------------------------------
# Transition storage
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


# ---------------------------------------------------------------------------
# Batchify helpers (agent-major ordering)
# ---------------------------------------------------------------------------

def batchify_obs(obs, num_agents, num_envs):
    """(NUM_ENVS, num_agents, H, W, C) -> (NUM_ACTORS, H, W, C)"""
    return jnp.transpose(obs, (1, 0, 2, 3, 4)).reshape(-1, *obs.shape[2:])


def batchify_reward(reward, num_agents):
    """(NUM_ENVS, num_agents) -> (NUM_ACTORS,)"""
    return jnp.transpose(reward).reshape(-1)


def batchify_done(done, num_agents):
    """done dict -> (NUM_ACTORS,)"""
    agent_dones = jnp.stack([done[str(a)] for a in range(num_agents)])
    return agent_dones.reshape(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trains RL algorithms on SocialJax environments using vmap + lax.scan.

    Uses NUM_ENVS parallel environments via jax.vmap and collects rollouts
    via jax.lax.scan for maximum throughput. Supports 1B+ step training.
    """

    def __init__(
        self,
        algorithm: Optional[Union[str, Any]] = None,
        env: Optional[Union[str, Any]] = None,
        config: Optional[Union[Dict[str, Any], SocialJaxConfig]] = None,
        callbacks: Optional[List[Any]] = None,
        seed: int = 42,
        **kwargs,
    ):
        self.seed = seed
        self.callbacks = callbacks or []
        self._callback_list = CallbackList(self.callbacks)

        # Config
        self._socialjax_config = self._init_config(algorithm, env, config, **kwargs)
        self.config = self._socialjax_config.algorithm.training.to_dict()

        # Environment (unwrapped, for observation/action spaces)
        env_cfg = self._socialjax_config.environment
        env_name = env if isinstance(env, str) else env_cfg.name
        env_kwargs = {
            "num_agents": env_cfg.num_agents,
            "num_inner_steps": self.config.get("num_steps", 1000),
            "shared_rewards": False,
            "cnn": True,
            "jit": True,
        }
        self._raw_env = socialjax.make(env_name, **env_kwargs)
        self.env = LogWrapper(self._raw_env)
        self._env_name = env_name
        self.num_agents = self._raw_env.num_agents

        # Algorithm
        obs_space = self.env.observation_space()
        action_space = self.env.action_space()
        if isinstance(obs_space, tuple):
            obs_space = obs_space[0]

        algo_name = algorithm if isinstance(algorithm, str) else self._socialjax_config.algorithm.name
        algo_class = get_algorithm(algo_name)
        self.algorithm = algo_class(
            observation_space=obs_space,
            action_space=action_space,
            config=self._socialjax_config.algorithm.to_dict(),
            num_agents=self.num_agents,
        )
        self._algo_name = algo_name

        # Derived config
        num_envs = self.config.get("num_envs", 256)
        num_steps = self.config.get("num_steps", 1000)
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_actors = self.num_agents * num_envs

        # Metrics (for external access)
        self.metrics = _TrainingMetrics()
        # State reference for save/checkpoint
        self._state = None
        self._runner_state = None
        self._train_state = None

    # ------------------------------------------------------------------
    # Config init (preserved from original)
    # ------------------------------------------------------------------

    def _init_config(self, algorithm, env, config, **kwargs):
        if isinstance(config, SocialJaxConfig):
            return config

        algo_name = algorithm if isinstance(algorithm, str) else None
        env_name = env if isinstance(env, str) else None

        training_keys = {
            "total_timesteps", "num_envs", "num_steps", "update_epochs",
            "num_minibatches", "gamma", "gae_lambda", "learning_rate",
            "clip_eps", "ent_coef", "vf_coef", "max_grad_norm", "seed",
        }
        training_overrides = {k: v for k, v in kwargs.items() if k in training_keys}

        if config is None:
            base_config = create_default_config(
                algorithm=algo_name or "ippo",
                environment=env_name or "coin_game",
            )
        elif isinstance(config, dict):
            if "algorithm" in config and "environment" in config:
                base_config = SocialJaxConfig.from_dict(config)
            else:
                base_config = create_default_config(
                    algorithm=algo_name or "ippo",
                    environment=env_name or "coin_game",
                    **config,
                )
        else:
            base_config = config

        if training_overrides:
            manager = ConfigManager()
            overrides = {"algorithm": {"training": training_overrides}}
            merged = manager._merge_dicts(base_config.to_dict(), overrides)
            base_config = SocialJaxConfig.from_dict(merged)

        return base_config

    # ------------------------------------------------------------------
    # Build JIT-compiled update step
    # ------------------------------------------------------------------

    def _make_update_step(self):
        """Build a JIT-compilable single update step (rollout + GAE + PPO update)."""
        env = self.env
        network = self.algorithm.network
        num_agents = self.num_agents
        num_envs = self.num_envs
        num_steps = self.num_steps
        num_actors = self.num_actors

        gamma = self.config.get("gamma", 0.99)
        gae_lambda = self.config.get("gae_lambda", 0.95)
        clip_eps = self.config.get("clip_eps", 0.2)
        vf_coef = self.config.get("vf_coef", 0.5)
        ent_coef = self.config.get("ent_coef", 0.01)
        update_epochs = self.config.get("update_epochs", 2)
        num_minibatches = self.config.get("num_minibatches", 500)
        minibatch_size = num_actors * num_steps // num_minibatches

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, rng = runner_state

            # --- Collect rollout via scan ---
            def _env_step(carry, unused):
                train_state, env_state, last_obs, last_done, rng = carry

                obs_batch = batchify_obs(last_obs, num_agents, num_envs)

                # Forward pass
                pi, value = network.apply(train_state.params, obs_batch)
                rng, act_rng = jax.random.split(rng)
                action = pi.sample(seed=act_rng)
                log_prob = pi.log_prob(action)

                # Step all envs in parallel
                env_act = [
                    action[a * num_envs:(a + 1) * num_envs]
                    for a in range(num_agents)
                ]
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_act)

                reward_batch = batchify_reward(reward, num_agents)
                done_batch = batchify_done(done, num_agents)
                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_actors,) + x.shape[2:]), info
                )

                transition = Transition(
                    done=last_done,
                    action=action,
                    value=value,
                    reward=reward_batch,
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info,
                )

                carry = (train_state, env_state, obsv, done_batch, rng)
                return carry, transition

            carry, traj_batch = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, last_done, rng),
                None,
                num_steps,
            )
            train_state, env_state, last_obs, last_done, rng = carry

            # --- GAE ---
            last_obs_batch = batchify_obs(last_obs, num_agents, num_envs)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _gae_step(gae_next, transition_data):
                gae, next_value = gae_next
                done, value, reward = transition_data
                delta = reward + gamma * next_value * (1 - done) - value
                gae = delta + gamma * gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _gae_step,
                (jnp.zeros_like(last_val), last_val),
                (traj_batch.done, traj_batch.value, traj_batch.reward),
                reverse=True,
                unroll=1,
            )
            targets = advantages + traj_batch.value

            # --- PPO update ---
            def _update_epoch(update_state, unused):
                def _update_minibatch(ts, batch_info):
                    traj, adv, tgt = batch_info

                    def _loss_fn(params):
                        pi, value = network.apply(params, traj.obs)
                        log_prob = pi.log_prob(traj.action)
                        entropy = pi.entropy().mean()

                        # Value loss (clipped)
                        value_clipped = traj.value + (
                            value - traj.value
                        ).clip(-clip_eps, clip_eps)
                        vf_loss = 0.5 * jnp.maximum(
                            jnp.square(value - tgt),
                            jnp.square(value_clipped - tgt),
                        ).mean()

                        # Policy loss
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * adv_norm,
                            jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm,
                        ).mean()

                        total_loss = loss_actor + vf_coef * vf_loss - ent_coef * entropy
                        return total_loss, (vf_loss, loss_actor, entropy)

                    (total_loss, (vf_loss, actor_loss, entropy)), grads = (
                        jax.value_and_grad(_loss_fn, has_aux=True)(ts.params)
                    )
                    ts = ts.apply_gradients(grads=grads)
                    loss_info = {
                        "total_loss": total_loss,
                        "value_loss": vf_loss,
                        "actor_loss": actor_loss,
                        "entropy": entropy,
                    }
                    return ts, loss_info

                ts, traj, adv, tgt, rng_e = update_state
                rng_e, shuffle_rng = jax.random.split(rng_e)
                batch_size = minibatch_size * num_minibatches
                perm = jax.random.permutation(shuffle_rng, batch_size)

                batch = (traj, adv, tgt)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, perm, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape([num_minibatches, -1] + list(x.shape[1:])),
                    batch,
                )
                ts, loss_info = jax.lax.scan(_update_minibatch, ts, minibatches)
                return (ts, traj, adv, tgt, rng_e), loss_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Aggregate metrics
            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj_batch.info)
            loss_metric = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)

            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return runner_state, (metric, loss_metric)

        return _update_step

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = None,
        rng: Optional[jax.random.PRNGKey] = None,
        start_timestep: int = 0,
    ):
        """Train the algorithm.

        Args:
            total_timesteps: Total timesteps to train (from start)
            rng: Random key
            start_timestep: Starting timestep (for resuming from checkpoint)

        Returns:
            Tuple of (TrainerResult, metrics_dict)
        """
        if total_timesteps is None:
            total_timesteps = self.config.get("total_timesteps", 10_000_000)

        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        num_envs = self.num_envs
        num_steps = self.num_steps
        steps_per_update = num_steps * num_envs
        num_updates = total_timesteps // steps_per_update

        # Calculate starting update step from start_timestep
        start_update = start_timestep // steps_per_update
        remaining_updates = num_updates - start_update

        if remaining_updates <= 0:
            print(f"Already completed {start_timestep:,} steps (target: {total_timesteps:,})")
            return None, {"elapsed": 0, "steps_per_second": 0}

        # LR schedule - account for starting position
        lr = self.config.get("learning_rate", 5e-4)
        anneal_lr = self.config.get("anneal_lr", True)
        num_minibatches = self.config.get("num_minibatches", 500)
        update_epochs = self.config.get("update_epochs", 2)

        if anneal_lr:
            def lr_schedule(count):
                # count is absolute update step, not relative
                frac = 1.0 - (count // (num_minibatches * update_epochs)) / num_updates
                return lr * frac
            tx = optax.chain(
                optax.clip_by_global_norm(self.config.get("max_grad_norm", 0.5)),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config.get("max_grad_norm", 0.5)),
                optax.adam(learning_rate=lr, eps=1e-5),
            )

        # Init network
        network = self.algorithm.network
        obs_space = self.env.observation_space()
        if isinstance(obs_space, tuple):
            obs_space = obs_space[0]
        obs_shape = obs_space.shape

        rng, net_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *obs_shape))

        # Use loaded params if available (for resume)
        if hasattr(self, '_loaded_params') and self._loaded_params is not None:
            params = self._loaded_params
            print(f"Resuming from checkpoint at timestep {start_timestep:,}")
        else:
            params = network.init(net_rng, init_x)

        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        # Init environments (vmap)
        rng, env_rng = jax.random.split(rng)
        reset_keys = jax.random.split(env_rng, num_envs)
        obsv, env_state = jax.vmap(self.env.reset)(reset_keys)

        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros(self.num_actors, dtype=bool),
            rng,
        )

        # Build and JIT compile update step
        _update_step = jax.jit(self._make_update_step())

        # Callbacks
        for cb in self.callbacks:
            if hasattr(cb, "on_training_start"):
                cb.on_training_start(self)

        log_freq = max(1, min(100, remaining_updates // 100))
        checkpoint_freq = self.config.get("checkpoint_freq_updates", max(1, remaining_updates // 20))

        print(f"Total updates:     {remaining_updates:,} (starting from {start_update:,})")
        print(f"Target timesteps:  {total_timesteps:,} (current: {start_timestep:,})")
        print(f"Steps per update:  {steps_per_update:,}")
        print(f"Log every:         {log_freq} updates")
        print("\nJIT compiling (first update may take a few minutes)...\n")

        t0 = time.time()
        steps_this_session = 0

        for relative_update in range(remaining_updates):
            update_step = start_update + relative_update  # Absolute update step
            runner_state, (metric, loss_metric) = _update_step(runner_state, None)
            self._runner_state = runner_state
            self._train_state = runner_state[0]

            steps_this_session += steps_per_update
            current_timestep = start_timestep + steps_this_session

            if (relative_update + 1) % log_freq == 0 or relative_update == 0:
                elapsed = time.time() - t0
                sps = steps_this_session / elapsed if elapsed > 0 else 0

                log_data = {
                    "update_step": update_step + 1,
                    "env_step": current_timestep,
                    "steps_per_second": sps,
                    "elapsed_hours": elapsed / 3600,
                }
                for k, v in metric.items():
                    log_data[k] = float(v)
                for k, v in loss_metric.items():
                    log_data[f"loss/{k}"] = float(v)

                # Console output
                ret_str = ""
                if "returned_episode_returns" in metric:
                    ret_str = f" | return={float(metric['returned_episode_returns']):.3f}"
                print(
                    f"[{update_step+1}/{num_updates}] "
                    f"steps={current_timestep:,} | "
                    f"SPS={sps:.0f} | "
                    f"elapsed={elapsed:.0f}s"
                    f"{ret_str}"
                )

                # Callbacks
                for cb in self.callbacks:
                    if hasattr(cb, "on_step"):
                        cb.on_step(self, update_step + 1, log_data)

        # Done
        t1 = time.time()
        elapsed = t1 - t0

        for cb in self.callbacks:
            if hasattr(cb, "on_training_end"):
                cb.on_training_end(self)

        final_timestep = start_timestep + steps_this_session
        print(f"\nTraining complete: {final_timestep:,} steps in {elapsed:.1f}s ({steps_this_session/elapsed:.0f} SPS)")

        result = _TrainerResult(
            params=runner_state[0].params,
            timestep=final_timestep,
            update_step=start_update + remaining_updates,
        )
        self._state = result
        return result, {"elapsed": elapsed, "steps_per_second": steps_this_session / elapsed}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, timestep: int = 0, update_step: int = 0):
        """Save checkpoint.

        Args:
            path: Directory path to save checkpoint
            timestep: Current timestep (for resume tracking)
            update_step: Current update step (for resume tracking)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._train_state is not None:
            data = {
                "params": self._train_state.params,
                "config": self.config,
                "env_name": self._env_name,
                "algo_name": self._algo_name,
                "timestep": timestep,
                "update_step": update_step,
            }
            with open(path / "checkpoint.pkl", "wb") as f:
                pickle.dump(data, f)

    def load(self, path: str) -> dict:
        """Load checkpoint and return metadata.

        Args:
            path: Directory path containing checkpoint.pkl

        Returns:
            Dict with checkpoint metadata (timestep, update_step, etc.)
        """
        path = Path(path)
        checkpoint_file = path / "checkpoint.pkl"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)

        # Store loaded params for use in train()
        self._loaded_params = data["params"]
        self._loaded_timestep = data.get("timestep", 0)
        self._loaded_update_step = data.get("update_step", 0)

        return {
            "timestep": self._loaded_timestep,
            "update_step": self._loaded_update_step,
            "env_name": data.get("env_name"),
            "algo_name": data.get("algo_name"),
        }


class _TrainerResult:
    """Result container from training."""
    def __init__(self, params, timestep, update_step):
        self.params = params
        self.timestep = timestep
        self.update_step = update_step
        self.episode_count = 0
        self.start_time = 0.0


class _TrainingMetrics:
    """Simple metrics container."""
    def __init__(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.losses = {}
        self.custom_metrics = {}

    def get_summary(self):
        return {}


# ---------------------------------------------------------------------------
# Callback protocol (kept for compatibility)
# ---------------------------------------------------------------------------
Callback = Any


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
RolloutBuffer = None  # No longer used


def create_trainer(
    algorithm: str,
    env: str,
    config: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[Any]] = None,
    seed: int = 42,
    **kwargs,
) -> Trainer:
    """Create a Trainer instance."""
    return Trainer(
        algorithm=algorithm,
        env=env,
        config=config,
        callbacks=callbacks,
        seed=seed,
        **kwargs,
    )
