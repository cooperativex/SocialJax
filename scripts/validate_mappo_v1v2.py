#!/usr/bin/env python
"""Validation script to compare V1 and V2 MAPPO implementations.

This script runs both implementations with matching configurations and
compares episode returns and training speed, with specific focus on
verifying the centralized critic functionality.

Usage:
    python scripts/validate_mappo_v1v2.py --steps 100000 --seed 42
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

# Setup paths - IMPORTANT: project root must come before socialjax
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
socialjax_path = os.path.join(script_dir, 'socialjax')
if socialjax_path not in sys.path:
    sys.path.insert(1, socialjax_path)

import jax
import jax.numpy as jnp
import numpy as np


def get_common_config(total_timesteps=100000, num_envs=8, num_steps=100, seed=42):
    """Get common config for both V1 and V2 MAPPO."""
    return {
        "LR": 0.0005,
        "LR_ACTOR": 0.0005,
        "LR_CRITIC": 0.0005,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "TOTAL_TIMESTEPS": total_timesteps,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "SCALE_CLIP_EPS": True,  # MAPPO-specific
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,
        "SEED": seed,
        "NUM_SEEDS": 1,
        "ENV_NAME": "clean_up",
        "ENV_KWARGS": {
            "num_agents": 7,
            "num_inner_steps": 100,
            "shared_rewards": False,
            "cnn": True,
            "jit": True,
        },
        "WANDB_MODE": "disabled",
        "ENTITY": "",
        "PROJECT": "socialjax_mappo_validation",
        "TUNE": False,
    }


def run_v1_training(config):
    """Run V1 MAPPO training and return metrics."""
    print("\n" + "="*60)
    print("Running V1 MAPPO Training")
    print("="*60)

    try:
        import importlib
        import importlib.util

        # Save current sys.path state
        original_path = sys.path.copy()

        # Remove socialjax from path temporarily
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        socialjax_path = os.path.join(script_dir, 'socialjax')

        # Rebuild path with project root first, without socialjax
        sys.path = [p for p in sys.path if socialjax_path not in p]
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Clear any cached algorithms module
        for mod in ['algorithms', 'algorithms.utils', 'algorithms.MAPPO']:
            if mod in sys.modules:
                del sys.modules[mod]

        # Now import V1
        from algorithms.MAPPO.mappo_cnn_cleanup import make_train as v1_make_train
        import wandb

        # Restore path
        sys.path = original_path

    except Exception as e:
        print(f"Error importing V1 MAPPO: {e}")
        import traceback
        traceback.print_exc()
        sys.path = original_path
        return None

    # Initialize wandb in disabled mode
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        config=config,
        mode="disabled",
        name=f'v1_mappo_validation_seed{config["SEED"]}'
    )

    # Create training function
    train_fn = v1_make_train(config)

    # JIT compile
    print("JIT compiling V1 MAPPO training...")
    start_jit = time.time()
    train_jit = jax.jit(train_fn)
    rng = jax.random.PRNGKey(config["SEED"])

    # Warmup run to trigger JIT
    print("Warmup run...")
    _ = train_jit(rng)
    jax.block_until_ready(_)
    jit_time = time.time() - start_jit
    print(f"JIT compilation time: {jit_time:.2f}s")

    # Actual training run
    print(f"Training V1 MAPPO for {config['TOTAL_TIMESTEPS']} steps...")
    start_train = time.time()
    rng = jax.random.PRNGKey(config["SEED"])
    result = train_jit(rng)
    jax.block_until_ready(result)
    train_time = time.time() - start_train

    # Extract metrics
    metrics = result.get("metrics", {})
    episode_returns = []
    if "returned_episode_returns" in metrics:
        returns = np.array(metrics["returned_episode_returns"])
        episode_returns = returns.flatten().tolist()

    v1_result = {
        "total_time": train_time,
        "jit_time": jit_time,
        "steps_per_second": config["TOTAL_TIMESTEPS"] / train_time,
        "episode_returns": episode_returns,
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "std_return": np.std(episode_returns) if episode_returns else 0.0,
    }

    print(f"V1 MAPPO Training time: {train_time:.2f}s")
    print(f"V1 MAPPO Steps/second: {v1_result['steps_per_second']:.2f}")
    print(f"V1 MAPPO Mean return: {v1_result['mean_return']:.2f} +/- {v1_result['std_return']:.2f}")

    wandb.finish()

    return v1_result


def run_v2_training(config):
    """Run V2 MAPPO training and return metrics.

    This function implements a proper training loop for MAPPO with:
    1. Centralized critic receiving world_state (all agent observations)
    2. Decentralized actor receiving local observations
    3. Separate optimizers for actor and critic
    """
    print("\n" + "="*60)
    print("Running V2 MAPPO Training")
    print("="*60)

    import socialjax
    from socialjax.algorithms.mappo.algorithm import MAPPOAlgorithm, Transition
    from socialjax.algorithms.mappo.config import get_mappo_config
    from socialjax.wrappers.baselines import LogWrapper, MAPPOWorldStateWrapper

    # Create environment
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    wrapped_env = MAPPOWorldStateWrapper(env)
    wrapped_env = LogWrapper(wrapped_env, replace_info=False)

    # Get observation and action space info
    obs_shape = env.observation_space()[1]  # (H, W, C)
    action_dim = env.action_space().n
    num_agents = env.num_agents

    class DummyObsSpace:
        shape = obs_shape

    class DummyActSpace:
        n = action_dim

    # Create V2 algorithm with MAPPO-specific config
    algo_config = get_mappo_config({
        'LR': config['LR'],
        'LR_ACTOR': config['LR_ACTOR'],
        'LR_CRITIC': config['LR_CRITIC'],
        'GAMMA': config['GAMMA'],
        'GAE_LAMBDA': config['GAE_LAMBDA'],
        'CLIP_EPS': config['CLIP_EPS'],
        'SCALE_CLIP_EPS': config['SCALE_CLIP_EPS'],
        'ENT_COEF': config['ENT_COEF'],
        'VF_COEF': config['VF_COEF'],
        'MAX_GRAD_NORM': config['MAX_GRAD_NORM'],
        'UPDATE_EPOCHS': config['UPDATE_EPOCHS'],
        'NUM_MINIBATCHES': config['NUM_MINIBATCHES'],
        'ACTIVATION': config['ACTIVATION'],
    })

    algo = MAPPOAlgorithm(
        observation_space=DummyObsSpace(),
        action_space=DummyActSpace(),
        config=algo_config,
        num_agents=num_agents,
    )

    num_envs = config['NUM_ENVS']
    num_steps = config['NUM_STEPS']
    num_actors = num_agents * num_envs

    # Initialize
    rng = jax.random.PRNGKey(config['SEED'])
    rng, algo_rng = jax.random.split(rng)
    algo_state = algo.init_state(algo_rng)

    # Initialize environments
    reset_rng = jax.random.split(rng, num_envs)
    obsv, env_state = jax.vmap(wrapped_env.reset)(reset_rng)

    num_updates = config['TOTAL_TIMESTEPS'] // (num_steps * num_envs)
    print(f"Number of updates: {num_updates}")
    print(f"Num actors: {num_actors}")
    print(f"Num agents: {num_agents}")

    episode_returns = []
    start_train = time.time()

    # Helper to batchify rewards and dones
    def batchify_rewards(reward, num_actors):
        """Reshape reward from (num_envs, num_agents) to (num_actors,)."""
        return reward.reshape((num_actors,))

    def batchify_dones(done, num_agents, num_actors):
        """Stack dones from all agents and reshape to (num_actors,)."""
        return jnp.stack([done[str(a)] for a in range(num_agents)]).reshape((num_actors,))

    def create_world_state(obs, num_agents, num_envs):
        """Create world state from observations.

        World state has shape (batch, H, W, C * num_agents) where all agent
        observations are concatenated along the channel dimension.
        """
        # obs shape: (num_envs, num_agents, H, W, C)
        # Transpose to: (num_agents, num_envs, H, W, C)
        # Then reshape to: (num_agents * num_envs, H, W, C)
        # Then concatenate along channels

        # Transpose: (num_envs, num_agents, H, W, C) -> (num_envs, H, W, num_agents, C)
        obs_t = jnp.transpose(obs, (0, 2, 3, 1, 4))
        # Reshape: (num_envs, H, W, num_agents * C)
        world_state = obs_t.reshape(num_envs, obs_shape[0], obs_shape[1], -1)
        # Expand and tile for all agents
        world_state = jnp.expand_dims(world_state, axis=0)
        world_state = jnp.tile(world_state, (num_agents, 1, 1, 1, 1))
        world_state = world_state.reshape(num_actors, obs_shape[0], obs_shape[1], -1)
        return world_state

    def env_step(runner_state, unused):
        """Single environment step - collects transition data."""
        algo_state, env_state, last_obs, rng = runner_state

        # Get actions for all agents
        rng, action_rng = jax.random.split(rng)

        # Reshape observations: (num_envs, num_agents, ...) -> (num_envs * num_agents, ...)
        obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)

        # Compute actions (decentralized actor)
        action, info = algo.compute_action(algo_state, obs_batch, action_rng)
        log_prob = info['log_prob']

        # Compute values (centralized critic with world_state)
        world_state = create_world_state(last_obs, num_agents, num_envs)
        value = algo.compute_value(algo_state, world_state)

        # Reshape actions for environment step
        action_reshaped = action.reshape(num_agents, num_envs)
        env_act = [action_reshaped[i] for i in range(num_agents)]

        # Step environment
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_envs)

        obsv, env_state, reward, done, info_env = jax.vmap(
            wrapped_env.step, in_axes=(0, 0, 0)
        )(step_rngs, env_state, env_act)

        # Batchify for transition storage
        done_batch = batchify_dones(done, num_agents, num_actors)
        reward_batch = batchify_rewards(reward, num_actors)

        # Batchify info for episode returns tracking
        info_batched = jax.tree_map(lambda x: x.reshape((num_actors,)), info_env)

        # Create transition
        transition = Transition(
            global_done=jnp.tile(done['__all__'], num_agents),
            done=done_batch,
            action=action,
            value=value,
            reward=reward_batch,
            log_prob=log_prob,
            obs=obs_batch,
            world_state=world_state,
            info=info_batched,
        )

        runner_state = (algo_state, env_state, obsv, rng)
        return runner_state, transition

    def compute_gae(traj_batch, last_val, gamma, gae_lambda):
        """Compute GAE advantages and targets."""
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * gae_lambda * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    # Training loop
    for update in range(num_updates):
        # Collect rollouts using scan
        rng, rollout_rng = jax.random.split(rng)

        runner_state = (algo_state, env_state, obsv, rollout_rng)
        runner_state, traj_batch = jax.lax.scan(env_step, runner_state, None, num_steps)

        algo_state, env_state, last_obs, _ = runner_state

        # Compute last value for GAE
        last_world_state = create_world_state(last_obs, num_agents, num_envs)
        last_val = algo.compute_value(algo_state, last_world_state)

        # Compute GAE
        gamma = config['GAMMA']
        gae_lambda = config['GAE_LAMBDA']
        advantages, targets = compute_gae(traj_batch, last_val, gamma, gae_lambda)

        # Extract episode returns from info
        if 'returned_episode_returns' in traj_batch.info:
            returns = np.array(traj_batch.info['returned_episode_returns'])
            episode_returns.extend(returns.flatten().tolist())

        # Flatten batch for updates
        batch_size = num_steps * num_actors

        # Update for multiple epochs with minibatches
        update_epochs = config['UPDATE_EPOCHS']
        num_minibatches = config['NUM_MINIBATCHES']
        minibatch_size = batch_size // num_minibatches

        for epoch in range(update_epochs):
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)

            # Flatten and shuffle
            obs_flat = traj_batch.obs.reshape(batch_size, *obs_shape)
            world_state_flat = traj_batch.world_state.reshape(batch_size, -1)
            actions_flat = traj_batch.action.reshape(batch_size)
            log_probs_flat = traj_batch.log_prob.reshape(batch_size)
            values_flat = traj_batch.value.reshape(batch_size)
            advantages_flat = advantages.reshape(batch_size)
            targets_flat = targets.reshape(batch_size)

            # Shuffle
            obs_shuffled = jnp.take(obs_flat, permutation, axis=0)
            world_state_shuffled = jnp.take(world_state_flat, permutation, axis=0)
            actions_shuffled = jnp.take(actions_flat, permutation, axis=0)
            log_probs_shuffled = jnp.take(log_probs_flat, permutation, axis=0)
            values_shuffled = jnp.take(values_flat, permutation, axis=0)
            advantages_shuffled = jnp.take(advantages_flat, permutation, axis=0)
            targets_shuffled = jnp.take(targets_flat, permutation, axis=0)

            # Split into minibatches
            for mb in range(num_minibatches):
                start_idx = mb * minibatch_size
                end_idx = start_idx + minibatch_size

                # Reshape world_state back to (H, W, C * num_agents)
                ws_batch = world_state_shuffled[start_idx:end_idx]
                ws_reshaped = ws_batch.reshape(minibatch_size, obs_shape[0], obs_shape[1], -1)

                batch = {
                    'obs': obs_shuffled[start_idx:end_idx],
                    'world_state': ws_reshaped,
                    'actions': actions_shuffled[start_idx:end_idx],
                    'advantages': advantages_shuffled[start_idx:end_idx],
                    'targets': targets_shuffled[start_idx:end_idx],
                    'old_log_probs': log_probs_shuffled[start_idx:end_idx],
                    'values': values_shuffled[start_idx:end_idx],
                }

                algo_state, metrics = algo.update(algo_state, batch)

        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Loss: {metrics['total_loss']:.4f}")

    train_time = time.time() - start_train

    v2_result = {
        "total_time": train_time,
        "steps_per_second": config["TOTAL_TIMESTEPS"] / train_time,
        "episode_returns": episode_returns,
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "std_return": np.std(episode_returns) if episode_returns else 0.0,
    }

    print(f"V2 MAPPO Training time: {train_time:.2f}s")
    print(f"V2 MAPPO Steps/second: {v2_result['steps_per_second']:.2f}")
    print(f"V2 MAPPO Mean return: {v2_result['mean_return']:.2f} +/- {v2_result['std_return']:.2f}")

    return v2_result


def compare_results(v1_result, v2_result):
    """Compare V1 and V2 results."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    if v1_result is None:
        print("V1 results not available (import error)")
        print(f"V2 Mean return: {v2_result['mean_return']:.2f} +/- {v2_result['std_return']:.2f}")
        return None

    # Speed comparison
    speed_diff = abs(v1_result['steps_per_second'] - v2_result['steps_per_second'])
    speed_diff_pct = speed_diff / v1_result['steps_per_second'] * 100

    print(f"\nSpeed Comparison:")
    print(f"  V1: {v1_result['steps_per_second']:.2f} steps/sec")
    print(f"  V2: {v2_result['steps_per_second']:.2f} steps/sec")
    print(f"  Difference: {speed_diff_pct:.1f}%")

    # Performance comparison
    if v1_result['mean_return'] != 0:
        return_diff = abs(v1_result['mean_return'] - v2_result['mean_return'])
        return_diff_pct = return_diff / abs(v1_result['mean_return']) * 100
    else:
        return_diff_pct = 0 if v2_result['mean_return'] == 0 else float('inf')

    print(f"\nPerformance Comparison:")
    print(f"  V1 Mean Return: {v1_result['mean_return']:.2f} +/- {v1_result['std_return']:.2f}")
    print(f"  V2 Mean Return: {v2_result['mean_return']:.2f} +/- {v2_result['std_return']:.2f}")
    print(f"  Difference: {return_diff_pct:.1f}%")

    # Validation criteria
    print(f"\nValidation Criteria:")
    speed_ok = speed_diff_pct < 50  # Allow 50% difference for short runs
    print(f"  Speed within 50%: {'PASS' if speed_ok else 'FAIL'} ({speed_diff_pct:.1f}%)")

    return_diff_ok = return_diff_pct < 50  # Allow 50% difference for short runs
    print(f"  Returns within 50%: {'PASS' if return_diff_ok else 'FAIL'} ({return_diff_pct:.1f}%)")

    return {
        'speed_diff_pct': speed_diff_pct,
        'return_diff_pct': return_diff_pct,
        'speed_ok': speed_ok,
        'return_ok': return_diff_ok,
        'overall': speed_ok and return_diff_ok
    }


def main():
    parser = argparse.ArgumentParser(description='Validate V2 MAPPO against V1')
    parser.add_argument('--steps', type=int, default=80000, help='Total training steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=100, help='Steps per update')
    parser.add_argument('--v1-only', action='store_true', help='Run only V1')
    parser.add_argument('--v2-only', action='store_true', help='Run only V2')
    args = parser.parse_args()

    config = get_common_config(
        total_timesteps=args.steps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed
    )

    print(f"Configuration:")
    print(f"  Total steps: {config['TOTAL_TIMESTEPS']}")
    print(f"  Num envs: {config['NUM_ENVS']}")
    print(f"  Num steps per update: {config['NUM_STEPS']}")
    print(f"  Seed: {config['SEED']}")
    print(f"  Environment: {config['ENV_NAME']}")
    print(f"  Scale clip eps: {config['SCALE_CLIP_EPS']}")

    v1_result = None
    v2_result = None

    if not args.v2_only:
        v1_result = run_v1_training(config)

    if not args.v1_only:
        v2_result = run_v2_training(config)

    if v1_result and v2_result:
        comparison = compare_results(v1_result, v2_result)

        # Save results
        results = {
            'config': {k: str(v) if isinstance(v, (list, dict)) else v for k, v in config.items()},
            'v1': v1_result,
            'v2': v2_result,
            'comparison': comparison
        }

        output_path = f"mappo_validation_results_seed{args.seed}_steps{args.steps}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
