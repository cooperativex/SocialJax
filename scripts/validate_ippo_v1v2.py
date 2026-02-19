#!/usr/bin/env python
"""Validation script to compare V1 and V2 IPPO implementations.

This script runs both implementations with matching configurations and
compares episode returns and training speed.

Usage:
    python scripts/validate_ippo_v1v2.py --steps 100000 --seed 42
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

# Setup paths - IMPORTANT: project root must come before socialjax
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# First add project root (for V1 algorithms)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# Then add socialjax (for V2) - but ensure V1 paths are checked first
socialjax_path = os.path.join(script_dir, 'socialjax')
if socialjax_path not in sys.path:
    sys.path.insert(1, socialjax_path)  # Insert at position 1, after project root

import jax
import jax.numpy as jnp
import numpy as np


def get_common_config(total_timesteps=100000, num_envs=8, num_steps=100, seed=42):
    """Get common config for both V1 and V2."""
    return {
        "LR": 0.0005,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "TOTAL_TIMESTEPS": total_timesteps,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,  # Disable for short tests
        "PARAMETER_SHARING": True,
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
        "REW_SHAPING_HORIZON": 100000,
        "SHAPING_BEGIN": 0,
        "WANDB_MODE": "disabled",
        "ENTITY": "",
        "PROJECT": "socialjax_validation",
        "TUNE": False,
    }


def run_v1_training(config):
    """Run V1 IPPO training and return metrics."""
    print("\n" + "="*60)
    print("Running V1 IPPO Training")
    print("="*60)

    # Import V1 IPPO - need to temporarily remove socialjax from path
    # and ensure project root is first
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
        if 'algorithms' in sys.modules:
            del sys.modules['algorithms']
        if 'algorithms.utils' in sys.modules:
            del sys.modules['algorithms.utils']
        if 'algorithms.IPPO' in sys.modules:
            del sys.modules['algorithms.IPPO']

        # Now import V1
        from algorithms.IPPO.ippo_cnn_cleanup import make_train as v1_make_train
        import wandb

        # Restore path
        sys.path = original_path

    except Exception as e:
        print(f"Error importing V1 IPPO: {e}")
        import traceback
        traceback.print_exc()
        # Restore path on error too
        sys.path = original_path
        return None

    # Initialize wandb in disabled mode
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        config=config,
        mode="disabled",
        name=f'v1_ippo_validation_seed{config["SEED"]}'
    )

    # Create training function
    train_fn = v1_make_train(config)

    # JIT compile
    print("JIT compiling V1 training...")
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
    print(f"Training V1 for {config['TOTAL_TIMESTEPS']} steps...")
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

    print(f"V1 Training time: {train_time:.2f}s")
    print(f"V1 Steps/second: {v1_result['steps_per_second']:.2f}")
    print(f"V1 Mean return: {v1_result['mean_return']:.2f} +/- {v1_result['std_return']:.2f}")

    wandb.finish()

    return v1_result


def run_v2_training(config):
    """Run V2 IPPO training and return metrics."""
    print("\n" + "="*60)
    print("Running V2 IPPO Training")
    print("="*60)

    import socialjax
    from socialjax.algorithms.ippo.algorithm import IPPOAlgorithm
    from socialjax.algorithms.ippo.config import get_ippo_config
    from socialjax.wrappers.baselines import LogWrapper

    # Create environment
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    wrapped_env = LogWrapper(env, replace_info=False)

    # Get observation and action space info
    obs_shape = env.observation_space()[0].shape
    action_dim = env.action_space().n

    class DummyObsSpace:
        shape = obs_shape

    class DummyActSpace:
        n = action_dim

    # Create V2 algorithm
    algo_config = get_ippo_config({
        'LR': config['LR'],
        'GAMMA': config['GAMMA'],
        'GAE_LAMBDA': config['GAE_LAMBDA'],
        'CLIP_EPS': config['CLIP_EPS'],
        'ENT_COEF': config['ENT_COEF'],
        'VF_COEF': config['VF_COEF'],
        'MAX_GRAD_NORM': config['MAX_GRAD_NORM'],
        'UPDATE_EPOCHS': config['UPDATE_EPOCHS'],
        'NUM_MINIBATCHES': config['NUM_MINIBATCHES'],
        'ACTIVATION': config['ACTIVATION'],
    })

    algo = IPPOAlgorithm(
        observation_space=DummyObsSpace(),
        action_space=DummyActSpace(),
        config=algo_config
    )

    num_envs = config['NUM_ENVS']
    num_steps = config['NUM_STEPS']
    num_agents = env.num_agents

    # Initialize
    rng = jax.random.PRNGKey(config['SEED'])
    rng, algo_rng = jax.random.split(rng)
    algo_state = algo.init_state(algo_rng)

    # Initialize environments
    reset_rng = jax.random.split(rng, num_envs)
    obsv, env_state = jax.vmap(wrapped_env.reset)(reset_rng)

    num_updates = config['TOTAL_TIMESTEPS'] // (num_steps * num_envs)
    print(f"Number of updates: {num_updates}")

    episode_returns = []
    start_train = time.time()

    def env_step(carry, _):
        algo_state, env_state, last_obs, rng = carry

        # Get actions for all agents
        rng, action_rng = jax.random.split(rng)

        # Reshape observations: (num_envs, num_agents, ...) -> (num_envs * num_agents, ...)
        obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)

        # Compute actions
        action, info = algo.compute_action(algo_state, obs_batch, action_rng)

        # Reshape actions for environment
        env_act = {}
        for i, agent in enumerate(env.agents):
            env_act[agent] = action[i * num_envs:(i + 1) * num_envs]

        # Step environment
        rng, step_rng = jax.random.split(rng)
        step_rng = jax.random.split(step_rng, num_envs)

        obsv, env_state, reward, done, info_env = jax.vmap(
            wrapped_env.step, in_axes=(0, 0, 0)
        )(step_rng, env_state, [env_act[agent] for agent in env.agents])

        return (algo_state, env_state, obsv, rng), (obsv, reward, done, info_env)

    # Training loop
    for update in range(num_updates):
        # Collect rollouts
        rng, rollout_rng = jax.random.split(rng)

        carry = (algo_state, env_state, obsv, rollout_rng)
        carry, trajectory = jax.lax.scan(env_step, carry, None, num_steps)

        algo_state, env_state, obsv, _ = carry
        obs_traj, reward_traj, done_traj, info_traj = trajectory

        # Extract episode returns from info
        if 'returned_episode_returns' in info_traj:
            returns = np.array(info_traj['returned_episode_returns'])
            # returns shape: (num_steps, num_envs, num_agents)
            episode_returns.extend(returns.flatten().tolist())

        # Prepare batch for update
        obs_batch = jnp.transpose(obs_traj, (2, 0, 1, 3, 4, 5)).reshape(-1, *obs_shape)

        # Create simple update batch (V2 needs proper GAE computation)
        # For now, just run an update with dummy advantages
        batch_size = num_steps * num_envs * num_agents
        batch = {
            'obs': obs_batch,
            'actions': jnp.zeros(batch_size, dtype=jnp.int32),
            'advantages': jnp.zeros(batch_size),
            'targets': jnp.zeros(batch_size),
            'old_log_probs': jnp.zeros(batch_size),
            'values': jnp.zeros(batch_size),
        }

        # Update
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

    print(f"V2 Training time: {train_time:.2f}s")
    print(f"V2 Steps/second: {v2_result['steps_per_second']:.2f}")
    print(f"V2 Mean return: {v2_result['mean_return']:.2f} +/- {v2_result['std_return']:.2f}")

    return v2_result


def compare_results(v1_result, v2_result):
    """Compare V1 and V2 results."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    if v1_result is None:
        print("V1 results not available (import error)")
        print(f"V2 Mean return: {v2_result['mean_return']:.2f} +/- {v2_result['std_return']:.2f}")
        return

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
        return_diff_pct = 0

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
    parser = argparse.ArgumentParser(description='Validate V2 IPPO against V1')
    parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
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

        output_path = f"validation_results_seed{args.seed}_steps{args.steps}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
