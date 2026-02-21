#!/usr/bin/env python
"""
Benchmark: CF vs IPPO on Coin Game

This script compares the performance of Counterfactual Regret (CF) vs
Independent PPO (IPPO) on the Coin Game environment.

Usage:
    python scripts/benchmark_cf_vs_ippo.py --mode quick   # 10K steps for testing
    python scripts/benchmark_cf_vs_ippo.py --mode full    # 1M steps for full benchmark

Test criteria (CF-BENCH-001):
- [x] Both algorithms complete training
- [x] Results are logged
- [x] Comparison metrics are available
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'socialjax'))

import jax
import jax.numpy as jnp
import numpy as np

# Import SocialJax
import socialjax


# ============================================================================
# Configuration
# ============================================================================

def get_config(mode: str = "quick") -> Dict[str, Any]:
    """Get benchmark configuration based on mode."""
    if mode == "quick":
        # Quick mode for testing (10K steps)
        return {
            "total_timesteps": 10_000,
            "num_envs": 8,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 5,
            "num_agents": 3,
        }
    elif mode == "medium":
        # Medium mode (100K steps)
        return {
            "total_timesteps": 100_000,
            "num_envs": 16,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 10,
            "num_agents": 3,
        }
    else:  # full
        # Full benchmark (1M steps)
        return {
            "total_timesteps": 1_000_000,
            "num_envs": 32,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 50,
            "num_agents": 3,
        }


# ============================================================================
# CF Training
# ============================================================================

def run_cf_training(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Run CF training on Coin Game."""
    print("\n" + "="*60)
    print("CF Training on Coin Game")
    print("="*60)

    from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig

    # Create environment
    env = socialjax.make('coin_game', num_agents=config['num_agents'])

    # Create CF config
    cf_config = CFConfig(
        env_name='coin_game',
        num_agents=config['num_agents'],
        num_envs=config['num_envs'],
        total_timesteps=config['total_timesteps'],
        num_steps=config['num_steps'],
        update_epochs=config['update_epochs'],
        num_minibatches=config['num_minibatches'],
        policy_lr=0.0003,
        reward_lr=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        use_auto_alpha=True,  # alpha = N-1
        log_freq=config['log_freq'],
        save_freq=0,  # No checkpointing during benchmark
    )

    # Create trainer
    trainer = CFTrainer(cf_config, env)

    # Training metrics collection
    all_metrics = []

    def callback(update_idx: int, metrics: Dict[str, Any]):
        all_metrics.append({
            'update': update_idx,
            'reward_model_loss': float(metrics['reward_model_loss']),
            'policy_loss': float(metrics['policy_loss']),
            'value_loss': float(metrics['value_loss']),
            'mean_reward': float(metrics['mean_reward']),
            'mean_shaped_reward': float(metrics['mean_shaped_reward']),
            'entropy': float(metrics['entropy']),
        })

    # Train
    start_time = time.time()
    final_state, final_metrics = trainer.train(callback=callback)
    elapsed = time.time() - start_time

    # Compute summary statistics
    if all_metrics:
        mean_rewards = [m['mean_reward'] for m in all_metrics[-20:]]
        mean_shaped = [m['mean_shaped_reward'] for m in all_metrics[-20:]]

        # Calculate collective reward (sum of all agents' rewards)
        collective_rewards = [m['mean_reward'] * config['num_agents'] for m in all_metrics[-20:]]
    else:
        mean_rewards = [0]
        mean_shaped = [0]
        collective_rewards = [0]

    results = {
        'algorithm': 'CF',
        'environment': 'coin_game',
        'num_agents': config['num_agents'],
        'total_timesteps': config['total_timesteps'],
        'training_time_seconds': elapsed,
        'steps_per_second': config['total_timesteps'] / elapsed if elapsed > 0 else 0,
        'final_metrics': {
            'mean_reward': float(np.mean(mean_rewards)),
            'mean_shaped_reward': float(np.mean(mean_shaped)),
            'collective_reward': float(np.mean(collective_rewards)),
            'reward_model_loss': float(final_metrics.get('reward_model_loss', 0)),
            'policy_loss': float(final_metrics.get('policy_loss', 0)),
            'entropy': float(final_metrics.get('entropy', 0)),
        },
        'training_history': all_metrics,
    }

    print(f"\nCF Training completed in {elapsed:.2f}s")
    print(f"  Final Mean Reward: {results['final_metrics']['mean_reward']:.4f}")
    print(f"  Final Collective Reward: {results['final_metrics']['collective_reward']:.4f}")

    return results


# ============================================================================
# IPPO Training (Simplified)
# ============================================================================

def run_ippo_training(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Run IPPO training on Coin Game using CF infrastructure."""
    print("\n" + "="*60)
    print("IPPO Training on Coin Game")
    print("="*60)

    # For fair comparison, we use the same network architecture as CF
    # but without the counterfactual reward shaping
    from socialjax.algorithms.cf.policy import (
        ActorCritic,
        Transition,
        compute_gae,
        compute_ppo_loss,
        create_actor_critic_train_state,
        ppo_update_epoch,
    )
    from flax.training.train_state import TrainState
    import optax

    # Create environment
    env = socialjax.make('coin_game', num_agents=config['num_agents'])
    num_agents = env.num_agents
    action_dim = env.action_space().n
    obs_shape = env.observation_space()[0].shape

    # Calculate number of updates
    num_actors = num_agents * config['num_envs']
    num_updates = config['total_timesteps'] // config['num_steps'] // config['num_envs']

    # Create network
    network = ActorCritic(
        action_dim=action_dim,
        cnn_features=(32, 32, 32),
        cnn_kernels=((5, 5), (3, 3), (3, 3)),
        hidden_dim=64,
        activation="relu",
    )

    # Initialize
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    sample_obs = jnp.zeros((1, *obs_shape))
    network_params = network.init(init_rng, sample_obs)

    # Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(0.0003, eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # Initialize environments
    env_rng = jax.random.split(env_rng, config['num_envs'])
    obs, env_state = jax.vmap(env.reset)(env_rng)

    # Training loop
    all_metrics = []

    start_time = time.time()

    @jax.jit
    def get_actions(policy_params, obs_batch, action_rng):
        """Get actions for all agents."""
        pi, value = network.apply(policy_params, obs_batch)
        action = pi.sample(seed=action_rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value

    @jax.jit
    def env_step_fn(carry, _):
        """Single environment step."""
        ts, es, last_obs, rng = carry

        # Get actions
        rng, action_rng = jax.random.split(rng)
        obs_batch = last_obs.reshape(-1, *obs_shape)
        actions, log_probs, values = get_actions(ts.params, obs_batch, action_rng)

        # Reshape actions
        actions = actions.reshape(config['num_envs'], num_agents)
        log_probs = log_probs.reshape(config['num_envs'], num_agents)
        values = values.reshape(config['num_envs'], num_agents)

        # Step environment
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, config['num_envs'])
        env_actions = [actions[:, i] for i in range(num_agents)]

        obsv, env_state_next, rewards, dones, infos = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(step_rngs, es, env_actions)

        transition = Transition(
            obs=last_obs,
            action=actions,
            reward=rewards,  # Use raw rewards (no shaping)
            done=dones["__all__"] if isinstance(dones, dict) else dones,
            log_prob=log_probs,
            value=values,
        )

        new_carry = (ts, env_state_next, obsv, rng)
        return new_carry, transition

    for update_idx in range(num_updates):
        # Collect trajectory
        rng, step_rng = jax.random.split(rng)
        carry = (train_state, env_state, obs, step_rng)

        carry, traj_batch = jax.lax.scan(
            env_step_fn,
            carry,
            None,
            length=config['num_steps'],
        )

        train_state, env_state, obs, _ = carry

        # Compute GAE (using raw rewards, no shaping)
        last_obs_batch = obs.reshape(-1, *obs_shape)
        _, last_value = jax.vmap(
            lambda o: network.apply(train_state.params, o[jnp.newaxis])
        )(last_obs_batch)
        last_value = last_value.reshape(config['num_envs'], num_agents)

        def compute_gae_for_agent(agent_id):
            agent_traj = Transition(
                obs=traj_batch.obs[:, :, agent_id],
                action=traj_batch.action[:, :, agent_id],
                reward=traj_batch.reward[:, :, agent_id],
                done=traj_batch.done,
                log_prob=traj_batch.log_prob[:, :, agent_id],
                value=traj_batch.value[:, :, agent_id],
            )
            advantages, targets = compute_gae(
                agent_traj,
                last_value[:, agent_id],
                0.99,  # gamma
                0.95,  # gae_lambda
            )
            return advantages, targets

        advantages_all, targets_all = jax.vmap(compute_gae_for_agent)(
            jnp.arange(num_agents)
        )
        advantages_all = advantages_all.transpose(1, 2, 0)
        targets_all = targets_all.transpose(1, 2, 0)

        # PPO update
        rng, update_rng = jax.random.split(rng)
        train_state, rng, update_metrics = ppo_update_epoch(
            train_state,
            traj_batch,
            advantages_all,
            targets_all,
            update_rng,
            num_minibatches=config['num_minibatches'],
            update_epochs=config['update_epochs'],
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )

        # Log
        if update_idx % config['log_freq'] == 0:
            mean_reward = float(traj_batch.reward.mean())
            collective_reward = mean_reward * num_agents
            metrics = {
                'update': update_idx,
                'mean_reward': mean_reward,
                'collective_reward': collective_reward,
                'policy_loss': float(update_metrics['total_loss'].mean()),
                'value_loss': float(update_metrics['value_loss'].mean()),
                'entropy': float(update_metrics['entropy'].mean()),
            }
            all_metrics.append(metrics)
            print(f"  Update {update_idx}/{num_updates}: "
                  f"reward={mean_reward:.4f}, collective={collective_reward:.4f}")

    elapsed = time.time() - start_time

    # Compute summary statistics
    if all_metrics:
        mean_rewards = [m['mean_reward'] for m in all_metrics[-20:]]
        collective_rewards = [m['collective_reward'] for m in all_metrics[-20:]]
    else:
        mean_rewards = [0]
        collective_rewards = [0]

    results = {
        'algorithm': 'IPPO',
        'environment': 'coin_game',
        'num_agents': config['num_agents'],
        'total_timesteps': config['total_timesteps'],
        'training_time_seconds': elapsed,
        'steps_per_second': config['total_timesteps'] / elapsed if elapsed > 0 else 0,
        'final_metrics': {
            'mean_reward': float(np.mean(mean_rewards)),
            'collective_reward': float(np.mean(collective_rewards)),
            'policy_loss': float(all_metrics[-1]['policy_loss']) if all_metrics else 0,
            'entropy': float(all_metrics[-1]['entropy']) if all_metrics else 0,
        },
        'training_history': all_metrics,
    }

    print(f"\nIPPO Training completed in {elapsed:.2f}s")
    print(f"  Final Mean Reward: {results['final_metrics']['mean_reward']:.4f}")
    print(f"  Final Collective Reward: {results['final_metrics']['collective_reward']:.4f}")

    return results


# ============================================================================
# Comparison
# ============================================================================

def compare_results(cf_results: Dict, ippo_results: Dict) -> Dict[str, Any]:
    """Compare CF and IPPO results."""
    print("\n" + "="*60)
    print("Comparison: CF vs IPPO on Coin Game")
    print("="*60)

    cf_collective = cf_results['final_metrics']['collective_reward']
    ippo_collective = ippo_results['final_metrics']['collective_reward']

    cf_mean = cf_results['final_metrics']['mean_reward']
    ippo_mean = ippo_results['final_metrics']['mean_reward']

    improvement_pct = ((cf_collective - ippo_collective) / abs(ippo_collective) * 100
                       if ippo_collective != 0 else 0)

    print(f"\n  Training Time:")
    print(f"    CF:   {cf_results['training_time_seconds']:.2f}s ({cf_results['steps_per_second']:.1f} steps/s)")
    print(f"    IPPO: {ippo_results['training_time_seconds']:.2f}s ({ippo_results['steps_per_second']:.1f} steps/s)")

    print(f"\n  Final Mean Reward (per agent):")
    print(f"    CF:   {cf_mean:.4f}")
    print(f"    IPPO: {ippo_mean:.4f}")
    print(f"    Diff: {cf_mean - ippo_mean:.4f}")

    print(f"\n  Final Collective Reward (all agents):")
    print(f"    CF:   {cf_collective:.4f}")
    print(f"    IPPO: {ippo_collective:.4f}")
    print(f"    Diff: {cf_collective - ippo_collective:.4f} ({improvement_pct:+.1f}%)")

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'cf': cf_results,
        'ippo': ippo_results,
        'comparison': {
            'collective_reward_cf': cf_collective,
            'collective_reward_ippo': ippo_collective,
            'collective_reward_diff': cf_collective - ippo_collective,
            'improvement_percent': improvement_pct,
            'cf_better': cf_collective > ippo_collective,
        }
    }

    return comparison


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark CF vs IPPO on Coin Game")
    parser.add_argument(
        '--mode',
        choices=['quick', 'medium', 'full'],
        default='quick',
        help='Benchmark mode: quick (10K), medium (100K), full (1M steps)'
    )
    parser.add_argument(
        '--output',
        default='evaluation/cf_vs_ippo',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    print("="*60)
    print(f"CF vs IPPO Benchmark on Coin Game")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print("="*60)

    # Get config
    config = get_config(args.mode)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Verify JAX
    print(f"\nJAX devices: {jax.devices()}")

    # Run CF training
    cf_results = run_cf_training(config, args.seed)

    # Run IPPO training
    ippo_results = run_ippo_training(config, args.seed + 1)

    # Compare results
    comparison = compare_results(cf_results, ippo_results)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(
        args.output,
        f"benchmark_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CF Collective Reward:   {comparison['comparison']['collective_reward_cf']:.4f}")
    print(f"IPPO Collective Reward: {comparison['comparison']['collective_reward_ippo']:.4f}")
    print(f"Improvement:            {comparison['comparison']['improvement_percent']:+.1f}%")

    if comparison['comparison']['cf_better']:
        print("\nCF outperforms IPPO on collective reward!")
    else:
        print("\nIPPO outperforms CF on collective reward.")

    # Test criteria
    print("\nTest Criteria (CF-BENCH-001):")
    print(f"  [x] Both algorithms completed training")
    print(f"  [x] Results logged to {output_file}")
    print(f"  [x] Comparison metrics available")

    return comparison


if __name__ == "__main__":
    main()
