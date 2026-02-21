#!/usr/bin/env python
"""
Long Training Script: CF Algorithm 1 Billion Steps

This script trains the Counterfactual Regret (CF) algorithm for 1 billion steps
and compares the results with IPPO.

Configuration:
- Total timesteps: 1,000,000,000 (1B)
- Environment: coin_game
- Number of agents: 2 (as per CF-TRAIN-001 spec)
- Alpha: 0.5 (optimal from CF-BENCH-002)
- Number of seeds: 3
- Checkpoint frequency: every 10M steps
- Evaluation frequency: every 5M steps

Usage:
    python scripts/train_cf_1b.py --mode quick   # 1M steps for testing
    python scripts/train_cf_1b.py --mode medium  # 100M steps
    python scripts/train_cf_1b.py --mode full    # 1B steps

Test criteria (CF-TRAIN-001):
- [ ] Training script runs correctly
- [ ] Checkpoints saved periodically
- [ ] Evaluation metrics recorded
- [ ] Final results compared with IPPO
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

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
    """Get training configuration based on mode."""
    if mode == "quick":
        # Quick mode for testing (1M steps) - small batch for stability
        return {
            "total_timesteps": 1_000_000,
            "num_envs": 8,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 10,
            "eval_freq": 100_000,       # Every 100K steps
            "checkpoint_freq": 200_000,  # Every 200K steps
            "num_seeds": 1,              # 1 seed for quick test
        }
    elif mode == "medium":
        # Medium mode (100M steps)
        return {
            "total_timesteps": 100_000_000,
            "num_envs": 16,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 100,
            "eval_freq": 5_000_000,       # Every 5M steps
            "checkpoint_freq": 10_000_000, # Every 10M steps
            "num_seeds": 2,
        }
    else:  # full
        # Full training (1B steps) - conservative batch size for long stability
        return {
            "total_timesteps": 1_000_000_000,
            "num_envs": 32,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 500,
            "eval_freq": 5_000_000,        # Every 5M steps
            "checkpoint_freq": 10_000_000,  # Every 10M steps
            "num_seeds": 3,
        }


# ============================================================================
# CF Long Training
# ============================================================================

def run_cf_long_training(
    config: Dict[str, Any],
    seed: int = 42,
    output_dir: str = "training_results/cf_1b",
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Run CF long training on Coin Game.

    Args:
        config: Training configuration
        seed: Random seed
        output_dir: Directory for checkpoints and results
        use_wandb: Whether to log to WandB

    Returns:
        Training results dictionary
    """
    print("\n" + "="*70)
    print(f"CF Long Training on Coin Game (Seed: {seed})")
    print("="*70)

    from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig

    # Create environment
    num_agents = 2  # As per CF-TRAIN-001 spec
    env = socialjax.make('coin_game', num_agents=num_agents)

    # Calculate number of updates
    num_updates = config['total_timesteps'] // config['num_steps'] // config['num_envs']
    checkpoint_update_freq = max(1, config['checkpoint_freq'] // (config['num_steps'] * config['num_envs']))
    eval_update_freq = max(1, config['eval_freq'] // (config['num_steps'] * config['num_envs']))

    # Create CF config with optimal alpha=0.5
    cf_config = CFConfig(
        env_name='coin_game',
        num_agents=num_agents,
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
        alpha=0.5,  # Optimal alpha from CF-BENCH-002
        use_auto_alpha=False,  # Use fixed alpha
        log_freq=config['log_freq'],
        save_freq=checkpoint_update_freq,  # Use built-in checkpointing
        save_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project="socialjax-cf-1b",
    )

    # Create trainer
    trainer = CFTrainer(cf_config, env)

    # Training metrics collection
    all_metrics = []
    eval_results = []
    checkpoints_saved = []
    global_step = [0]  # Use list to allow mutation in callback

    # WandB setup
    if use_wandb:
        import wandb
        wandb.init(
            project="socialjax-cf-1b",
            name=f"cf_coin_game_seed{seed}",
            config={
                **cf_config.__dict__,
                'seed': seed,
            },
        )

    start_time = time.time()

    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {config['total_timesteps']:,}")
    print(f"  Number of updates: {num_updates}")
    print(f"  Number of envs: {config['num_envs']}")
    print(f"  Steps per update: {config['num_steps']}")
    print(f"  Alpha: {cf_config.alpha}")
    print(f"  Checkpoint frequency: every {config['checkpoint_freq']:,} steps")
    print(f"  Evaluation frequency: every {config['eval_freq']:,} steps")

    print(f"\nStarting training...")

    # Define callback for metrics collection
    def callback(update_idx: int, metrics: Dict[str, Any]):
        global_step[0] = (update_idx + 1) * config['num_steps'] * config['num_envs']

        # Store metrics
        metrics_record = {
            'update': update_idx,
            'global_step': global_step[0],
            'reward_model_loss': float(metrics['reward_model_loss']),
            'policy_loss': float(metrics['policy_loss']),
            'value_loss': float(metrics['value_loss']),
            'mean_reward': float(metrics['mean_reward']),
            'mean_shaped_reward': float(metrics['mean_shaped_reward']),
            'entropy': float(metrics['entropy']),
            'collective_reward': float(metrics['mean_reward'] * num_agents),
        }
        all_metrics.append(metrics_record)

        # Log to console
        if update_idx % config['log_freq'] == 0:
            elapsed = time.time() - start_time
            steps_per_sec = global_step[0] / elapsed if elapsed > 0 else 0
            print(f"  Update {update_idx:,}/{num_updates:,} | "
                  f"Step {global_step[0]:,} | "
                  f"Reward: {metrics['mean_reward']:.4f} | "
                  f"Collective: {metrics_record['collective_reward']:.4f} | "
                  f"Speed: {steps_per_sec:.1f} steps/s")

        # Log to WandB
        if use_wandb:
            wandb.log({
                'reward_model_loss': metrics['reward_model_loss'],
                'policy_loss': metrics['policy_loss'],
                'mean_reward': metrics['mean_reward'],
                'collective_reward': metrics_record['collective_reward'],
                'entropy': metrics['entropy'],
                'global_step': global_step[0],
            })

    # Train using the trainer's built-in train method
    final_state, final_trainer_metrics = trainer.train(callback=callback)

    elapsed = time.time() - start_time
    global_step_val = global_step[0]

    # Compute summary statistics
    final_window = all_metrics[-100:] if len(all_metrics) >= 100 else all_metrics
    final_metrics = {
        'mean_reward': float(np.mean([m['mean_reward'] for m in final_window])),
        'collective_reward': float(np.mean([m['collective_reward'] for m in final_window])),
        'reward_model_loss': float(np.mean([m['reward_model_loss'] for m in final_window])),
        'policy_loss': float(np.mean([m['policy_loss'] for m in final_window])),
        'entropy': float(np.mean([m['entropy'] for m in final_window])),
    }

    # Find saved checkpoints
    if os.path.exists(output_dir):
        checkpoints_saved = [
            {'step': int(f.split('_')[1].split('.')[0]), 'path': os.path.join(output_dir, f)}
            for f in os.listdir(output_dir)
            if f.startswith('checkpoint_') and f.endswith('.pkl')
        ]

    results = {
        'algorithm': 'CF',
        'environment': 'coin_game',
        'num_agents': num_agents,
        'total_timesteps': config['total_timesteps'],
        'training_time_seconds': elapsed,
        'steps_per_second': global_step_val / elapsed if elapsed > 0 else 0,
        'seed': seed,
        'alpha': cf_config.alpha,
        'final_metrics': final_metrics,
        'training_history': all_metrics,
        'eval_results': eval_results,
        'checkpoints_saved': checkpoints_saved,
    }

    print(f"\n{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"  Total Time: {elapsed/3600:.2f} hours")
    print(f"  Total Steps: {global_step_val:,}")
    print(f"  Speed: {results['steps_per_second']:.1f} steps/s")
    print(f"  Final Mean Reward: {final_metrics['mean_reward']:.4f}")
    print(f"  Final Collective Reward: {final_metrics['collective_reward']:.4f}")
    print(f"  Checkpoints: {len(checkpoints_saved)} saved")

    if use_wandb:
        wandb.finish()

    return results


def evaluate_cf(
    runner_state,
    trainer,
    env,
    num_agents: int,
    seed: int,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """
    Evaluate CF policy.

    Args:
        runner_state: Current training state
        trainer: CFTrainer instance
        env: Environment
        num_agents: Number of agents
        seed: Random seed
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    rng = jax.random.PRNGKey(seed + 1000)

    episode_rewards = []
    episode_collective = []

    for _ in range(num_episodes):
        rng, env_rng = jax.random.split(rng)
        obs, env_state = env.reset(env_rng)
        done = False
        episode_reward = np.zeros(num_agents)

        while not done:
            # Get action from policy (deterministic for evaluation)
            obs_batch = obs.reshape(-1, *env.observation_space()[0].shape)
            rng, action_rng = jax.random.split(rng)

            # Get actions
            actions = []
            log_probs = []
            values = []
            for i in range(num_agents):
                pi, value = trainer.policy_network.apply(
                    runner_state.policy_state.params,
                    obs[i][jnp.newaxis]
                )
                action = pi.sample(seed=action_rng)  # Use sample for diversity
                actions.append(action[0])
                values.append(value[0])

            # Step environment
            rng, step_rng = jax.random.split(rng)
            obs, env_state, rewards, dones, infos = env.step(step_rng, env_state, actions)

            episode_reward += np.array(rewards)
            done = dones["__all__"] if isinstance(dones, dict) else dones

        episode_rewards.append(episode_reward.mean())
        episode_collective.append(episode_reward.sum())

    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'collective_reward': float(np.mean(episode_collective)),
        'std_reward': float(np.std(episode_rewards)),
        'num_episodes': num_episodes,
    }


# ============================================================================
# IPPO Long Training (for comparison)
# ============================================================================

def run_ippo_long_training(
    config: Dict[str, Any],
    seed: int = 42,
    output_dir: str = "training_results/ippo_1b",
) -> Dict[str, Any]:
    """
    Run IPPO long training on Coin Game for comparison.

    Args:
        config: Training configuration
        seed: Random seed
        output_dir: Directory for results

    Returns:
        Training results dictionary
    """
    print("\n" + "="*70)
    print(f"IPPO Long Training on Coin Game (Seed: {seed})")
    print("="*70)

    from socialjax.algorithms.cf.policy import (
        ActorCritic,
        Transition,
        compute_gae,
        ppo_update_epoch,
    )
    from flax.training.train_state import TrainState
    import optax

    # Create environment
    num_agents = 2
    env = socialjax.make('coin_game', num_agents=num_agents)
    action_dim = env.action_space().n
    obs_shape = env.observation_space()[0].shape

    # Calculate number of updates
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
    global_step = 0

    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {config['total_timesteps']:,}")
    print(f"  Number of updates: {num_updates}")
    print(f"  Number of envs: {config['num_envs']}")

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

        rng, action_rng = jax.random.split(rng)
        obs_batch = last_obs.reshape(-1, *obs_shape)
        actions, log_probs, values = get_actions(ts.params, obs_batch, action_rng)

        actions = actions.reshape(config['num_envs'], num_agents)
        log_probs = log_probs.reshape(config['num_envs'], num_agents)
        values = values.reshape(config['num_envs'], num_agents)

        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, config['num_envs'])
        env_actions = [actions[:, i] for i in range(num_agents)]

        obsv, env_state_next, rewards, dones, infos = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(step_rngs, es, env_actions)

        transition = Transition(
            obs=last_obs,
            action=actions,
            reward=rewards,
            done=dones["__all__"] if isinstance(dones, dict) else dones,
            log_prob=log_probs,
            value=values,
        )

        new_carry = (ts, env_state_next, obsv, rng)
        return new_carry, transition

    for update_idx in range(num_updates):
        rng, step_rng = jax.random.split(rng)
        carry = (train_state, env_state, obs, step_rng)

        carry, traj_batch = jax.lax.scan(
            env_step_fn,
            carry,
            None,
            length=config['num_steps'],
        )

        train_state, env_state, obs, _ = carry
        global_step += config['num_steps'] * config['num_envs']

        # Compute GAE
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
                0.99,
                0.95,
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
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0
            mean_reward = float(traj_batch.reward.mean())
            collective_reward = mean_reward * num_agents

            metrics_record = {
                'update': update_idx,
                'global_step': global_step,
                'mean_reward': mean_reward,
                'collective_reward': collective_reward,
                'policy_loss': float(update_metrics['total_loss'].mean()),
                'entropy': float(update_metrics['entropy'].mean()),
            }
            all_metrics.append(metrics_record)

            print(f"  Update {update_idx:,}/{num_updates:,} | "
                  f"Step {global_step:,} | "
                  f"Reward: {mean_reward:.4f} | "
                  f"Collective: {collective_reward:.4f} | "
                  f"Speed: {steps_per_sec:.1f} steps/s")

    elapsed = time.time() - start_time

    # Compute summary statistics
    final_window = all_metrics[-100:] if len(all_metrics) >= 100 else all_metrics
    final_metrics = {
        'mean_reward': float(np.mean([m['mean_reward'] for m in final_window])),
        'collective_reward': float(np.mean([m['collective_reward'] for m in final_window])),
        'policy_loss': float(np.mean([m['policy_loss'] for m in final_window])),
        'entropy': float(np.mean([m['entropy'] for m in final_window])),
    }

    results = {
        'algorithm': 'IPPO',
        'environment': 'coin_game',
        'num_agents': num_agents,
        'total_timesteps': config['total_timesteps'],
        'training_time_seconds': elapsed,
        'steps_per_second': global_step / elapsed if elapsed > 0 else 0,
        'seed': seed,
        'final_metrics': final_metrics,
        'training_history': all_metrics,
    }

    print(f"\n{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"  Total Time: {elapsed/3600:.2f} hours")
    print(f"  Total Steps: {global_step:,}")
    print(f"  Speed: {results['steps_per_second']:.1f} steps/s")
    print(f"  Final Mean Reward: {final_metrics['mean_reward']:.4f}")
    print(f"  Final Collective Reward: {final_metrics['collective_reward']:.4f}")

    return results


# ============================================================================
# Results Comparison
# ============================================================================

def compare_results(cf_results: List[Dict], ippo_results: List[Dict]) -> Dict[str, Any]:
    """
    Compare CF and IPPO results across multiple seeds.

    Args:
        cf_results: List of CF results (one per seed)
        ippo_results: List of IPPO results (one per seed)

    Returns:
        Comparison results
    """
    print("\n" + "="*70)
    print("Comparison: CF vs IPPO (Multi-seed)")
    print("="*70)

    # Aggregate across seeds
    cf_collectives = [r['final_metrics']['collective_reward'] for r in cf_results]
    ippo_collectives = [r['final_metrics']['collective_reward'] for r in ippo_results]

    cf_means = [r['final_metrics']['mean_reward'] for r in cf_results]
    ippo_means = [r['final_metrics']['mean_reward'] for r in ippo_results]

    cf_mean_avg = np.mean(cf_means)
    cf_mean_std = np.std(cf_means)
    ippo_mean_avg = np.mean(ippo_means)
    ippo_mean_std = np.std(ippo_means)

    cf_coll_avg = np.mean(cf_collectives)
    cf_coll_std = np.std(cf_collectives)
    ippo_coll_avg = np.mean(ippo_collectives)
    ippo_coll_std = np.std(ippo_collectives)

    improvement_pct = ((cf_coll_avg - ippo_coll_avg) / abs(ippo_coll_avg) * 100
                       if ippo_coll_avg != 0 else 0)

    print(f"\n  Final Mean Reward (per agent):")
    print(f"    CF:   {cf_mean_avg:.4f} +/- {cf_mean_std:.4f}")
    print(f"    IPPO: {ippo_mean_avg:.4f} +/- {ippo_mean_std:.4f}")
    print(f"    Diff: {cf_mean_avg - ippo_mean_avg:.4f}")

    print(f"\n  Final Collective Reward (all agents):")
    print(f"    CF:   {cf_coll_avg:.4f} +/- {cf_coll_std:.4f}")
    print(f"    IPPO: {ippo_coll_avg:.4f} +/- {ippo_coll_std:.4f}")
    print(f"    Diff: {cf_coll_avg - ippo_coll_avg:.4f} ({improvement_pct:+.1f}%)")

    print(f"\n  Training Speed:")
    cf_speeds = [r['steps_per_second'] for r in cf_results]
    ippo_speeds = [r['steps_per_second'] for r in ippo_results]
    print(f"    CF:   {np.mean(cf_speeds):.1f} steps/s")
    print(f"    IPPO: {np.mean(ippo_speeds):.1f} steps/s")

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'num_seeds': len(cf_results),
        'cf': {
            'mean_reward_avg': cf_mean_avg,
            'mean_reward_std': cf_mean_std,
            'collective_reward_avg': cf_coll_avg,
            'collective_reward_std': cf_coll_std,
            'results': cf_results,
        },
        'ippo': {
            'mean_reward_avg': ippo_mean_avg,
            'mean_reward_std': ippo_mean_std,
            'collective_reward_avg': ippo_coll_avg,
            'collective_reward_std': ippo_coll_std,
            'results': ippo_results,
        },
        'comparison': {
            'collective_reward_diff': cf_coll_avg - ippo_coll_avg,
            'improvement_percent': improvement_pct,
            'cf_better': cf_coll_avg > ippo_coll_avg,
            'statistically_significant': abs(cf_coll_avg - ippo_coll_avg) > (cf_coll_std + ippo_coll_std),
        }
    }

    return comparison


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CF 1B Step Training")
    parser.add_argument(
        '--mode',
        choices=['quick', 'medium', 'full'],
        default='quick',
        help='Training mode: quick (1M), medium (100M), full (1B steps)'
    )
    parser.add_argument(
        '--output',
        default='training_results/cf_1b',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (first seed)'
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=None,
        help='Number of seeds (overrides config)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable WandB logging'
    )
    parser.add_argument(
        '--cf-only',
        action='store_true',
        help='Only run CF training (skip IPPO)'
    )
    parser.add_argument(
        '--ippo-only',
        action='store_true',
        help='Only run IPPO training (skip CF)'
    )

    args = parser.parse_args()

    print("="*70)
    print("CF Algorithm 1 Billion Step Training")
    print(f"Mode: {args.mode}")
    print("="*70)

    # Get config
    config = get_config(args.mode)

    # Override num_seeds if specified
    if args.num_seeds is not None:
        config['num_seeds'] = args.num_seeds

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Verify JAX
    print(f"\nJAX devices: {jax.devices()}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run training
    cf_results = []
    ippo_results = []

    if not args.ippo_only:
        # Run CF training for all seeds
        for seed_idx in range(config['num_seeds']):
            seed = args.seed + seed_idx
            cf_result = run_cf_long_training(
                config,
                seed=seed,
                output_dir=os.path.join(args.output, f'cf_seed{seed}'),
                use_wandb=args.wandb,
            )
            cf_results.append(cf_result)

            # Save individual result
            result_file = os.path.join(args.output, f'cf_seed{seed}_results.json')
            with open(result_file, 'w') as f:
                # Remove training history for smaller file
                cf_result_save = {k: v for k, v in cf_result.items() if k != 'training_history'}
                json.dump(cf_result_save, f, indent=2, default=str)

    if not args.cf_only:
        # Run IPPO training for all seeds
        for seed_idx in range(config['num_seeds']):
            seed = args.seed + seed_idx + 100  # Different seed offset
            ippo_result = run_ippo_long_training(
                config,
                seed=seed,
                output_dir=os.path.join(args.output, f'ippo_seed{seed}'),
            )
            ippo_results.append(ippo_result)

            # Save individual result
            result_file = os.path.join(args.output, f'ippo_seed{seed}_results.json')
            with open(result_file, 'w') as f:
                ippo_result_save = {k: v for k, v in ippo_result.items() if k != 'training_history'}
                json.dump(ippo_result_save, f, indent=2, default=str)

    # Compare results if both were run
    if cf_results and ippo_results:
        comparison = compare_results(cf_results, ippo_results)

        # Save comparison
        comparison_file = os.path.join(
            args.output,
            f"comparison_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(comparison_file, 'w') as f:
            # Save without full results for readability
            comparison_save = {
                'timestamp': comparison['timestamp'],
                'num_seeds': comparison['num_seeds'],
                'cf': comparison['cf'],
                'ippo': comparison['ippo'],
                'comparison': comparison['comparison'],
            }
            # Remove nested results
            comparison_save['cf'].pop('results', None)
            comparison_save['ippo'].pop('results', None)
            json.dump(comparison_save, f, indent=2, default=str)

        print(f"\nComparison saved to: {comparison_file}")

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    if cf_results:
        print(f"\nCF Results ({len(cf_results)} seeds):")
        print(f"  Collective Reward: {np.mean([r['final_metrics']['collective_reward'] for r in cf_results]):.4f}")

    if ippo_results:
        print(f"\nIPPO Results ({len(ippo_results)} seeds):")
        print(f"  Collective Reward: {np.mean([r['final_metrics']['collective_reward'] for r in ippo_results]):.4f}")

    # Test criteria
    print("\nTest Criteria (CF-TRAIN-001):")
    print(f"  [{'x' if cf_results else ' '}] Training script runs correctly")
    print(f"  [{'x' if any(r['checkpoints_saved'] for r in cf_results) else ' '}] Checkpoints saved periodically")
    print(f"  [{'x' if any(r['eval_results'] for r in cf_results) else ' '}] Evaluation metrics recorded")
    print(f"  [{'x' if cf_results and ippo_results else ' '}] Final results compared with IPPO")


if __name__ == "__main__":
    main()
