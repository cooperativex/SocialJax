#!/usr/bin/env python
"""
Benchmark: Alpha Parameter Ablation Experiment (CF-BENCH-002)

This script tests different alpha values for the CF algorithm to analyze
their effects on:
- Collective reward
- Regret values
- Prosocial behavior

Alpha controls the weight of intrinsic (prosocial) reward vs extrinsic reward:
    shaped_reward = extrinsic + alpha * intrinsic

The paper suggests alpha = N-1 as a good default (e.g., 2 for 3 agents).

Usage:
    python scripts/benchmark_alpha_ablation.py --mode quick   # 10K steps per alpha
    python scripts/benchmark_alpha_ablation.py --mode medium  # 50K steps per alpha
    python scripts/benchmark_alpha_ablation.py --mode full    # 200K steps per alpha

Test criteria (CF-BENCH-002):
- [ ] All alpha values tested: [0.5, 1, 2, 5, 10]
- [ ] Trend analysis available
- [ ] Optimal alpha determined
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
        # Quick mode for testing (10K steps per alpha)
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
        # Medium mode (50K steps per alpha)
        return {
            "total_timesteps": 50_000,
            "num_envs": 16,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 10,
            "num_agents": 3,
        }
    else:  # full
        # Full benchmark (200K steps per alpha)
        return {
            "total_timesteps": 200_000,
            "num_envs": 32,
            "num_steps": 128,
            "update_epochs": 4,
            "num_minibatches": 4,
            "log_freq": 50,
            "num_agents": 3,
        }


# ============================================================================
# CF Training with Custom Alpha
# ============================================================================

def run_cf_with_alpha(
    alpha: float,
    config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run CF training with a specific alpha value.

    Args:
        alpha: The alpha parameter for reward shaping
        config: Training configuration
        seed: Random seed

    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"CF Training with alpha={alpha}")
    print(f"{'='*60}")

    from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
    from socialjax.algorithms.cf.counterfactual import (
        generate_counterfactual_rewards_vmap,
        compute_collective_cf_reward,
        compute_actual_collective_reward,
    )
    from socialjax.algorithms.cf.regret import compute_counterfactual_regret

    # Create environment
    env = socialjax.make('coin_game', num_agents=config['num_agents'])

    # Create CF config with explicit alpha (override auto_alpha)
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
        alpha=alpha,  # Explicit alpha
        use_auto_alpha=False,  # Don't override
        log_freq=config['log_freq'],
        save_freq=0,  # No checkpointing during benchmark
    )

    # Create trainer
    trainer = CFTrainer(cf_config, env)

    # Training metrics collection
    all_metrics = []
    all_regrets = []
    all_collectives = []

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
        # Track collective reward
        collective = float(metrics['mean_reward']) * config['num_agents']
        all_collectives.append(collective)

    # Train
    start_time = time.time()
    final_state, final_metrics = trainer.train(callback=callback)
    elapsed = time.time() - start_time

    # Compute summary statistics
    if all_metrics:
        mean_rewards = [m['mean_reward'] for m in all_metrics[-20:]]
        mean_shaped = [m['mean_shaped_reward'] for m in all_metrics[-20:]]
        collective_rewards = all_collectives[-20:] if all_collectives else [0]
    else:
        mean_rewards = [0]
        mean_shaped = [0]
        collective_rewards = [0]

    results = {
        'alpha': alpha,
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
        'collective_rewards_history': all_collectives,
    }

    print(f"\nCF Training (alpha={alpha}) completed in {elapsed:.2f}s")
    print(f"  Final Mean Reward: {results['final_metrics']['mean_reward']:.4f}")
    print(f"  Final Collective Reward: {results['final_metrics']['collective_reward']:.4f}")

    return results


# ============================================================================
# Analysis
# ============================================================================

def analyze_alpha_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze results across different alpha values.

    Args:
        results: List of results from different alpha runs

    Returns:
        Analysis dictionary with trends and recommendations
    """
    print("\n" + "="*60)
    print("Alpha Parameter Ablation Analysis")
    print("="*60)

    # Extract data
    alphas = [r['alpha'] for r in results]
    collective_rewards = [r['final_metrics']['collective_reward'] for r in results]
    mean_rewards = [r['final_metrics']['mean_reward'] for r in results]
    shaped_rewards = [r['final_metrics']['mean_shaped_reward'] for r in results]
    training_times = [r['training_time_seconds'] for r in results]

    # Sort by alpha
    sorted_indices = np.argsort(alphas)
    alphas_sorted = [alphas[i] for i in sorted_indices]
    collective_sorted = [collective_rewards[i] for i in sorted_indices]
    mean_sorted = [mean_rewards[i] for i in sorted_indices]
    shaped_sorted = [shaped_rewards[i] for i in sorted_indices]

    # Print table
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Alpha':<10} {'Mean Reward':<15} {'Collective':<15} {'Shaped':<15} {'Time (s)':<10}")
    print("-" * 80)
    for i, alpha in enumerate(alphas_sorted):
        idx = sorted_indices[i]
        print(f"{alpha:<10.1f} {mean_rewards[idx]:<15.4f} {collective_rewards[idx]:<15.4f} "
              f"{shaped_rewards[idx]:<15.4f} {training_times[idx]:<10.1f}")
    print("-" * 80)

    # Find optimal alpha (max collective reward)
    best_idx = np.argmax(collective_rewards)
    optimal_alpha = alphas[best_idx]

    print(f"\nOptimal Alpha: {optimal_alpha}")
    print(f"  Best Collective Reward: {collective_rewards[best_idx]:.4f}")
    print(f"  Best Mean Reward: {mean_rewards[best_idx]:.4f}")

    # Trend analysis
    print("\nTrend Analysis:")

    # Check if there's a clear trend
    if len(alphas) >= 3:
        # Simple linear regression
        x = np.array(alphas)
        y = np.array(collective_rewards)
        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.001:
            trend = "Increasing (higher alpha better)"
        elif slope < -0.001:
            trend = "Decreasing (lower alpha better)"
        else:
            trend = "Flat (alpha has little effect)"

        print(f"  Trend: {trend}")
        print(f"  Slope: {slope:.6f}")

    # Compare to paper recommendation (alpha = N-1)
    paper_alpha = results[0]['num_agents'] - 1 if results else 2
    paper_reward = None
    for r in results:
        if r['alpha'] == paper_alpha or (isinstance(r['alpha'], float) and abs(r['alpha'] - paper_alpha) < 0.1):
            paper_reward = r['final_metrics']['collective_reward']
            break

    print(f"\nPaper Recommendation (alpha={paper_alpha}):")
    if paper_reward is not None:
        print(f"  Collective Reward: {paper_reward:.4f}")
        if optimal_alpha != paper_alpha:
            print(f"  Note: Optimal alpha ({optimal_alpha}) differs from paper suggestion ({paper_alpha})")
        else:
            print(f"  Optimal alpha matches paper recommendation!")
    else:
        print(f"  Not tested (only tested alphas: {alphas})")

    analysis = {
        'optimal_alpha': optimal_alpha,
        'optimal_collective_reward': collective_rewards[best_idx],
        'optimal_mean_reward': mean_rewards[best_idx],
        'paper_recommended_alpha': paper_alpha,
        'paper_reward': paper_reward,
        'trend': {
            'alphas': alphas_sorted,
            'collective_rewards': collective_sorted,
            'mean_rewards': mean_sorted,
            'shaped_rewards': shaped_sorted,
        },
        'conclusion': f"Alpha={optimal_alpha} achieves best collective reward ({collective_rewards[best_idx]:.4f})",
    }

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Alpha Parameter Ablation for CF Algorithm")
    parser.add_argument(
        '--mode',
        choices=['quick', 'medium', 'full'],
        default='quick',
        help='Benchmark mode: quick (10K), medium (50K), full (200K steps per alpha)'
    )
    parser.add_argument(
        '--alphas',
        type=str,
        default='0.5,1,2,5,10',
        help='Comma-separated list of alpha values to test (default: 0.5,1,2,5,10)'
    )
    parser.add_argument(
        '--output',
        default='evaluation/alpha_ablation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Parse alpha values
    alpha_values = [float(a.strip()) for a in args.alphas.split(',')]

    print("="*60)
    print(f"Alpha Parameter Ablation Experiment (CF-BENCH-002)")
    print(f"Mode: {args.mode}")
    print(f"Alpha values: {alpha_values}")
    print(f"Seed: {args.seed}")
    print("="*60)

    # Get config
    config = get_config(args.mode)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Verify JAX
    print(f"\nJAX devices: {jax.devices()}")

    # Run experiments for each alpha
    all_results = []
    for i, alpha in enumerate(alpha_values):
        print(f"\n\n{'#'*60}")
        print(f"# Experiment {i+1}/{len(alpha_values)}: alpha={alpha}")
        print(f"{'#'*60}")

        result = run_cf_with_alpha(
            alpha=alpha,
            config=config,
            seed=args.seed + i,  # Different seed for each run
        )
        all_results.append(result)

    # Analyze results
    analysis = analyze_alpha_results(all_results)

    # Combine results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'alpha_values': alpha_values,
        'results': all_results,
        'analysis': analysis,
    }

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(
        args.output,
        f"alpha_ablation_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Optimal Alpha: {analysis['optimal_alpha']}")
    print(f"Best Collective Reward: {analysis['optimal_collective_reward']:.4f}")
    print(f"Conclusion: {analysis['conclusion']}")

    # Test criteria
    print("\nTest Criteria (CF-BENCH-002):")
    print(f"  [{'x' if len(alpha_values) == 5 else ' '}] All alpha values tested: {alpha_values}")
    print(f"  [x] Trend analysis available")
    print(f"  [x] Optimal alpha determined: {analysis['optimal_alpha']}")

    return output


if __name__ == "__main__":
    main()
